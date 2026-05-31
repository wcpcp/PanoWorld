from __future__ import annotations

from collections import OrderedDict
from typing import List, Literal, Sequence

import torch
from torch import nn

from .erp_geometry import (
    TokenLayout,
    build_da2_embeddings_per_image,
    build_features_per_image,
)


class ERPSphericalPosAdapter(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        hidden_dim: int = 512,
        feature_dim: int = 10,
        mode: Literal["paper", "extended"] = "extended",
        gate_init: float = 0.01,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim) if use_layernorm else nn.Identity()
        self.gate = nn.Parameter(torch.full((embed_dim,), float(gate_init)))

        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def _adapt_split(
        self,
        token_splits: Sequence[torch.Tensor],
        grid_thw: torch.Tensor,
        spatial_merge_size: int,
        *,
        token_layout: TokenLayout,
    ) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        for tokens, spherical_feat in zip(
            token_splits,
            build_features_per_image(
                grid_thw,
                spatial_merge_size,
                mode=self.mode,
                token_layout=token_layout,
                device=token_splits[0].device,
                dtype=token_splits[0].dtype,
            ),
        ):
            delta = self.norm(self.proj(spherical_feat))
            outputs.append(tokens + delta * self.gate.tanh())
        return outputs

    def _adapt_concat(
        self,
        tokens: torch.Tensor,
        grid_thw: torch.Tensor,
        spatial_merge_size: int,
        *,
        token_layout: TokenLayout,
    ) -> torch.Tensor:
        features = build_features_per_image(
            grid_thw,
            spatial_merge_size,
            mode=self.mode,
            token_layout=token_layout,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        token_splits = list(tokens.split([feat.shape[0] for feat in features], dim=0))
        adapted = self._adapt_split(
            token_splits,
            grid_thw,
            spatial_merge_size,
            token_layout=token_layout,
        )
        return torch.cat(adapted, dim=0)

    def forward(
        self,
        tokens: torch.Tensor | Sequence[torch.Tensor],
        grid_thw: torch.Tensor,
        spatial_merge_size: int,
        *,
        token_layout: TokenLayout = "merged",
    ) -> torch.Tensor | List[torch.Tensor]:
        if isinstance(tokens, torch.Tensor):
            return self._adapt_concat(
                tokens,
                grid_thw,
                spatial_merge_size,
                token_layout=token_layout,
            )
        return self._adapt_split(
            tokens,
            grid_thw,
            spatial_merge_size,
            token_layout=token_layout,
        )


class ERPSphericalCrossAttentionAdapter(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        hidden_dim: int = 512,
        feature_dim: int = 4,
        mode: Literal["paper", "extended"] = "paper",
        num_heads: int = 8,
        sphere_embed_type: Literal["fourier", "mlp"] = "fourier",
        gate_init: float = 0.01,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Cross-attention ERP adapter requires embed_dim ({embed_dim}) "
                f"to be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.mode = mode
        self.sphere_embed_type = sphere_embed_type
        self.q_norm = nn.LayerNorm(embed_dim) if use_layernorm else nn.Identity()
        self.kv_norm = nn.LayerNorm(embed_dim) if use_layernorm else nn.Identity()
        self.out_norm = nn.LayerNorm(embed_dim) if use_layernorm else nn.Identity()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.gate = nn.Parameter(torch.full((embed_dim,), float(gate_init)))
        if sphere_embed_type == "mlp":
            self.sphere_proj = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, embed_dim),
            )
            nn.init.zeros_(self.sphere_proj[-1].weight)
            nn.init.zeros_(self.sphere_proj[-1].bias)
        else:
            self.sphere_proj = None

        nn.init.zeros_(self.attn.out_proj.weight)
        self._fourier_cache: OrderedDict[tuple, tuple[torch.Tensor, ...]] = OrderedDict()
        self._fourier_cache_max_entries = 8

    def _fourier_cache_key(
        self,
        grid_thw: torch.Tensor,
        spatial_merge_size: int,
        token_layout: TokenLayout,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple:
        grid_key = tuple(tuple(int(v) for v in row) for row in grid_thw.detach().cpu().tolist())
        return (
            grid_key,
            int(spatial_merge_size),
            str(token_layout),
            self.embed_dim,
            str(device),
            str(dtype),
        )

    def _get_cached_fourier_embeddings(
        self,
        grid_thw: torch.Tensor,
        spatial_merge_size: int,
        *,
        token_layout: TokenLayout,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[torch.Tensor]:
        cache_key = self._fourier_cache_key(
            grid_thw,
            spatial_merge_size,
            token_layout,
            device,
            dtype,
        )
        cached = self._fourier_cache.get(cache_key)
        if cached is not None:
            self._fourier_cache.move_to_end(cache_key)
            return list(cached)

        embeddings = build_da2_embeddings_per_image(
            grid_thw,
            spatial_merge_size,
            embed_dim=self.embed_dim,
            token_layout=token_layout,
            device=device,
            dtype=dtype,
        )
        self._fourier_cache[cache_key] = tuple(emb.contiguous() for emb in embeddings)
        if len(self._fourier_cache) > self._fourier_cache_max_entries:
            self._fourier_cache.popitem(last=False)
        return embeddings

    def _build_sphere_embeddings(
        self,
        grid_thw: torch.Tensor,
        spatial_merge_size: int,
        *,
        token_layout: TokenLayout,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[torch.Tensor]:
        if self.sphere_embed_type == "fourier":
            return self._get_cached_fourier_embeddings(
                grid_thw,
                spatial_merge_size,
                token_layout=token_layout,
                device=device,
                dtype=dtype,
            )

        base_features = build_features_per_image(
            grid_thw,
            spatial_merge_size,
            mode=self.mode,
            token_layout=token_layout,
            device=device,
            dtype=dtype,
        )
        return [self.sphere_proj(feat) for feat in base_features]

    def _adapt_split(
        self,
        token_splits: Sequence[torch.Tensor],
        grid_thw: torch.Tensor,
        spatial_merge_size: int,
        *,
        token_layout: TokenLayout,
    ) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        sphere_embeddings = self._build_sphere_embeddings(
            grid_thw,
            spatial_merge_size,
            token_layout=token_layout,
            device=token_splits[0].device,
            dtype=token_splits[0].dtype,
        )
        for tokens, spherical_embed in zip(token_splits, sphere_embeddings):
            q = self.q_norm(tokens).unsqueeze(0)
            kv = self.kv_norm(spherical_embed).unsqueeze(0)
            delta, _ = self.attn(q, kv, kv, need_weights=False)
            delta = self.out_norm(delta.squeeze(0))
            outputs.append(tokens + delta * self.gate.tanh())
        return outputs

    def _adapt_concat(
        self,
        tokens: torch.Tensor,
        grid_thw: torch.Tensor,
        spatial_merge_size: int,
        *,
        token_layout: TokenLayout,
    ) -> torch.Tensor:
        sphere_embeddings = self._build_sphere_embeddings(
            grid_thw,
            spatial_merge_size,
            token_layout=token_layout,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        token_splits = list(tokens.split([feat.shape[0] for feat in sphere_embeddings], dim=0))
        adapted = self._adapt_split(
            token_splits,
            grid_thw,
            spatial_merge_size,
            token_layout=token_layout,
        )
        return torch.cat(adapted, dim=0)

    def forward(
        self,
        tokens: torch.Tensor | Sequence[torch.Tensor],
        grid_thw: torch.Tensor,
        spatial_merge_size: int,
        *,
        token_layout: TokenLayout = "merged",
    ) -> torch.Tensor | List[torch.Tensor]:
        if isinstance(tokens, torch.Tensor):
            return self._adapt_concat(
                tokens,
                grid_thw,
                spatial_merge_size,
                token_layout=token_layout,
            )
        return self._adapt_split(
            tokens,
            grid_thw,
            spatial_merge_size,
            token_layout=token_layout,
        )

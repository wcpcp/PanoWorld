from __future__ import annotations

import math
from typing import List, Literal, Tuple

import torch


TokenLayout = Literal["premerge", "merged"]


def merged_grid_shapes(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> List[Tuple[int, int, int]]:
    shapes: List[Tuple[int, int, int]] = []
    for t, h, w in grid_thw.tolist():
        merged_h = max(int(h) // spatial_merge_size, 1)
        merged_w = max(int(w) // spatial_merge_size, 1)
        shapes.append((int(t), merged_h, merged_w))
    return shapes


def token_grid_shapes(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    *,
    token_layout: TokenLayout,
) -> List[Tuple[int, int, int]]:
    if token_layout == "premerge":
        return [(int(t), int(h), int(w)) for t, h, w in grid_thw.tolist()]
    if token_layout == "merged":
        return merged_grid_shapes(grid_thw, spatial_merge_size)
    raise ValueError(f"Unsupported token layout: {token_layout}")


def build_erp_sincos_features(
    num_frames: int,
    grid_h: int,
    grid_w: int,
    *,
    mode: Literal["paper", "extended"] = "extended",
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ys = (torch.arange(grid_h, device=device, dtype=torch.float32) + 0.5) / float(grid_h)
    xs = (torch.arange(grid_w, device=device, dtype=torch.float32) + 0.5) / float(grid_w)

    pitch = (0.5 - ys) * math.pi
    yaw = (xs - 0.5) * (2.0 * math.pi)
    pitch_grid, yaw_grid = torch.meshgrid(pitch, yaw, indexing="ij")

    if mode == "paper":
        base = torch.stack(
            [
                torch.sin(yaw_grid),
                torch.cos(yaw_grid),
                torch.sin(pitch_grid),
                torch.cos(pitch_grid),
            ],
            dim=-1,
        ).reshape(grid_h * grid_w, -1)
    elif mode == "extended":
        area_weight = torch.cos(pitch_grid).clamp_min(1e-4)
        base = torch.stack(
            [
                torch.sin(yaw_grid),
                torch.cos(yaw_grid),
                torch.sin(pitch_grid),
                torch.cos(pitch_grid),
                torch.sin(2.0 * yaw_grid),
                torch.cos(2.0 * yaw_grid),
                torch.sin(2.0 * pitch_grid),
                torch.cos(2.0 * pitch_grid),
                area_weight,
                pitch_grid / (0.5 * math.pi),
            ],
            dim=-1,
        ).reshape(grid_h * grid_w, -1)
    else:
        raise ValueError(f"Unsupported ERP position mode: {mode}")

    if num_frames > 1:
        base = base.repeat(num_frames, 1)

    return base.to(device=device, dtype=dtype)


def build_da2_spherical_embedding(
    num_frames: int,
    grid_h: int,
    grid_w: int,
    *,
    embed_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ys = (torch.arange(grid_h, device=device, dtype=torch.float32) + 0.5) / float(grid_h)
    xs = (torch.arange(grid_w, device=device, dtype=torch.float32) + 0.5) / float(grid_w)

    theta = ys * math.pi
    phi = xs * (2.0 * math.pi)
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="ij")
    angle_field = torch.stack([phi_grid, theta_grid], dim=-1).reshape(grid_h * grid_w, 2)

    features_per_band = 4
    num_bands = max(1, math.ceil(embed_dim / features_per_band))
    max_freq = max(float(max(grid_h, grid_w)), 1.0)
    scales = 2.0 ** torch.linspace(
        0.0,
        math.log2(max_freq),
        steps=num_bands,
        device=device,
        dtype=torch.float32,
    )

    expanded = angle_field.unsqueeze(-1) * scales
    embedded = torch.stack([expanded.sin(), expanded.cos()], dim=-1).reshape(
        grid_h * grid_w,
        -1,
    )
    if embedded.shape[-1] > embed_dim:
        embedded = embedded[:, :embed_dim]
    elif embedded.shape[-1] < embed_dim:
        embedded = torch.nn.functional.pad(embedded, (0, embed_dim - embedded.shape[-1]))

    if num_frames > 1:
        embedded = embedded.repeat(num_frames, 1)

    return embedded.to(device=device, dtype=dtype)


def build_features_per_image(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    *,
    mode: Literal["paper", "extended"] = "extended",
    token_layout: TokenLayout = "merged",
    device: torch.device,
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    return [
        build_erp_sincos_features(
            t,
            h,
            w,
            mode=mode,
            device=device,
            dtype=dtype,
        )
        for t, h, w in token_grid_shapes(
            grid_thw,
            spatial_merge_size,
            token_layout=token_layout,
        )
    ]


def build_da2_embeddings_per_image(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    *,
    embed_dim: int,
    token_layout: TokenLayout = "merged",
    device: torch.device,
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    return [
        build_da2_spherical_embedding(
            t,
            h,
            w,
            embed_dim=embed_dim,
            device=device,
            dtype=dtype,
        )
        for t, h, w in token_grid_shapes(
            grid_thw,
            spatial_merge_size,
            token_layout=token_layout,
        )
    ]

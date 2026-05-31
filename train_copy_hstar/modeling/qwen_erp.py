from __future__ import annotations

from types import MethodType
from typing import List

import torch

from .vision_adapter import ERPSphericalCrossAttentionAdapter, ERPSphericalPosAdapter


def _is_tensor_sequence(value) -> bool:
    return isinstance(value, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in value)


def _parse_csv(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip().lower() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(part).strip().lower() for part in value if str(part).strip()]
    return [str(value).strip().lower()]


def _get_visual_module(top_model):
    if hasattr(top_model, "visual"):
        return top_model.visual
    if hasattr(top_model, "model") and hasattr(top_model.model, "visual"):
        return top_model.model.visual
    return None


def _get_text_hidden_size(top_model) -> int:
    text_config = getattr(getattr(top_model, "config", None), "text_config", None)
    if text_config is not None and hasattr(text_config, "hidden_size"):
        return int(text_config.hidden_size)

    language_model = getattr(getattr(top_model, "model", None), "language_model", None)
    lm_config = getattr(language_model, "config", None)
    if lm_config is not None and hasattr(lm_config, "hidden_size"):
        return int(lm_config.hidden_size)

    hidden_size = getattr(getattr(top_model, "config", None), "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)

    raise ValueError("Unable to infer text hidden size for ERP adapter")


def _get_visual_hidden_size(visual) -> int:
    visual_config = getattr(visual, "config", None)
    hidden_size = getattr(visual_config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Unable to infer visual hidden size for ERP adapter")
    return int(hidden_size)


def _make_adapter(
    *,
    embed_dim: int,
    hidden_dim: int,
    feature_dim: int,
    pos_mode: str,
    adapter_type: str,
    cross_attn_embed_type: str,
    num_heads: int,
    gate_init: float,
    use_layernorm: bool,
) -> ERPSphericalPosAdapter | ERPSphericalCrossAttentionAdapter:
    if adapter_type == "additive":
        return ERPSphericalPosAdapter(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            mode=pos_mode,
            gate_init=gate_init,
            use_layernorm=use_layernorm,
        )
    if adapter_type == "da2_cross_attn":
        return ERPSphericalCrossAttentionAdapter(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            mode=pos_mode,
            num_heads=num_heads,
            sphere_embed_type=cross_attn_embed_type,
            gate_init=gate_init,
            use_layernorm=use_layernorm,
        )
    raise ValueError(f"Unsupported ERP adapter_type: {adapter_type}")


def attach_erp_adapter(top_model, erp_cfg) -> None:
    if not getattr(erp_cfg, "enabled", False):
        return
    if hasattr(top_model, "_pano_erp_attached"):
        return

    hidden_dim = int(getattr(erp_cfg, "hidden_dim", 512))
    num_heads = int(getattr(erp_cfg, "num_heads", 8))
    gate_init = float(getattr(erp_cfg, "gate_init", 0.01))
    pos_mode = str(getattr(erp_cfg, "pos_mode", "paper")).lower()
    adapter_type = str(getattr(erp_cfg, "adapter_type", "additive")).lower()
    cross_attn_embed_type = str(getattr(erp_cfg, "cross_attn_embed_type", "fourier")).lower()
    stages = set(_parse_csv(getattr(erp_cfg, "stage", "output")))
    target = str(getattr(erp_cfg, "target", "pooler")).lower()
    use_layernorm = bool(getattr(erp_cfg, "use_layernorm", True))

    feature_dim = 4 if pos_mode == "paper" else 10
    valid_stages = {"patch", "merger", "output"}
    valid_targets = {"pooler", "deepstack", "both"}
    valid_adapter_types = {"additive", "da2_cross_attn"}
    valid_cross_attn_embed_types = {"fourier", "mlp"}

    if not stages:
        stages = {"output"}

    unknown_stages = stages - valid_stages
    if unknown_stages:
        raise ValueError(f"Unsupported ERP stage(s): {sorted(unknown_stages)}")
    if target not in valid_targets:
        raise ValueError(f"Unsupported ERP target: {target}")
    if adapter_type not in valid_adapter_types:
        raise ValueError(f"Unsupported ERP adapter_type: {adapter_type}")
    if cross_attn_embed_type not in valid_cross_attn_embed_types:
        raise ValueError(f"Unsupported ERP cross_attn_embed_type: {cross_attn_embed_type}")

    visual = _get_visual_module(top_model)
    if visual is None:
        raise ValueError("Unable to locate visual module for ERP adapter attachment")

    if "output" in stages:
        top_model.erp_output_adapter = _make_adapter(
            embed_dim=_get_text_hidden_size(top_model),
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            pos_mode=pos_mode,
            adapter_type=adapter_type,
            cross_attn_embed_type=cross_attn_embed_type,
            num_heads=num_heads,
            gate_init=gate_init,
            use_layernorm=use_layernorm,
        )
    if "patch" in stages:
        top_model.erp_patch_adapter = _make_adapter(
            embed_dim=_get_visual_hidden_size(visual),
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            pos_mode=pos_mode,
            adapter_type=adapter_type,
            cross_attn_embed_type=cross_attn_embed_type,
            num_heads=num_heads,
            gate_init=gate_init,
            use_layernorm=use_layernorm,
        )
    if "merger" in stages:
        top_model.erp_merger_adapter = _make_adapter(
            embed_dim=_get_visual_hidden_size(visual),
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            pos_mode=pos_mode,
            adapter_type=adapter_type,
            cross_attn_embed_type=cross_attn_embed_type,
            num_heads=num_heads,
            gate_init=gate_init,
            use_layernorm=use_layernorm,
        )

    if stages & {"patch", "merger"} and not hasattr(visual, "_pano_origin_forward"):
        visual._pano_origin_forward = visual.forward

        def visual_forward_with_ctx(this, hidden_states, grid_thw=None, **kwargs):
            this._pano_current_grid_thw = grid_thw
            try:
                return this._pano_origin_forward(hidden_states, grid_thw=grid_thw, **kwargs)
            finally:
                this._pano_current_grid_thw = None

        visual.forward = MethodType(visual_forward_with_ctx, visual)

    if "patch" in stages:
        if not hasattr(visual, "patch_embed"):
            raise ValueError("ERP patch stage requested but visual.patch_embed is missing")
        if not hasattr(visual.patch_embed, "_pano_origin_forward"):
            visual.patch_embed._pano_origin_forward = visual.patch_embed.forward

            def patch_embed_with_erp(this, hidden_states):
                outputs = this._pano_origin_forward(hidden_states)
                grid_thw = getattr(visual, "_pano_current_grid_thw", None)
                if grid_thw is None:
                    return outputs
                return top_model.erp_patch_adapter(
                    outputs,
                    grid_thw,
                    int(visual.spatial_merge_size),
                    token_layout="premerge",
                )

            visual.patch_embed.forward = MethodType(patch_embed_with_erp, visual.patch_embed)

    if "merger" in stages:
        if not hasattr(visual, "merger"):
            raise ValueError("ERP merger stage requested but visual.merger is missing")
        if not hasattr(visual.merger, "_pano_origin_forward"):
            visual.merger._pano_origin_forward = visual.merger.forward

            def merger_with_erp(this, hidden_states):
                grid_thw = getattr(visual, "_pano_current_grid_thw", None)
                if grid_thw is not None:
                    hidden_states = top_model.erp_merger_adapter(
                        hidden_states,
                        grid_thw,
                        int(visual.spatial_merge_size),
                        token_layout="premerge",
                    )
                return this._pano_origin_forward(hidden_states)

            visual.merger.forward = MethodType(merger_with_erp, visual.merger)

    if "output" in stages:
        model_core = getattr(top_model, "model", None)
        if model_core is None or not hasattr(model_core, "get_image_features"):
            raise ValueError("ERP output stage requested but model.get_image_features is missing")
        if not hasattr(model_core, "_origin_get_image_features"):
            model_core._origin_get_image_features = model_core.get_image_features

            def get_image_features_with_erp(this, pixel_values, image_grid_thw=None, **kwargs):
                try:
                    outputs = this._origin_get_image_features(
                        pixel_values,
                        image_grid_thw=image_grid_thw,
                        **kwargs,
                    )
                except TypeError as exc:
                    if "return_dict" not in str(exc):
                        raise
                    safe_kwargs = {k: v for k, v in kwargs.items() if k != "return_dict"}
                    outputs = this._origin_get_image_features(
                        pixel_values,
                        image_grid_thw=image_grid_thw,
                        **safe_kwargs,
                    )

                if image_grid_thw is None:
                    return outputs

                spatial_merge_size = int(this.visual.spatial_merge_size)
                if hasattr(outputs, "pooler_output"):
                    if target in {"pooler", "both"}:
                        outputs.pooler_output = top_model.erp_output_adapter(
                            outputs.pooler_output,
                            image_grid_thw,
                            spatial_merge_size,
                            token_layout="merged",
                        )
                    if target in {"deepstack", "both"} and getattr(outputs, "deepstack_features", None):
                        outputs.deepstack_features = [
                            top_model.erp_output_adapter(
                                feat,
                                image_grid_thw,
                                spatial_merge_size,
                                token_layout="merged",
                            )
                            for feat in outputs.deepstack_features
                        ]
                    return outputs

                if _is_tensor_sequence(outputs):
                    adapted = (
                        top_model.erp_output_adapter(
                            outputs,
                            image_grid_thw,
                            spatial_merge_size,
                            token_layout="merged",
                        )
                        if target in {"pooler", "both"}
                        else list(outputs)
                    )
                    return tuple(adapted) if isinstance(outputs, tuple) else adapted

                if isinstance(outputs, tuple) and outputs:
                    pooler_output = outputs[0]
                    deepstack_features = outputs[1] if len(outputs) > 1 else None
                    if target in {"pooler", "both"}:
                        pooler_output = top_model.erp_output_adapter(
                            pooler_output,
                            image_grid_thw,
                            spatial_merge_size,
                            token_layout="merged",
                        )
                    if target in {"deepstack", "both"} and deepstack_features is not None:
                        deepstack_features = [
                            top_model.erp_output_adapter(
                                feat,
                                image_grid_thw,
                                spatial_merge_size,
                                token_layout="merged",
                            )
                            for feat in deepstack_features
                        ]

                    head = [tuple(pooler_output) if isinstance(pooler_output, list) else pooler_output]
                    if len(outputs) > 1:
                        head.append(deepstack_features)
                    return tuple(head + list(outputs[2:]))

                return outputs

            model_core.get_image_features = MethodType(get_image_features_with_erp, model_core)

    top_model._pano_erp_attached = True

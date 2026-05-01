import torch

from comfy.model_detection import detect_unet_config, model_config_from_unet_config
import comfy.supported_models


def _make_longcat_comfyui_sd():
    """Minimal ComfyUI-format state dict for pre-converted LongCat-Image weights."""
    sd = {}
    H = 32  # Reduce hidden state dimension to reduce memory usage
    C_IN = 16
    C_CTX = 3584

    sd["img_in.weight"] = torch.empty(H, C_IN * 4)
    sd["img_in.bias"] = torch.empty(H)
    sd["txt_in.weight"] = torch.empty(H, C_CTX)
    sd["txt_in.bias"] = torch.empty(H)

    sd["time_in.in_layer.weight"] = torch.empty(H, 256)
    sd["time_in.in_layer.bias"] = torch.empty(H)
    sd["time_in.out_layer.weight"] = torch.empty(H, H)
    sd["time_in.out_layer.bias"] = torch.empty(H)

    sd["final_layer.adaLN_modulation.1.weight"] = torch.empty(2 * H, H)
    sd["final_layer.adaLN_modulation.1.bias"] = torch.empty(2 * H)
    sd["final_layer.linear.weight"] = torch.empty(C_IN * 4, H)
    sd["final_layer.linear.bias"] = torch.empty(C_IN * 4)

    for i in range(19):
        sd[f"double_blocks.{i}.img_attn.norm.key_norm.weight"] = torch.empty(128)
        sd[f"double_blocks.{i}.img_attn.qkv.weight"] = torch.empty(3 * H, H)
        sd[f"double_blocks.{i}.img_mod.lin.weight"] = torch.empty(H, H)
    for i in range(38):
        sd[f"single_blocks.{i}.modulation.lin.weight"] = torch.empty(H, H)

    return sd


def _make_flux_schnell_comfyui_sd():
    """Minimal ComfyUI-format state dict for standard Flux Schnell."""
    sd = {}
    H = 32  # Reduce hidden state dimension to reduce memory usage
    C_IN = 16

    sd["img_in.weight"] = torch.empty(H, C_IN * 4)
    sd["img_in.bias"] = torch.empty(H)
    sd["txt_in.weight"] = torch.empty(H, 4096)
    sd["txt_in.bias"] = torch.empty(H)

    sd["double_blocks.0.img_attn.norm.key_norm.weight"] = torch.empty(128)
    sd["double_blocks.0.img_attn.qkv.weight"] = torch.empty(3 * H, H)
    sd["double_blocks.0.img_mod.lin.weight"] = torch.empty(H, H)

    for i in range(19):
        sd[f"double_blocks.{i}.img_attn.norm.key_norm.weight"] = torch.empty(128)
    for i in range(38):
        sd[f"single_blocks.{i}.modulation.lin.weight"] = torch.empty(H, H)

    return sd


class TestModelDetection:
    """Verify that model detection selects the most specific model regardless of
    the ordering of entries in ``comfy.supported_models.models``."""

    def test_longcat_detection_is_order_independent(self):
        """Detection must pick LongCatImage over FluxSchnell regardless of
        their relative order in the models list, because LongCatImage has a
        strictly more specific ``unet_config``."""
        original_models = comfy.supported_models.models
        sd = _make_longcat_comfyui_sd()
        unet_config = detect_unet_config(sd, "")

        try:
            for ordering in ("longcat_first", "schnell_first"):
                models = list(original_models)
                longcat = next(m for m in models if m.__name__ == "LongCatImage")
                schnell = next(m for m in models if m.__name__ == "FluxSchnell")
                models.remove(longcat)
                models.remove(schnell)
                if ordering == "longcat_first":
                    models.extend([longcat, schnell])
                else:
                    models.extend([schnell, longcat])
                comfy.supported_models.models = models

                model_config = model_config_from_unet_config(unet_config, sd)
                assert model_config is not None
                assert type(model_config).__name__ == "LongCatImage", (
                    f"Expected LongCatImage with ordering={ordering}, "
                    f"got {type(model_config).__name__}"
                )
        finally:
            comfy.supported_models.models = original_models

    def test_longcat_comfyui_detected_as_longcat(self):
        sd = _make_longcat_comfyui_sd()
        unet_config = detect_unet_config(sd, "")
        assert unet_config is not None
        assert unet_config["image_model"] == "flux"
        assert unet_config["context_in_dim"] == 3584
        assert unet_config["vec_in_dim"] is None
        assert unet_config["guidance_embed"] is False
        assert unet_config["txt_ids_dims"] == [1, 2]

        model_config = model_config_from_unet_config(unet_config, sd)
        assert model_config is not None
        assert type(model_config).__name__ == "LongCatImage"

    def test_longcat_comfyui_keys_pass_through_unchanged(self):
        """Pre-converted weights should not be transformed by process_unet_state_dict."""
        sd = _make_longcat_comfyui_sd()
        unet_config = detect_unet_config(sd, "")
        model_config = model_config_from_unet_config(unet_config, sd)

        processed = model_config.process_unet_state_dict(dict(sd))
        assert "img_in.weight" in processed
        assert "txt_in.weight" in processed
        assert "time_in.in_layer.weight" in processed
        assert "final_layer.linear.weight" in processed

    def test_flux_schnell_comfyui_detected_as_flux_schnell(self):
        sd = _make_flux_schnell_comfyui_sd()
        unet_config = detect_unet_config(sd, "")
        assert unet_config is not None
        assert unet_config["image_model"] == "flux"
        assert unet_config["context_in_dim"] == 4096
        assert unet_config["txt_ids_dims"] == []

        model_config = model_config_from_unet_config(unet_config, sd)
        assert model_config is not None
        assert type(model_config).__name__ == "FluxSchnell"

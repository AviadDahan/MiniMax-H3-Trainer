"""Loading MiniMax-H3 components.

The H3 repository ships two packagings of the same weights. The trainer only
speaks the diffusers-native flat layout at the repository root::

    MiniMax-H3/
        transformer/        FL2VA -- text / first frame / first+last frame
        transformer_ref/    Ref2VA -- omni-reference (images, video, audio)
        text_encoder/       Qwen3-VL-32B
        tokenizer/  processor/
        vae/                H3-VisualVAE   (16x spatial, 4x temporal, 24 channels)
        audio_vae/          H3-AudioVAE    (32 kHz stereo, 40 Hz latent grid)

(The ``FL2VA/`` and ``Ref2VA/`` directories in the HF repo are the original
MiniMax packaging with custom modelling code -- same weights, unusable from
diffusers, and 144GB each. ``scripts/download_model.sh`` skips them.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from h3_trainer import logger
from h3_trainer.constants import MINIMAX_H3_TEXT_ENCODER_LAYER

TRANSFORMER_SUBFOLDERS = {"fl2va": "transformer", "ref2va": "transformer_ref"}


def resolve_model_path(model_path: str | Path) -> Path:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")
    if not (path / "model_index.json").exists() and not (path / "transformer").exists():
        raise FileNotFoundError(
            f"{path} does not look like a diffusers MiniMax-H3 checkout (no model_index.json and no "
            f"transformer/ subfolder). Run scripts/download_model.sh."
        )
    return path


def load_transformer(
    model_path: str | Path,
    variant: str = "fl2va",
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str | None = None,
    quantization: str = "none",
):
    """Load the H3 transformer for one task variant."""
    from diffusers import MiniMaxH3Transformer3DModel

    model_path = resolve_model_path(model_path)
    if variant not in TRANSFORMER_SUBFOLDERS:
        raise ValueError(f"variant must be one of {sorted(TRANSFORMER_SUBFOLDERS)}, got {variant!r}")
    subfolder = TRANSFORMER_SUBFOLDERS[variant]
    if not (model_path / subfolder).exists():
        raise FileNotFoundError(
            f"{model_path / subfolder} is missing -- the {variant} transformer was not downloaded."
        )

    logger.info("Loading %s transformer from %s", variant, model_path / subfolder)
    if quantization != "none":
        from h3_trainer.quantization import load_quantized_transformer

        model = load_quantized_transformer(model_path, subfolder, quantization, dtype)
    else:
        model = MiniMaxH3Transformer3DModel.from_pretrained(
            model_path, subfolder=subfolder, torch_dtype=dtype
        )
    if device is not None:
        model = model.to(device)
    return model


#: Modules that must stay on the primary device when the transformer is sharded.
#: H3's forward uses the packed layout's index vectors for ``index_select`` against
#: its own inputs and outputs, outside any accelerate hook -- so every module that
#: touches them has to sit on the device those index tensors live on. Only the 50
#: transformer blocks are safe to spread; activations flow between them and come
#: back before the output heads run.
TRANSFORMER_PINNED_MODULES = (
    "proj_in",
    "audio_proj_in",
    "context_embedder",
    "time_proj",
    "time_embedder",
    "rope",
    "token_refiner",
    "norm_out",
    "proj_out",
    "audio_proj_out",
)


def build_transformer_device_map(
    num_layers: int,
    devices: list[int] | None = None,
    primary: int = 0,
) -> dict[str, int]:
    """Spread the transformer blocks over GPUs, pinning everything else.

    A plain ``device_map="auto"`` splits the model wherever it likes, which puts
    the output projection on the last GPU while the index tensors are still on the
    first -- and ``index_select`` then fails on a device mismatch. Distributing
    only ``transformer_blocks.*`` keeps that from happening while still cutting
    per-GPU weight memory by roughly the number of GPUs.
    """
    if devices is None:
        devices = list(range(torch.cuda.device_count()))
    if not devices:
        raise RuntimeError("No CUDA devices available")
    if primary not in devices:
        devices = [primary, *devices]

    device_map = {name: primary for name in TRANSFORMER_PINNED_MODULES}
    per_device = max(1, (num_layers + len(devices) - 1) // len(devices))
    for index in range(num_layers):
        device_map[f"transformer_blocks.{index}"] = devices[min(index // per_device, len(devices) - 1)]
    return device_map


def load_sharded_transformer(
    model_path: str | Path,
    variant: str = "fl2va",
    dtype: torch.dtype = torch.bfloat16,
    devices: list[int] | None = None,
    primary: int = 0,
):
    """Load the transformer in full precision, blocks spread across GPUs."""
    import json

    from diffusers import MiniMaxH3Transformer3DModel

    model_path = resolve_model_path(model_path)
    subfolder = TRANSFORMER_SUBFOLDERS[variant]
    with (model_path / subfolder / "config.json").open() as handle:
        num_layers = int(json.load(handle)["num_layers"])

    device_map = build_transformer_device_map(num_layers, devices, primary)
    placement = {}
    for name, device in device_map.items():
        placement.setdefault(device, []).append(name)
    logger.info(
        "Sharding %d transformer blocks over GPUs %s (heads pinned to cuda:%d)",
        num_layers,
        sorted({d for k, d in device_map.items() if k.startswith("transformer_blocks")}),
        primary,
    )
    return MiniMaxH3Transformer3DModel.from_pretrained(
        model_path, subfolder=subfolder, torch_dtype=dtype, device_map=device_map
    )


def load_video_vae(model_path: str | Path, device: torch.device | str | None = None):
    """H3-VisualVAE. Kept in fp32: its convolution path is not bf16-safe."""
    from diffusers import AutoencoderKLMiniMaxH3

    model_path = resolve_model_path(model_path)
    vae = AutoencoderKLMiniMaxH3.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
    vae.eval().requires_grad_(False)
    return vae.to(device) if device is not None else vae


def load_audio_vae(model_path: str | Path, device: torch.device | str | None = None):
    """H3-AudioVAE (mono model applied per stereo channel), fp32."""
    from diffusers import AutoencoderKLMiniMaxH3Audio

    model_path = resolve_model_path(model_path)
    vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
        model_path, subfolder="audio_vae", torch_dtype=torch.float32
    )
    vae.eval().requires_grad_(False)
    return vae.to(device) if device is not None else vae


@dataclass
class TextEncoderBundle:
    """Qwen3-VL text encoder plus the tokenizer/processor H3 pairs it with."""

    model: object
    tokenizer: object
    processor: object
    device: torch.device
    dtype: torch.dtype

    @torch.no_grad()
    def encode(self, prompt: str) -> torch.Tensor:
        """Prompt -> layer-50 hidden states, ``(num_tokens, text_dim)`` in bf16.

        H3 does not use the encoder's final layer: the transformer's
        ``context_embedder`` is trained against hidden state 50 specifically, so
        the layer index is part of the model contract, not a tuning knob.
        """
        input_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)
        mm_token_type_ids = torch.tensor(
            self.processor.create_mm_token_type_ids(input_ids.tolist()), device=self.device
        )
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
            output_hidden_states=True,
        )
        return outputs.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER][0].to(torch.bfloat16).cpu()


def load_text_encoder(
    model_path: str | Path,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict | None = "auto",
) -> TextEncoderBundle:
    """Load H3's Qwen3-VL conditioner.

    It is a 32B model -- ~64GB in bf16 -- so it does not fit on a single 48GB
    card. ``device_map="auto"`` spreads it across whatever GPUs are visible and
    spills to CPU if needed. That is fine here: the conditioner only runs during
    preprocessing, and its output is cached.

    Pass ``device_map=None`` to force the whole model onto one device (correct on
    80GB+ cards, and the only option if accelerate is unavailable).
    """
    from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    model_path = resolve_model_path(model_path)
    logger.info(
        "Loading Qwen3-VL text encoder from %s (device_map=%s)", model_path / "text_encoder", device_map
    )
    kwargs: dict = {"torch_dtype": dtype}
    if device_map is not None:
        # A dict here is a max_memory budget (per-GPU caps), not a placement map;
        # it is how the caller keeps the conditioner off a particular card.
        if isinstance(device_map, dict) and all(isinstance(k, int) for k in device_map):
            kwargs["device_map"] = "auto"
            kwargs["max_memory"] = device_map
        else:
            kwargs["device_map"] = device_map
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path / "text_encoder", **kwargs)
    model.eval().requires_grad_(False)
    if device_map is None:
        model = model.to(device)
        input_device = torch.device(device)
    else:
        # With a sharded model, inputs belong on whichever device holds the
        # embedding layer; accelerate moves activations from there.
        input_device = next(model.parameters()).device
        logger.info("Text encoder sharded across: %s", sorted({str(p.device) for p in model.parameters()}))

    return TextEncoderBundle(
        model=model,
        tokenizer=Qwen2TokenizerFast.from_pretrained(model_path / "tokenizer"),
        processor=Qwen3VLProcessor.from_pretrained(model_path / "processor"),
        device=input_device,
        dtype=dtype,
    )


def enable_gradient_checkpointing(model: torch.nn.Module, use_deepspeed: bool) -> None:
    """Turn on activation checkpointing with the implementation the backend needs.

    Under ZeRO-3, torch's checkpointing (reentrant and non-reentrant alike)
    recomputes against parameters DeepSpeed has already re-partitioned, so the
    recompute sees shape-[0] tensors and crashes. DeepSpeed's own implementation
    cooperates with the gather hooks.
    """
    if use_deepspeed:
        import deepspeed

        model.enable_gradient_checkpointing(deepspeed.checkpointing.checkpoint)
    else:
        model.enable_gradient_checkpointing()

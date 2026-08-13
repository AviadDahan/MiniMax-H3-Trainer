"""Quantizing the frozen base transformer so a LoRA fits on a smaller card.

H3 is 33B parameters: 66GB in bf16, which does not fit on a 48GB GPU alongside
activations. There are two ways out, and they exclude each other:

* **Shard** the bf16 weights across GPUs with DeepSpeed ZeRO-3. Correct, exact,
  and on a machine without NVLink it spends most of its time all-gathering
  parameters over PCIe.
* **Quantize** the frozen base to 4- or 8-bit and keep full replicas under DDP.
  ~17GB (NF4) or ~33GB (int8) per GPU, no cross-GPU parameter traffic at all.
  The LoRA adapters stay in bf16, so what is being *learned* is never quantized
  -- only the frozen weights it sits on top of.

Quantization is not free: NF4 base weights measurably change the model's output
distribution, so a LoRA trained on an NF4 base and applied to a bf16 base is
slightly off-recipe. For style/character adapters this is well within the noise;
for anything precision-critical, use ZeRO-3.
"""

from __future__ import annotations

from pathlib import Path

import torch

from h3_trainer import logger

QUANTIZATION_MODES = ("none", "int8-quanto", "fp8-quanto", "nf4-bnb", "int8-bnb")

#: Modules left in full precision: the input/output projections and the time
#: embedding. They are small, everything downstream is sensitive to them, and
#: keeping them exact costs well under a gigabyte.
#:
#: ``adaln_proj`` is deliberately **not** on this list even though it is
#: precision-sensitive in the same way. H3 puts roughly a third of its parameters
#: in the per-block AdaLN modulation branches, so excluding them from
#: quantization leaves ~26GB in bf16 and defeats the entire point -- an "NF4"
#: load that still needs a 48GB card.
SKIP_MODULES = (
    "proj_in",
    "proj_out",
    "audio_proj_in",
    "audio_proj_out",
    "context_embedder",
    "time_embedder",
    "time_proj",
    "norm_out",
)


def load_quantized_transformer(
    model_path: str | Path,
    subfolder: str,
    quantization: str,
    dtype: torch.dtype = torch.bfloat16,
):
    """Load the H3 transformer with quantized frozen weights."""
    if quantization not in QUANTIZATION_MODES:
        raise ValueError(f"quantization must be one of {QUANTIZATION_MODES}, got {quantization!r}")
    if quantization.endswith("-bnb"):
        return _load_bnb(model_path, subfolder, quantization, dtype)
    return _load_quanto(model_path, subfolder, quantization, dtype)


def _load_bnb(model_path: str | Path, subfolder: str, quantization: str, dtype: torch.dtype):
    from diffusers import BitsAndBytesConfig, MiniMaxH3Transformer3DModel

    if quantization == "nf4-bnb":
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=list(SKIP_MODULES),
        )
    else:
        config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=list(SKIP_MODULES))

    logger.info("Loading transformer with bitsandbytes %s", quantization)
    return MiniMaxH3Transformer3DModel.from_pretrained(
        model_path,
        subfolder=subfolder,
        quantization_config=config,
        torch_dtype=dtype,
    )


def _load_quanto(model_path: str | Path, subfolder: str, quantization: str, dtype: torch.dtype):
    from diffusers import MiniMaxH3Transformer3DModel

    try:
        from optimum.quanto import freeze, qfloat8, qint8, quantize
    except ImportError as exc:  # pragma: no cover
        raise ImportError("optimum-quanto is required for *-quanto quantization: pip install optimum-quanto") from exc

    weights = {"int8-quanto": qint8, "fp8-quanto": qfloat8}[quantization]
    logger.info("Loading transformer in %s then quantizing with optimum-quanto", dtype)
    model = MiniMaxH3Transformer3DModel.from_pretrained(model_path, subfolder=subfolder, torch_dtype=dtype)
    quantize(model, weights=weights, exclude=[f"*{name}*" for name in SKIP_MODULES])
    freeze(model)
    return model


def estimate_memory_gb(num_parameters: int, quantization: str) -> float:
    """Rough weight footprint, for the startup memory report."""
    bytes_per_parameter = {
        "none": 2.0,
        "int8-quanto": 1.0,
        "int8-bnb": 1.0,
        "fp8-quanto": 1.0,
        "nf4-bnb": 0.55,  # 4 bits plus quantization constants
    }[quantization]
    return num_parameters * bytes_per_parameter / 1e9

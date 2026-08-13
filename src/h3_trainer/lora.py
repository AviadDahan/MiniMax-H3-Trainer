"""LoRA wiring for the H3 transformer, and export into the community layout.

Two things here are worth reading before changing anything:

**Target-module verification.** PEFT only raises when *none* of the requested
target modules match. A name that matches nothing is silently dropped, so a
plausible-looking target list can train far less than it claims. (The public
MiniMax-H3 reference trainer targets ``to_qkv`` -- a name from the original
MiniMax packaging that does not exist in the diffusers conversion -- and
therefore never adapts Q/K/V at all, while still reporting a healthy parameter
count from ``to_out.0`` plus the time embedder's ``linear_1``/``linear_2``.)
:func:`verify_target_modules` turns that class of mistake into a startup error.

**Export re-fusion.** Diffusers splits H3's attention into ``to_q``/``to_k``/
``to_v``; the original checkpoint -- and therefore every published H3 LoRA and
every ComfyUI loader -- uses a single fused ``qkv_proj``. Three rank-r updates
concatenated along the output axis are exactly one rank-3r update with a
block-diagonal B, so the conversion is lossless. See :func:`fuse_qkv_lora`.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors.torch import save_file

from h3_trainer import logger

if TYPE_CHECKING:
    from h3_trainer.config import LoraConfig as H3LoraConfig


def _module_matches(module_name: str, target: str) -> bool:
    """PEFT's matching rule: exact name, or a dotted suffix of the module path."""
    return module_name == target or module_name.endswith("." + target)


def verify_target_modules(model: torch.nn.Module, target_modules: list[str]) -> dict[str, int]:
    """Count how many modules each target matches; raise if any matches none."""
    counts = {target: 0 for target in target_modules}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        for target in target_modules:
            if _module_matches(name, target):
                counts[target] += 1
    unmatched = [target for target, count in counts.items() if count == 0]
    if unmatched:
        available = sorted(
            {
                re.sub(r"\.\d+\.", ".N.", name)
                for name, module in model.named_modules()
                if isinstance(module, torch.nn.Linear)
            }
        )
        raise ValueError(
            f"lora.target_modules entries matched no Linear layer: {unmatched}. PEFT would drop them "
            f"silently and train a smaller adapter than you asked for.\nLinear modules in this model:\n  "
            + "\n  ".join(available)
        )
    return counts


def apply_lora(model: torch.nn.Module, config: H3LoraConfig) -> torch.nn.Module:
    """Attach a PEFT LoRA adapter to the transformer and freeze everything else."""
    from peft import LoraConfig as PeftLoraConfig

    counts = verify_target_modules(model, config.target_modules)
    logger.info(
        "LoRA targets matched: %s",
        ", ".join(f"{target}x{count}" for target, count in counts.items()),
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.add_adapter(
        PeftLoraConfig(
            r=config.rank,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            target_modules=list(config.target_modules),
            init_lora_weights=config.init_lora_weights,
        )
    )
    return model


def set_trainable(model: torch.nn.Module, training_mode: str, lora_config: H3LoraConfig | None = None) -> None:
    """Freeze/unfreeze according to ``model.training_mode``."""
    if training_mode == "full":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    elif training_mode == "heads":
        # The output projections only: a cheap end-to-end smoke test of the whole
        # pipeline (data -> packing -> loss -> optimizer) on a single GPU.
        for name, parameter in model.named_parameters():
            parameter.requires_grad_("proj_out" in name)
    elif training_mode == "lora":
        if lora_config is None:
            raise ValueError("training_mode 'lora' needs a lora config")
        apply_lora(model, lora_config)
    else:
        raise ValueError(f"Unknown training_mode {training_mode!r}")


def trainable_parameter_count(model: torch.nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts, ZeRO-3 aware.

    Under ZeRO-3 ``numel()`` is 0 for partitioned parameters; ``ds_numel`` carries
    the real size. Reporting 0 trainable parameters at startup is a classic
    false alarm on that path.
    """
    trainable = total = 0
    for parameter in model.parameters():
        count = getattr(parameter, "ds_numel", None) or parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return trainable, total


def trainable_state_dict(model: torch.nn.Module, dtype: torch.dtype | None = None) -> dict[str, torch.Tensor]:
    """Only the tensors that are being trained.

    For LoRA/heads this is megabytes instead of the ~66GB full state; writing the
    full state from rank 0 takes long enough to trip the NCCL watchdog and kill
    the run.
    """
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    state = {}
    for name, tensor in model.state_dict().items():
        if name in trainable_names:
            tensor = tensor.detach().cpu()
            state[name] = tensor.to(dtype) if dtype is not None else tensor
    return state


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

_PEFT_KEY = re.compile(
    r"^(?:base_model\.model\.)?(?P<path>.+?)\.lora_(?P<ab>[AB])\.(?:default\.)?weight$"
)


def normalize_peft_keys(state_dict: dict[str, torch.Tensor]) -> dict[tuple[str, str], torch.Tensor]:
    """``{(module_path, 'A'|'B'): tensor}`` from PEFT's decorated key names."""
    normalized: dict[tuple[str, str], torch.Tensor] = {}
    for key, tensor in state_dict.items():
        match = _PEFT_KEY.match(key)
        if match is None:
            continue
        normalized[(match.group("path"), match.group("ab"))] = tensor
    return normalized


def fuse_qkv_lora(
    q: tuple[torch.Tensor, torch.Tensor] | None,
    k: tuple[torch.Tensor, torch.Tensor] | None,
    v: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse per-projection (A, B) LoRA pairs into one pair for a fused qkv_proj.

    The fused weight is ``concat([Wq; Wk; Wv])`` along the output axis, so its
    update is ``concat([Bq@Aq; Bk@Ak; Bv@Av])``. Stacking the A matrices and
    placing the B matrices block-diagonally reproduces that exactly at rank 3r:

        A = [Aq; Ak; Av]                     (3r, in)
        B = diag(Bq, Bk, Bv)                 (3*out, 3r)
        B @ A = [Bq@Aq; Bk@Ak; Bv@Av]        (3*out, in)

    A missing projection contributes zero rows/blocks, which keeps the fused
    output shape correct when only some of Q/K/V were adapted.
    """
    present = [pair for pair in (q, k, v) if pair is not None]
    if not present:
        raise ValueError("Nothing to fuse: q, k and v are all None")
    in_features = present[0][0].shape[1]
    rank = present[0][0].shape[0]
    out_features = present[0][1].shape[0]
    dtype, device = present[0][0].dtype, present[0][0].device

    a_blocks, b_blocks = [], []
    for pair in (q, k, v):
        if pair is None:
            a_blocks.append(torch.zeros(rank, in_features, dtype=dtype, device=device))
            b_blocks.append(torch.zeros(out_features, rank, dtype=dtype, device=device))
        else:
            a_blocks.append(pair[0])
            b_blocks.append(pair[1])

    fused_a = torch.cat(a_blocks, dim=0)
    fused_b = torch.zeros(out_features * 3, rank * 3, dtype=dtype, device=device)
    for index, block in enumerate(b_blocks):
        fused_b[index * out_features : (index + 1) * out_features, index * rank : (index + 1) * rank] = block
    return fused_a, fused_b


def to_comfyui_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert a PEFT LoRA state dict into the community H3 LoRA layout.

    ``transformer_blocks.N.attn.to_{q,k,v}`` fuse into
    ``diffusion_model.blocks.N.attn.qkv_proj``; everything else is renamed in
    place. The result is what ComfyUI's LoRA loader and the published fal H3
    adapters expect.
    """
    pairs = normalize_peft_keys(state_dict)
    if not pairs:
        raise ValueError("No LoRA tensors found; is this a LoRA checkpoint?")

    by_path: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for (path, ab), tensor in pairs.items():
        by_path[path][ab] = tensor

    converted: dict[str, torch.Tensor] = {}
    qkv_groups: dict[str, dict[str, dict[str, torch.Tensor]]] = defaultdict(dict)

    for path, ab_map in by_path.items():
        qkv = re.match(r"^transformer_blocks\.(\d+)\.attn\.to_(q|k|v)$", path)
        if qkv:
            qkv_groups[qkv.group(1)][qkv.group(2)] = ab_map
            continue
        converted_path = _rename_for_comfyui(path)
        for ab, tensor in ab_map.items():
            converted[f"{converted_path}.lora_{ab}.weight"] = tensor.contiguous()

    for block_index, projections in qkv_groups.items():
        def pair(name: str) -> tuple[torch.Tensor, torch.Tensor] | None:
            entry = projections.get(name)
            if entry is None or "A" not in entry or "B" not in entry:
                return None
            return entry["A"], entry["B"]

        fused_a, fused_b = fuse_qkv_lora(pair("q"), pair("k"), pair("v"))
        base = f"diffusion_model.blocks.{block_index}.attn.qkv_proj"
        converted[f"{base}.lora_A.weight"] = fused_a.contiguous()
        converted[f"{base}.lora_B.weight"] = fused_b.contiguous()

    return converted


def _rename_for_comfyui(path: str) -> str:
    renamed = re.sub(r"^transformer_blocks\.(\d+)\.", r"blocks.\1.", path)
    renamed = re.sub(r"\.ff\.net\.0\.proj$", ".ff.linear_1", renamed)
    renamed = re.sub(r"\.ff\.net\.2$", ".ff.linear_2", renamed)
    renamed = re.sub(r"\.to_out\.0$", ".to_out", renamed)
    return f"diffusion_model.{renamed}"


def export_lora(
    state_dict: dict[str, torch.Tensor],
    output_path: str | Path,
    metadata: dict[str, str] | None = None,
    fmt: str = "comfyui",
) -> Path:
    """Write a LoRA adapter to ``.safetensors`` in ``comfyui`` or raw ``peft`` layout."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "comfyui":
        payload = to_comfyui_state_dict(state_dict)
    elif fmt == "peft":
        payload = {key: tensor.contiguous() for key, tensor in state_dict.items()}
    else:
        raise ValueError(f"Unknown export format {fmt!r} (expected 'comfyui' or 'peft')")
    save_file(payload, str(output_path), metadata=metadata or {})
    logger.info("Wrote %d tensors to %s (%s layout)", len(payload), output_path, fmt)
    return output_path

"""Adapter export: the fused-QKV conversion has to be exact, not approximate."""

import pytest
import torch

from h3_trainer.lora import (
    fuse_qkv_lora,
    normalize_peft_keys,
    to_comfyui_state_dict,
    verify_target_modules,
)

RANK, IN, OUT = 4, 16, 8


def _pair() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(RANK, IN), torch.randn(OUT, RANK)


def test_fused_qkv_reproduces_the_concatenated_update_exactly():
    """B@A of the fused pair must equal concat([Bq@Aq; Bk@Ak; Bv@Av]).

    The fused weight is the three projections stacked along the output axis, so
    an export that does not reproduce this changes what the adapter does.
    """
    q, k, v = _pair(), _pair(), _pair()
    fused_a, fused_b = fuse_qkv_lora(q, k, v)

    assert fused_a.shape == (3 * RANK, IN)
    assert fused_b.shape == (3 * OUT, 3 * RANK)

    expected = torch.cat([q[1] @ q[0], k[1] @ k[0], v[1] @ v[0]], dim=0)
    assert torch.allclose(fused_b @ fused_a, expected, atol=1e-5)


def test_fusing_a_subset_zeroes_the_missing_projections():
    q = _pair()
    fused_a, fused_b = fuse_qkv_lora(q, None, None)
    update = fused_b @ fused_a
    assert torch.allclose(update[:OUT], q[1] @ q[0], atol=1e-5)
    assert torch.count_nonzero(update[OUT:]) == 0


def test_fusing_nothing_is_an_error():
    with pytest.raises(ValueError, match="Nothing to fuse"):
        fuse_qkv_lora(None, None, None)


def test_peft_keys_are_normalized_regardless_of_decoration():
    state = {
        "base_model.model.transformer_blocks.0.attn.to_q.lora_A.default.weight": torch.zeros(1),
        "transformer_blocks.0.attn.to_q.lora_B.weight": torch.zeros(1),
        "transformer_blocks.0.attn.norm_q.weight": torch.zeros(1),  # not a LoRA tensor
    }
    pairs = normalize_peft_keys(state)
    assert set(pairs) == {("transformer_blocks.0.attn.to_q", "A"), ("transformer_blocks.0.attn.to_q", "B")}


def test_comfyui_export_fuses_attention_and_renames_the_rest():
    state = {}
    for projection in ("to_q", "to_k", "to_v"):
        a, b = _pair()
        state[f"transformer_blocks.7.attn.{projection}.lora_A.weight"] = a
        state[f"transformer_blocks.7.attn.{projection}.lora_B.weight"] = b
    ff_a, ff_b = _pair()
    state["transformer_blocks.7.ff.net.2.lora_A.weight"] = ff_a
    state["transformer_blocks.7.ff.net.2.lora_B.weight"] = ff_b

    converted = to_comfyui_state_dict(state)

    assert "diffusion_model.blocks.7.attn.qkv_proj.lora_A.weight" in converted
    assert "diffusion_model.blocks.7.attn.qkv_proj.lora_B.weight" in converted
    # The three separate projections must not survive as themselves.
    assert not any("to_q" in key for key in converted)
    # Non-attention modules are renamed into the community layout, not fused.
    assert "diffusion_model.blocks.7.ff.linear_2.lora_A.weight" in converted
    assert converted["diffusion_model.blocks.7.ff.linear_2.lora_A.weight"].shape == (RANK, IN)


def test_export_rejects_a_state_dict_with_no_adapter():
    with pytest.raises(ValueError, match="No LoRA tensors"):
        to_comfyui_state_dict({"transformer_blocks.0.attn.to_q.weight": torch.zeros(1)})


def test_target_verification_catches_names_that_match_nothing():
    """The failure mode this exists for: PEFT ignores unmatched targets silently."""

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = torch.nn.Linear(4, 4)
            self.to_out = torch.nn.ModuleList([torch.nn.Linear(4, 4)])

    model = torch.nn.Module()
    model.attn = Block()

    counts = verify_target_modules(model, ["to_q", "to_out.0"])
    assert counts == {"to_q": 1, "to_out.0": 1}

    with pytest.raises(ValueError, match="matched no Linear layer"):
        verify_target_modules(model, ["to_q", "to_qkv"])

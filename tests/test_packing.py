"""Packed-sequence assembly: row counts, masks, timestep pinning, round-trips."""

import pytest
import torch

from h3_trainer.constants import PATCH_SIZE, Geometry
from h3_trainer.packing import (
    CONDITION_AUDIO_TIMESTEP,
    assemble,
    build_layout,
    check_reference_rows,
    encode_reference_geometry,
    loss_masks,
    prepared_references_from_cache,
    row_timesteps,
)

GEOMETRY = Geometry.create(width=256, height=256, num_frames=22)
TEXT_ROWS = 12


def _layout(**kwargs):
    return build_layout(
        num_text_rows=TEXT_ROWS,
        latent_frames=GEOMETRY.latent_frames,
        latent_height=GEOMETRY.latent_height,
        latent_width=GEOMETRY.latent_width,
        audio_latents=GEOMETRY.audio_latents,
        **kwargs,
    )


def test_geometry_matches_the_layout_the_packer_builds():
    layout = _layout()
    assert int(layout.video_indices.numel()) == GEOMETRY.video_rows
    assert int(layout.audio_indices.numel()) == GEOMETRY.audio_rows
    assert int(layout.text_indices.numel()) == TEXT_ROWS
    assert layout.sequence_length == GEOMETRY.sequence_length(TEXT_ROWS)


def test_keyframe_anchors_add_conditioning_rows_that_carry_no_loss():
    layout = _layout(keyframe_anchors=("first",))
    assert layout.num_condition_video_rows > 0
    video_mask, audio_mask = loss_masks(layout, "cpu")
    assert video_mask[: layout.num_condition_video_rows].sum() == 0
    assert video_mask[layout.num_condition_video_rows :].all()
    assert audio_mask.all()  # keyframes contribute no audio rows


def test_conditioning_rows_are_pinned_near_clean():
    layout = _layout(keyframe_anchors=("first",))
    video_timestep, audio_timestep = 0.4, 0.7
    timesteps, indices = row_timesteps(layout, video_timestep, audio_timestep, "cpu")
    per_row = timesteps[indices]

    condition_rows = layout.video_indices[: layout.num_condition_video_rows]
    target_rows = layout.video_indices[layout.num_condition_video_rows :]
    # Visual conditioning sits just shy of clean; targets sit at their own noise level.
    assert torch.allclose(per_row[condition_rows], torch.tensor(0.999))
    assert torch.allclose(per_row[target_rows], torch.tensor(video_timestep))
    assert torch.allclose(per_row[layout.audio_indices], torch.tensor(audio_timestep))


def test_assemble_produces_model_ready_inputs():
    layout = _layout()
    video = torch.randn(GEOMETRY.video_rows, 96)
    audio = torch.randn(GEOMETRY.audio_rows, 32)
    text = torch.randn(TEXT_ROWS, 5120)

    packed = assemble(
        layout=layout,
        target_video=video,
        target_audio=audio,
        text_embeds=text,
        sigma_video=0.5,
        sigma_audio=0.2,
        video_timestep=0.5,
        audio_timestep=0.8,
        dtype=torch.float32,
    )
    kwargs = packed.to_model_kwargs()
    assert kwargs["hidden_states"].shape == (1, GEOMETRY.video_rows, 96)
    assert kwargs["audio_hidden_states"].shape == (1, GEOMETRY.audio_rows, 32)
    assert kwargs["encoder_hidden_states"].shape == (1, TEXT_ROWS, 5120)
    assert kwargs["position_ids"].shape == (layout.sequence_length, 3)
    assert kwargs["token_tags"].shape == (layout.sequence_length,)
    assert packed.sequence_length == layout.sequence_length

    # The noised rows must sit on the interpolant between the clean latents and
    # the noise that was drawn, at exactly the requested sigma.
    expected = 0.5 * packed.clean_video + 0.5 * packed.noise_video
    assert torch.allclose(packed.video, expected, atol=1e-5)


def test_assemble_rejects_latents_from_a_different_geometry():
    layout = _layout()
    with pytest.raises(ValueError, match="does not match the packed layout"):
        assemble(
            layout=layout,
            target_video=torch.randn(GEOMETRY.video_rows - 4, 96),
            target_audio=torch.randn(GEOMETRY.audio_rows, 32),
            text_embeds=torch.randn(TEXT_ROWS, 5120),
            sigma_video=0.5,
            sigma_audio=0.5,
            video_timestep=0.5,
            audio_timestep=0.5,
            dtype=torch.float32,
        )


def test_explicit_token_tags_must_match_the_prompt_length():
    with pytest.raises(ValueError, match="text_token_tags"):
        _layout(tags=torch.ones(TEXT_ROWS + 3, dtype=torch.long))


def test_reference_layout_places_reference_rows_before_the_targets():
    references = prepared_references_from_cache(
        [("image", None, encode_reference_geometry("image", 1, 8, 8, 0))]
    )
    layout = _layout(references=references)
    assert layout.num_condition_video_rows == references[0].num_video_rows
    assert int(layout.video_indices.numel()) == GEOMETRY.video_rows + references[0].num_video_rows
    video_mask, _ = loss_masks(layout, "cpu")
    assert video_mask.sum() == GEOMETRY.video_rows  # references never carry loss


def test_reference_audio_rows_are_handed_over_clean():
    references = prepared_references_from_cache(
        [("audio", None, encode_reference_geometry("audio", 0, 0, 0, 40))]
    )
    layout = _layout(references=references)
    timesteps, indices = row_timesteps(layout, 0.3, 0.6, "cpu")
    per_row = timesteps[indices]
    reference_rows = layout.audio_indices[: layout.num_condition_audio_rows]
    assert layout.num_condition_audio_rows == 80  # 40 latents x 2 channels
    assert torch.allclose(per_row[reference_rows], torch.tensor(CONDITION_AUDIO_TIMESTEP))


def test_reference_row_counts_are_verified_against_the_cache():
    references = prepared_references_from_cache(
        [("video", None, encode_reference_geometry("video", 2, 8, 8, 10))]
    )
    good_video = torch.randn(references[0].num_video_rows, 96)
    good_audio = torch.randn(references[0].num_audio_rows, 32)
    check_reference_rows(references, good_video, good_audio)
    with pytest.raises(ValueError, match="disagree"):
        check_reference_rows(references, good_video[:-1], good_audio)


def test_patchify_round_trip():
    from diffusers.modular_pipelines.minimax_h3.packing import (
        patchify_video_latents,
        unpatchify_video_tokens,
    )

    latents = torch.randn(1, 24, GEOMETRY.latent_frames, GEOMETRY.latent_height, GEOMETRY.latent_width)
    rows = patchify_video_latents(latents, PATCH_SIZE)
    assert rows.shape == (GEOMETRY.video_rows, 24 * 4)
    restored = unpatchify_video_tokens(
        rows,
        num_latent_frames=GEOMETRY.latent_frames,
        latent_height=GEOMETRY.latent_height,
        latent_width=GEOMETRY.latent_width,
        channels=24,
        patch_size=PATCH_SIZE,
    )
    assert torch.allclose(restored, latents)

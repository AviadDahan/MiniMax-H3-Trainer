"""The numeric conventions that silently corrupt weights when they are wrong.

Every assertion here corresponds to a way an H3 fine-tune can look perfectly
healthy -- smooth loss curve, no errors -- while training the model backwards.
"""

import pytest
import torch

from h3_trainer.flow_matching import (
    SigmaPair,
    add_noise,
    audio_loss_weight,
    flow_matching_loss,
    seeded_noise_like,
    shift_sigma,
    timestep_from_sigma,
    velocity_target,
)


def test_timestep_is_one_minus_sigma():
    # H3's contract: t=1 is clean data, t=0 is pure noise -- the opposite of the
    # SD3/Wan convention most training code assumes.
    assert timestep_from_sigma(0.0) == 1.0
    assert timestep_from_sigma(1.0) == 0.0
    assert timestep_from_sigma(0.25) == pytest.approx(0.75)


def test_velocity_target_points_at_the_data():
    clean = torch.tensor([2.0, -1.0])
    noise = torch.tensor([0.5, 0.5])
    # v = x0 - eps, not eps - x0. The scheduler reconstructs x0 = x_t + (1-t)*v,
    # so the opposite sign actively walks away from the data.
    assert torch.allclose(velocity_target(clean, noise), torch.tensor([1.5, -1.5]))


def test_reconstruction_identity_holds():
    """x_t + (1 - t) * v == x0, the identity the scheduler denoises with."""
    clean = torch.randn(8, 4)
    noise = torch.randn(8, 4)
    sigma = 0.37
    x_t = add_noise(clean, noise, sigma)
    t = timestep_from_sigma(sigma)
    assert torch.allclose(x_t + (1.0 - t) * velocity_target(clean, noise), clean, atol=1e-5)


def test_add_noise_endpoints():
    clean, noise = torch.randn(4, 2), torch.randn(4, 2)
    assert torch.allclose(add_noise(clean, noise, 0.0), clean)
    assert torch.allclose(add_noise(clean, noise, 1.0), noise)


@pytest.mark.parametrize("u", [0.05, 0.3, 0.5, 0.9])
def test_shifted_schedules_are_monotone_and_bounded(u):
    for shift in (3.0, 12.0):
        sigma = shift_sigma(u, shift)
        assert 0.0 < sigma < 1.0
    # A larger shift pushes the same u to more noise -- that is the whole point
    # of video using 12.0 while audio uses 3.0.
    assert shift_sigma(u, 12.0) > shift_sigma(u, 3.0)


def test_sigma_pair_shares_one_uniform_draw():
    """Video and audio sigmas must come from the same u.

    At inference the two schedulers advance in lockstep off a shared step index.
    Drawing two independent sigmas would train (video, audio) noise pairings the
    model never encounters.
    """
    pair = SigmaPair.from_u(0.4, video_shift=12.0, audio_shift=3.0)
    assert pair.video == pytest.approx(shift_sigma(0.4, 12.0))
    assert pair.audio == pytest.approx(shift_sigma(0.4, 3.0))
    assert pair.video != pair.audio
    assert pair.video_timestep == pytest.approx(1.0 - pair.video)
    assert pair.audio_timestep == pytest.approx(1.0 - pair.audio)


def test_shift_of_one_is_the_identity():
    for u in (0.1, 0.6, 0.95):
        assert shift_sigma(u, 1.0) == pytest.approx(u)


def test_zero_audio_is_weighted_out_but_stays_in_the_graph():
    silent = torch.zeros(16, 32)
    real = torch.randn(16, 32)
    assert audio_loss_weight(silent) == 0.0
    assert audio_loss_weight(real) == 1.0

    # The term must still be differentiable at weight 0: dropping it makes ranks
    # disagree about which gradients exist, and DDP/ZeRO hang.
    prediction = torch.randn(1, 16, 32, requires_grad=True)
    loss = flow_matching_loss(prediction, silent, torch.randn(16, 32), weight=audio_loss_weight(silent))
    loss.backward()
    assert float(loss) == 0.0
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad) == 0


def test_loss_masks_conditioning_rows_out_of_the_prediction():
    """The mask applies to the prediction only.

    The model predicts every row of a modality, conditioning rows included, while
    the cached clean/noise latents hold only the target rows -- the conditioning
    rows were prepended when the sequence was assembled. Masking the target too
    would index rows it never had, which is exactly what IC-LoRA training hit.
    """
    clean = torch.randn(7, 4)
    noise = torch.randn(7, 4)
    prediction = torch.cat(
        [torch.full((3, 4), 1e3), velocity_target(clean, noise)]  # 3 conditioning rows, then targets
    ).unsqueeze(0)
    mask = torch.ones(10, dtype=torch.bool)
    mask[:3] = False

    assert float(flow_matching_loss(prediction, clean, noise, row_mask=mask)) == pytest.approx(0.0, abs=1e-6)


def test_loss_rejects_a_mask_that_does_not_line_up():
    clean, noise = torch.randn(7, 4), torch.randn(7, 4)
    prediction = torch.randn(1, 10, 4)
    with pytest.raises(ValueError, match="do not line up"):
        flow_matching_loss(prediction, clean, noise, row_mask=torch.ones(10, dtype=torch.bool))


def test_seeded_noise_is_reproducible_and_seed_dependent():
    reference = torch.zeros(6, 3)
    assert torch.equal(seeded_noise_like(reference, 7), seeded_noise_like(reference, 7))
    assert not torch.equal(seeded_noise_like(reference, 7), seeded_noise_like(reference, 8))

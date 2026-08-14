"""Validation references must reach the conditioner *prepared*, not as raw requests.

Regression test for the second bug that stopped this run's validation from ever
producing a clip. ``prepare()`` built its references as bare dicts:

    {"image": c.image, "video": c.video, "audio": c.audio}

while ``encode_ref2va_prompt`` reads ``reference.kind`` to split images from
videos, ``reference.frames`` to feed the video processor, and writes
``reference.block_timestamps`` back. A dict has none of those, so pre-encoding
died with ``'dict' object has no attribute 'kind'``.

That exception was caught and logged at warning level, and sampling fell back to
letting the pipeline encode prompts in-loop -- with the 63GB conditioner left on
the CPU, where one 448x768x124 sample ran past its 1800 s budget. The visible
symptom was a hang, three levels away from the cause.

The tests here are deliberately CPU-only: they need PyAV and diffusers, not model
weights, so they run in the ordinary suite rather than only on a training box.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from h3_trainer.preprocessing.media import write_video_with_audio
from h3_trainer.validation_runner import ValidationRunner

pytest.importorskip("diffusers.modular_pipelines.minimax_h3")

WIDTH, HEIGHT, FRAMES = 448, 768, 124


@pytest.fixture
def skeleton_clip(tmp_path):
    """A silent clip shaped like a rendered pose reference."""
    frames = np.zeros((FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8)
    frames[:, 300:340, 200:240] = 255  # something non-uniform to encode
    path = tmp_path / "ref02_latin_00.mp4"
    write_video_with_audio(frames, path)
    return path


def make_runner(clip):
    """A runner with one reference-conditioned validation sample, no model loaded."""
    sample = SimpleNamespace(
        prompt="a person dancing, following the motion of the reference skeleton",
        video_dims=None,
        seed=None,
        conditions=[SimpleNamespace(type="reference", image=None, video=str(clip), audio=None)],
    )
    config = SimpleNamespace(
        validation=SimpleNamespace(samples=[sample], video_dims=[WIDTH, HEIGHT, FRAMES]),
        model=SimpleNamespace(model_path="unused", variant="ref2va"),
    )
    runner = ValidationRunner.__new__(ValidationRunner)  # no weights, no device
    runner.config = config
    return runner, sample


def test_prepared_references_carry_a_kind(skeleton_clip):
    """The exact attribute whose absence produced the failure."""
    runner, sample = make_runner(skeleton_clip)
    prepared = runner._prepared_references(sample)
    assert len(prepared) == 1
    assert prepared[0].kind == "video"


def test_prepared_references_carry_decoded_frames(skeleton_clip):
    """encode_ref2va_prompt feeds `.frames` to the video processor."""
    runner, sample = make_runner(skeleton_clip)
    reference = runner._prepared_references(sample)[0]
    assert reference.frames is not None
    assert reference.frames.ndim == 4 and reference.frames.shape[-1] == 3
    assert reference.frames.dtype == np.uint8
    # Truncated to the generated frame count, never longer.
    assert reference.frames.shape[0] <= FRAMES


def test_prepared_references_are_not_dicts(skeleton_clip):
    """A dict passes `if references:` and fails three frames deep. Fail here instead."""
    runner, sample = make_runner(skeleton_clip)
    for reference in runner._prepared_references(sample):
        assert not isinstance(reference, dict)
        for attribute in ("kind", "frames", "block_timestamps"):
            assert hasattr(reference, attribute), f"missing {attribute!r}"


def test_reference_requests_are_the_dataclass_the_blocks_demand(skeleton_clip):
    """The pipeline rejects anything else with a ValueError naming the index."""
    from diffusers.modular_pipelines.minimax_h3.packing_ref2va import MiniMaxH3Reference

    runner, sample = make_runner(skeleton_clip)
    requests = runner._reference_requests(sample)
    assert len(requests) == 1
    assert isinstance(requests[0], MiniMaxH3Reference)


def test_no_reference_conditions_yields_no_references(skeleton_clip):
    """A plain text validation sample must not acquire references from nowhere."""
    runner, sample = make_runner(skeleton_clip)
    sample.conditions = []
    assert runner._prepared_references(sample) == []
    assert runner._reference_requests(sample) == []

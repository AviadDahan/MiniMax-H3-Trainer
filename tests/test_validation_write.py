"""A validation sample must reach disk, and must never take the run down with it.

Two regressions from the same step-0 sample. Denoising ran correctly on the GPU
and produced a real clip; then:

* ``_write`` called ``np.asarray`` on the audio, which ``output_type="np"`` leaves
  as a **torch tensor on the GPU**. numpy refuses to convert a CUDA tensor rather
  than copying it, so the write raised ``TypeError``.
* ``_write`` was called *outside* the try/except guarding sampling, so that
  TypeError propagated out of ``run()`` and killed the training process -- after
  the run had already paid for the sample.

The second is the one that matters. Sampling is the risky part and it was
guarded; writing a file is the cheap part and it was not, so the cheapest step in
validation was the only one that could end a run.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from h3_trainer.validation_runner import ValidationRunner, _to_numpy

FRAMES, HEIGHT, WIDTH = 8, 64, 32


def test_to_numpy_passes_arrays_through():
    array = np.zeros((4, 4), dtype=np.uint8)
    assert _to_numpy(array) is array or np.array_equal(_to_numpy(array), array)


def test_to_numpy_converts_a_detached_tensor():
    """The audio path: a tensor numpy will not convert on its own."""
    tensor = torch.ones(2, 100, requires_grad=True)
    result = _to_numpy(tensor)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 100)


def test_to_numpy_handles_bfloat16():
    """numpy has no bfloat16; the cast to float32 is what makes this work."""
    result = _to_numpy(torch.ones(2, 8, dtype=torch.bfloat16))
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32


def make_runner(tmp_path, samples=1):
    runner = ValidationRunner.__new__(ValidationRunner)
    runner.output_dir = Path(tmp_path)
    runner._sampling_disabled = False
    runner._conditioning = None
    runner._pipeline = object()  # non-None so _build_pipeline short-circuits
    runner.transformer = SimpleNamespace(training=False, eval=lambda: None, train=lambda: None)
    runner.config = SimpleNamespace(
        validation=SimpleNamespace(
            samples=[
                SimpleNamespace(prompt="p", video_dims=[WIDTH, HEIGHT, 22], seed=None, conditions=[])
                for _ in range(samples)
            ],
            video_dims=[WIDTH, HEIGHT, 22],
            inference_steps=1,
            seed=0,
            sample_timeout_seconds=0,
            frame_rate=24.0,
        )
    )
    return runner


def test_a_write_failure_does_not_end_the_run(tmp_path, monkeypatch):
    """The regression that killed 13 hours of training on a TypeError."""
    runner = make_runner(tmp_path)
    monkeypatch.setattr(runner, "_build_pipeline", lambda: (lambda **kw: SimpleNamespace(videos=None)))
    monkeypatch.setattr(runner, "_prompt_kwargs", lambda index, sample: {})
    monkeypatch.setattr(runner, "_conditioning_kwargs", lambda sample: {})

    def explode(*args, **kwargs):
        raise TypeError("can't convert cuda:0 device type tensor to numpy")

    monkeypatch.setattr(runner, "_write", explode)
    assert runner.run(step=0) == []  # returned, did not raise


def test_one_bad_sample_does_not_stop_the_next(tmp_path, monkeypatch):
    runner = make_runner(tmp_path, samples=2)
    monkeypatch.setattr(runner, "_build_pipeline", lambda: (lambda **kw: SimpleNamespace(videos=None)))
    monkeypatch.setattr(runner, "_prompt_kwargs", lambda index, sample: {})
    monkeypatch.setattr(runner, "_conditioning_kwargs", lambda sample: {})

    written = []

    def flaky(result, step, index):
        if index == 0:
            raise TypeError("boom")
        path = Path(tmp_path) / f"step{step:07d}_sample{index}.mp4"
        written.append(path)
        return path

    monkeypatch.setattr(runner, "_write", flaky)
    assert runner.run(step=0) == written
    assert len(written) == 1


def test_write_muxes_a_cuda_style_audio_tensor(tmp_path):
    """End to end through the real _write, with the tensor types the pipeline returns."""
    runner = make_runner(tmp_path)
    result = SimpleNamespace(
        videos=np.zeros((FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8),
        audios=torch.zeros(2, 32000, dtype=torch.float32),  # tensor, as the pipeline returns it
    )
    path = runner._write(result, step=150, index=0)
    assert path.exists() and path.stat().st_size > 0
    assert path.name == "step0000150_sample0.mp4"


def test_write_refuses_a_result_with_no_video(tmp_path):
    runner = make_runner(tmp_path)
    with pytest.raises(RuntimeError, match="no video"):
        runner._write(SimpleNamespace(videos=None, audios=None), step=0, index=0)

"""The validation sampler must hand its transformer to the right component name.

This is a regression test for a bug that wasted nearly two hours of a run and
would have quietly invalidated every media sample it did produce.

``ModularPipeline.update_components`` ignores any keyword it does not recognise,
logging a warning and continuing. The Ref2VA blocks register their denoiser as
``transformer_ref``; the validation runner passed ``transformer=``. The keyword was
dropped, the pipeline kept its own transformer -- not the one carrying the LoRA
under test, and never moved onto a device -- and sampling fell back to CPU, where a
448x768x124 clip does not finish in any useful time.

Two failure modes, both silent:

* the sample would show the BASE model, not the adapter, so the check meant to
  prove the adapter works would have proved nothing;
* on a large bucket it never returns at all, which reads as a hang.

So this asserts the name is resolved from the pipeline, not assumed.
"""

import pytest

from h3_trainer.config import ModelConfig


class FakePipeline:
    """Mimics the one behaviour that matters: unknown keywords are dropped."""

    def __init__(self, component_names):
        self._component_specs = {name: object() for name in component_names}
        for name in component_names:
            setattr(self, name, None)

    def update_components(self, **kwargs):
        for name, value in kwargs.items():
            if name in self._component_specs:
                setattr(self, name, value)
            # anything else is silently ignored, exactly like diffusers


def resolve(runner_cls, config, pipeline):
    return runner_cls._transformer_component_name(config, pipeline)


@pytest.fixture
def runner_factory():
    """A ValidationRunner with only the attributes the resolver touches."""
    from h3_trainer.validation_runner import ValidationRunner

    def build(variant):
        runner = ValidationRunner.__new__(ValidationRunner)
        model = ModelConfig.model_construct(variant=variant)

        class Cfg:
            pass

        cfg = Cfg()
        cfg.model = model
        runner.config = cfg
        return runner

    return build


def test_ref2va_resolves_to_transformer_ref(runner_factory):
    """The variant this repo's IC-LoRA work actually uses."""
    runner = runner_factory("ref2va")
    pipeline = FakePipeline(["vae", "audio_vae", "scheduler", "transformer_ref"])
    assert runner._transformer_component_name(pipeline) == "transformer_ref"


def test_fl2va_resolves_to_transformer(runner_factory):
    runner = runner_factory("fl2va")
    pipeline = FakePipeline(["vae", "audio_vae", "scheduler", "transformer"])
    assert runner._transformer_component_name(pipeline) == "transformer"


def test_resolved_name_actually_lands_on_the_pipeline(runner_factory):
    """The whole point: the object under test must be the one that gets sampled."""
    runner = runner_factory("ref2va")
    pipeline = FakePipeline(["vae", "transformer_ref"])
    sentinel = object()
    name = runner._transformer_component_name(pipeline)
    pipeline.update_components(**{name: sentinel})
    assert getattr(pipeline, name) is sentinel

    # And the original bug: the wrong name is dropped without complaint.
    pipeline.update_components(transformer=object())
    assert getattr(pipeline, "transformer_ref") is sentinel
    assert not hasattr(pipeline, "transformer")


def test_missing_transformer_raises_rather_than_sampling_the_wrong_model(runner_factory):
    runner = runner_factory("ref2va")
    pipeline = FakePipeline(["vae", "audio_vae"])
    with pytest.raises(RuntimeError, match="no transformer component"):
        runner._transformer_component_name(pipeline)


def test_falls_back_when_the_blocks_disagree_with_the_variant(runner_factory):
    """Blocks renamed underneath us: prefer what the pipeline declares."""
    runner = runner_factory("ref2va")
    pipeline = FakePipeline(["vae", "transformer"])
    assert runner._transformer_component_name(pipeline) == "transformer"

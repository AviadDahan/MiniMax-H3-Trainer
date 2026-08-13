"""Every shipped config must load.

A config that no longer validates is a broken example, and examples are the part
of a trainer people actually copy. Validation is not cosmetic here: the schema is
where H3's geometry rules live (frames on the ``17n + 5`` grid, dimensions
divisible by 32, references only on the ref2va variant), so a config that parses
has already been checked against the model's contract.

``model.model_path`` points at a local checkout, so these run only where the
weights are present; elsewhere they skip rather than fail.
"""

from pathlib import Path

import pytest
import yaml

from h3_trainer.config import H3TrainerConfig

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))


def test_configs_directory_is_not_empty():
    """Guards the parametrization below: zero files would pass silently."""
    assert CONFIGS, f"no configs found under {CONFIG_DIR}"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_validates(path: Path):
    payload = yaml.safe_load(path.read_text())
    model_path = Path(payload["model"]["model_path"])
    if not model_path.exists():
        pytest.skip(f"{model_path} not present on this machine")

    config = H3TrainerConfig.model_validate(payload)

    # Round-trip: what we write back must itself be valid input. This catches
    # serialization drift, which otherwise only shows up when someone resumes
    # from a run's saved config.
    H3TrainerConfig.model_validate(yaml.safe_load(yaml.safe_dump(config.model_dump(mode="json"))))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_reference_conditions_only_on_the_ref2va_variant(path: Path):
    """The FL2VA transformer has no reference rows in its layout.

    Checked separately from schema validation so the failure names the file
    rather than surfacing as a generic validation error mid-run.
    """
    payload = yaml.safe_load(path.read_text())
    strategy = payload.get("training_strategy", {})
    uses_references = any(
        condition.get("type") == "reference"
        for modality in ("video", "audio")
        for condition in (strategy.get(modality) or {}).get("conditions", []) or []
    )
    if uses_references:
        assert payload["model"]["variant"] == "ref2va", (
            f"{path.name} packs reference conditions but targets the "
            f"{payload['model']['variant']} transformer"
        )

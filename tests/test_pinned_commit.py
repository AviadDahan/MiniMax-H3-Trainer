"""The pinned diffusers commit is named in three places; they must agree.

H3's classes ship in no released diffusers wheel, so the integration commit is
pinned -- in `pyproject.toml` (what an install resolves), in `install_env.sh`
(what the setup script fetches) and in the ImportError text in `constants.py`
(what a user is told to run when the import fails). Three copies of a SHA drift,
and the failure is nasty: an environment built from one of them silently gets
different packing helpers than the code was verified against.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHA = re.compile(r"diffusers(?:\.git)?@([0-9a-f]{40})")

SOURCES = [
    ROOT / "pyproject.toml",
    ROOT / "scripts" / "install_env.sh",
    ROOT / "src" / "h3_trainer" / "constants.py",
]


def test_the_pinned_diffusers_commit_is_the_same_everywhere():
    found = {}
    for path in SOURCES:
        text = path.read_text()
        matches = set(SHA.findall(text))
        # install_env.sh assigns the bare SHA to a variable rather than writing a URL.
        if not matches:
            matches = set(re.findall(r'DIFFUSERS_COMMIT="([0-9a-f]{40})"', text))
        assert matches, f"{path.name} names no pinned diffusers commit"
        assert len(matches) == 1, f"{path.name} names more than one commit: {sorted(matches)}"
        found[path.name] = matches.pop()

    assert len(set(found.values())) == 1, f"pinned commit disagrees across files: {found}"


def test_diffusers_is_an_actual_dependency():
    """Without this line, installing the package yields an env that cannot import it."""
    assert "diffusers @ git+" in (ROOT / "pyproject.toml").read_text()


def test_declared_entry_points_resolve():
    """A console script pointing at a missing module installs fine and fails on use.

    pyproject once declared `h3-train = h3_trainer.cli:train_main` against a module
    that did not exist -- `pip install .` succeeded and the command raised
    ModuleNotFoundError. The documented interface is `python scripts/train.py`.
    """
    import importlib
    import tomllib

    with (ROOT / "pyproject.toml").open("rb") as handle:
        scripts = tomllib.load(handle).get("project", {}).get("scripts", {})

    for name, target in scripts.items():
        module_name, _, attribute = target.partition(":")
        module = importlib.import_module(module_name)
        assert hasattr(module, attribute), f"entry point {name} -> {target} does not resolve"

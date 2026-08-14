"""``output_dir`` is a root; each launch gets a timestamped directory under it.

Runs here are hours long and get relaunched -- after a crash, a config change, or
a corrected estimate. Sharing one directory means the second launch overwrites the
first's ``train.log``, appends into the same ``metrics.jsonl``, and drops
checkpoints from a different configuration alongside the originals with nothing to
distinguish them. During this repo's own pose run that had to be worked around by
hand, twice, by moving the previous attempt out of the way before relaunching.

The multi-rank case is the subtle one: ranks start milliseconds apart, so each
calling ``time.time()`` independently would scatter them across sibling
directories and rank 0 would never see rank 1's checkpoints.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from h3_trainer.trainer import H3Trainer


class FakeContext:
    """Stands in for DistributedContext, including its clock agreement."""

    def __init__(self, rank=0, agreed_time=None):
        self.rank = rank
        self.agreed = agreed_time

    @property
    def is_main(self):
        return self.rank == 0

    def all_reduce_min(self, value):
        # Real ranks agree on the earliest clock; mimic that when asked to.
        return self.agreed if self.agreed is not None else value

    def barrier(self):
        pass


def make_trainer(rank=0, agreed_time=None):
    trainer = H3Trainer.__new__(H3Trainer)
    trainer.context = FakeContext(rank=rank, agreed_time=agreed_time)
    return trainer


def test_run_dir_is_a_timestamped_child_of_the_root(tmp_path):
    trainer = make_trainer()
    run = trainer._new_run_dir(tmp_path)
    assert run.parent == tmp_path
    # parses as the documented format, so runs sort chronologically by name
    datetime.strptime(run.name, "%Y%m%d-%H%M%S")


def test_every_rank_lands_in_the_same_directory():
    """The failure this prevents: rank 1 writing where rank 0 never looks."""
    agreed = 1_700_000_000
    dirs = {make_trainer(rank=r, agreed_time=agreed)._new_run_dir(Path("/runs/x")).name for r in range(4)}
    assert len(dirs) == 1


def test_timestamp_is_utc_not_local():
    """Local time makes run names ambiguous across DST and machines."""
    agreed = 1_700_000_000
    run = make_trainer(agreed_time=agreed)._new_run_dir(Path("/runs/x"))
    expected = datetime.fromtimestamp(agreed, tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
    assert run.name == expected


def test_relaunch_does_not_collide_with_the_previous_run(tmp_path):
    first = make_trainer(agreed_time=1_700_000_000)._new_run_dir(tmp_path)
    second = make_trainer(agreed_time=1_700_000_060)._new_run_dir(tmp_path)
    assert first != second
    first.mkdir(parents=True)
    (first / "train.log").write_text("first run")
    second.mkdir(parents=True)
    (second / "train.log").write_text("second run")
    assert (first / "train.log").read_text() == "first run"


def test_latest_points_at_the_newest_run(tmp_path):
    trainer = make_trainer()
    first = tmp_path / "20260101-000000"
    first.mkdir()
    trainer._point_latest_at(first)
    assert (tmp_path / "latest").resolve() == first

    second = tmp_path / "20260101-010000"
    second.mkdir()
    trainer._point_latest_at(second)
    assert (tmp_path / "latest").resolve() == second


def test_latest_is_relative_so_the_root_can_be_moved(tmp_path):
    """An absolute link breaks the moment the runs directory is relocated."""
    trainer = make_trainer()
    run = tmp_path / "20260101-000000"
    run.mkdir()
    trainer._point_latest_at(run)
    assert not Path((tmp_path / "latest").readlink()).is_absolute()


def test_latest_failure_does_not_end_the_run(tmp_path, monkeypatch):
    """A 14-hour run must not die because a symlink could not be written."""
    trainer = make_trainer()
    run = tmp_path / "20260101-000000"
    run.mkdir()

    def refuse(*args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(Path, "symlink_to", refuse)
    trainer._point_latest_at(run)  # must not raise

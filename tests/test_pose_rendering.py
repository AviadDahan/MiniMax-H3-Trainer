"""Joint markers must never paint outside the joint.

Regression test for a bug that put white bars clean across the skeleton videos --
into the training data, and into every held-out reference the adapter is evaluated
against. All 8 evaluation clips carried it, 8-21 frames out of 124 each.

The cause is that a negative slice bound is not out of range in numpy, it counts
back from the far edge. MediaPipe predicts landmarks outside the frame whenever a
limb leaves it, so a wrist at x = -300 turned

    canvas[y0:y1, max(0, x - 3) : x + 4]

into ``canvas[y0:y1, 0 : width - 296]`` -- a bar spanning almost the whole frame
rather than the empty slice the code assumed. Latin dance, with the widest arm
extensions, was hit hardest.

It matters more than a cosmetic blemish: the skeleton is the *instruction* an
IC-LoRA follows, and a bright band spanning the frame is a strong, consistent
feature that means nothing.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "extract_pose.py"
_spec = importlib.util.spec_from_file_location("extract_pose", SCRIPT)
extract_pose = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract_pose)

draw_point = extract_pose.draw_point
draw_line = extract_pose.draw_line

H, W = 768, 448


def blank():
    return np.zeros((H, W, 3), dtype=np.uint8)


def painted_columns(canvas, row):
    return int((canvas[row] > 0).any(axis=-1).sum())


def test_point_inside_paints_a_small_square():
    canvas = blank()
    draw_point(canvas, 200, 300)
    assert painted_columns(canvas, 300) == 7  # radius 3 -> 7px wide
    assert (canvas > 0).any()


@pytest.mark.parametrize("x", [-300, -10, -4, -1])
def test_negative_x_paints_nothing_wide(x):
    """The exact bug: a negative bound used to wrap to the far edge."""
    canvas = blank()
    draw_point(canvas, x, 300)
    assert painted_columns(canvas, 300) <= 4, f"x={x} painted a bar"


@pytest.mark.parametrize("y", [-300, -10, -4, -1])
def test_negative_y_paints_nothing_tall(y):
    canvas = blank()
    draw_point(canvas, 200, y)
    assert int((canvas[:, 200] > 0).any(axis=-1).sum()) <= 4, f"y={y} painted a bar"


def test_point_far_outside_paints_nothing_at_all():
    canvas = blank()
    for x, y in ((-500, -500), (W + 500, H + 500), (-500, 400), (200, H + 500)):
        draw_point(canvas, x, y)
    assert not (canvas > 0).any()


def test_point_straddling_an_edge_paints_only_the_visible_part():
    canvas = blank()
    draw_point(canvas, 1, 300)  # centre inside, left half outside
    assert 0 < painted_columns(canvas, 300) <= 5


def test_no_row_is_ever_mostly_painted_by_one_joint():
    """The signature the dataset scan looked for: a row >55% covered."""
    canvas = blank()
    for x in (-400, -50, -1, 0, W - 1, W + 50, W + 400):
        draw_point(canvas, x, 300)
    assert painted_columns(canvas, 300) / W < 0.55


def test_draw_line_clamps_both_ends():
    """draw_line was already correct; keep it that way."""
    canvas = blank()
    draw_line(canvas, (-500, 300), (-490, 305), (255, 0, 0))
    assert not (canvas > 0).any()

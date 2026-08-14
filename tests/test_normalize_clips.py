"""Where a striding cut places its clips.

Cutting long footage into bucket-sized clips is arithmetic that is easy to get
subtly wrong, and wrong here is expensive: an off-by-one tail produces a clip
with fewer frames than the bucket, which survives ffmpeg, survives the manifest,
and is only rejected once preprocessing loads it — after the encode.

``span`` is source seconds consumed per clip, which equals the clip's own duration
only when ``--retime`` is 1.0.
"""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "normalize_clips.py"
_spec = importlib.util.spec_from_file_location("normalize_clips", SCRIPT)
normalize_clips = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(normalize_clips)

clip_offsets = normalize_clips.clip_offsets

SPAN = 124 / 24.0  # a 448x768x124 bucket: 5.167 s


def test_no_stride_keeps_the_single_clip_behaviour():
    """The default must not change: one clip, at --start, whatever the duration."""
    assert clip_offsets(duration=30.0, span=SPAN, start=0.0, stride=0.0) == [0.0]
    assert clip_offsets(duration=30.0, span=SPAN, start=2.5, stride=0.0) == [2.5]


def test_stride_walks_the_whole_source():
    offsets = clip_offsets(duration=20.0, span=SPAN, start=0.0, stride=5.0)
    assert offsets == [0.0, 5.0, 10.0]


def test_a_partial_tail_is_dropped_not_padded():
    """14.9 s of source fits two 5.167 s clips at stride 5, not three: the third
    would start at 10.0 and run to 15.167, past the end."""
    assert clip_offsets(duration=14.9, span=SPAN, start=0.0, stride=5.0) == [0.0, 5.0]


def test_stride_below_span_overlaps():
    offsets = clip_offsets(duration=20.0, span=SPAN, start=0.0, stride=2.6)
    assert len(offsets) == 6
    assert offsets[1] - offsets[0] < SPAN  # successive clips share frames


def test_max_clips_caps_the_count():
    offsets = clip_offsets(duration=60.0, span=SPAN, start=0.0, stride=2.6, max_clips=3)
    assert offsets == [0.0, 2.6, 5.2]


def test_start_offsets_every_clip():
    offsets = clip_offsets(duration=20.0, span=SPAN, start=1.0, stride=5.0)
    assert offsets == [1.0, 6.0, 11.0]


def test_source_shorter_than_one_clip_yields_nothing():
    assert clip_offsets(duration=3.0, span=SPAN, start=0.0, stride=2.6) == []

#!/usr/bin/env python
"""Split a manifest into train and held-out parts, deterministically.

    python scripts/split_manifest.py dataset.json --holdout 4

The trainer already carves a validation split out of the cache (`val_split_every`),
but that split is only ever seen as *loss*. Demonstrating conditioning needs
something stronger: media the adapter has never been encoded against at all, so a
generation from it cannot be explained by memorization.

The split is by sorted id and a fixed seed, so re-running gives the same answer --
a demo that quietly changes its held-out set between runs is not a demo.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--holdout", type=int, default=4, help="number of samples to hold out")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = json.loads(args.manifest.read_text())
    if args.holdout >= len(rows):
        raise SystemExit(f"--holdout {args.holdout} leaves nothing to train on ({len(rows)} rows)")

    rows.sort(key=lambda row: row.get("id", ""))
    picked = set(random.Random(args.seed).sample(range(len(rows)), args.holdout))
    held = [row for index, row in enumerate(rows) if index in picked]
    train = [row for index, row in enumerate(rows) if index not in picked]

    train_path = args.manifest.with_name(f"{args.manifest.stem}_train.json")
    held_path = args.manifest.with_name(f"{args.manifest.stem}_heldout.json")
    train_path.write_text(json.dumps(train, indent=1))
    held_path.write_text(json.dumps(held, indent=1))

    print(f"{len(train)} train -> {train_path}")
    print(f"{len(held)} held out -> {held_path}")
    for row in held:
        print(f"  held out {row.get('id')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

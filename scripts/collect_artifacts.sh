#!/usr/bin/env bash
# Surface every run output under artifacts/ inside the repo.
#
# Anything still being written -- the live training run, its evaluations, its
# checkpoints and W&B data, and the dataset -- is **symlinked**, so new results
# appear the moment they land instead of at whatever moment this script last ran.
# Only finished, static evidence is copied.
#
# Safe to re-run at any time.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

REPO="$(cd "$HERE/.." && pwd)"
OUT="$REPO/artifacts"
RUNS="${H3_RUNS:-/data/aviad/runs}"
DATA="${H3_DATASETS:-/data/aviad/datasets}"

mkdir -p "$OUT"/{inference,verification,smoke-runs}

# ------------------------------------------------------------- live symlinks
# One source of truth: these point at the directories the jobs write into.
link() {
    local target="$1" name="$2"
    [ -e "$target" ] || return 0
    rm -rf "${OUT:?}/$name"
    ln -sfn "$target" "$OUT/$name"
}

link "$RUNS"                      "runs"              # every run, live
link "$RUNS/character_av_lora"    "character-run"     # checkpoints, eval_*, logs, wandb
link "$DATA/character"            "character-dataset" # anchor, clips, manifest, latent cache

# ------------------------------------------------------- static evidence (copies)
# These are finished and will not change, so a copy keeps them next to the index
# even if the run directories are later cleaned up.
cp -f "$RUNS"/inference_smoke/*.mp4 "$OUT/inference/" 2>/dev/null || true
cp -f "$DATA"/smoke/.precomputed/decoded_videos/*.mp4 "$OUT/verification/" 2>/dev/null || true

for name in overfit_test iclora_smoke character_feasibility smoke_lowvram smoke_zero3; do
    if [ -d "$RUNS/$name" ]; then
        mkdir -p "$OUT/smoke-runs/$name"
        cp -f "$RUNS/$name"/train.log "$RUNS/$name"/metrics.jsonl "$OUT/smoke-runs/$name/" 2>/dev/null || true
    fi
done

echo "--- live symlinks ---"
ls -l "$OUT" | grep '^l' || true
echo
echo "--- latest evaluations visible now ---"
ls -d "$OUT"/character-run/eval_* 2>/dev/null || echo "(none yet)"
echo
echo "--- checkpoints visible now ---"
ls -d "$OUT"/character-run/checkpoint-* 2>/dev/null | xargs -n1 basename 2>/dev/null || echo "(none yet)"

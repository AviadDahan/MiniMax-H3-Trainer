#!/usr/bin/env bash
# Gather every run output into artifacts/ inside the repo.
#
# Copies, never moves: the originals stay under $H3_RUNS and $H3_DATASETS, so
# re-running this after more training just refreshes the collection. The media
# and weights are gitignored -- artifacts/README.md is the committed index.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

REPO="$(cd "$HERE/.." && pwd)"
OUT="$REPO/artifacts"
RUNS="${H3_RUNS:-/data/aviad/runs}"
DATA="${H3_DATASETS:-/data/aviad/datasets}"

mkdir -p "$OUT"/{inference,character/{anchor,clips,adapters,logs,evaluations},verification,smoke-runs}

# ---------------------------------------------------------------- inference
cp -f "$RUNS"/inference_smoke/*.mp4 "$OUT/inference/" 2>/dev/null || true

# ---------------------------------------------------------------- character
CHAR="$DATA/character"
cp -f "$CHAR"/anchor/* "$OUT/character/anchor/" 2>/dev/null || true
cp -f "$CHAR"/clips/*.mp4 "$OUT/character/clips/" 2>/dev/null || true
cp -f "$CHAR"/dataset.json "$CHAR"/rejected.json "$OUT/character/" 2>/dev/null || true
cp -f "$CHAR"/.precomputed/index.json "$OUT/character/precomputed_index.json" 2>/dev/null || true
cp -rf "$CHAR"/.precomputed/decoded_videos "$OUT/character/vae_decode_check" 2>/dev/null || true

RUN="$RUNS/character_av_lora"
cp -f "$RUN"/train.log "$RUN"/metrics.jsonl "$RUN"/metrics.png "$RUN"/eval_prompts.txt \
      "$OUT/character/logs/" 2>/dev/null || true
# W&B run directories, if any (offline runs are syncable as-is).
cp -rf "$RUN"/wandb "$OUT/character/logs/wandb" 2>/dev/null || true

# Adapters: both layouts per checkpoint. Optimizer state is deliberately left
# behind -- it is resume state, not a deliverable, and it is 2x the adapter size.
for ckpt in "$RUN"/checkpoint-*; do
    [ -d "$ckpt" ] || continue
    step="$(basename "$ckpt" | sed 's/checkpoint-//')"
    cp -f "$ckpt/adapter.safetensors" "$OUT/character/adapters/step${step}_peft.safetensors" 2>/dev/null || true
    cp -f "$ckpt/lora_comfyui.safetensors" "$OUT/character/adapters/step${step}_comfyui.safetensors" 2>/dev/null || true
    cp -f "$ckpt/config.json" "$OUT/character/adapters/step${step}_config.json" 2>/dev/null || true
done

for eval_dir in "$RUN"/eval_*; do
    [ -d "$eval_dir" ] || continue
    cp -rf "$eval_dir" "$OUT/character/evaluations/$(basename "$eval_dir")" 2>/dev/null || true
done

# ------------------------------------------------------- correctness evidence
for name in overfit_test iclora_smoke character_feasibility smoke_lowvram smoke_zero3; do
    if [ -d "$RUNS/$name" ]; then
        mkdir -p "$OUT/smoke-runs/$name"
        cp -f "$RUNS/$name"/train.log "$RUNS/$name"/metrics.jsonl "$OUT/smoke-runs/$name/" 2>/dev/null || true
    fi
done
cp -f "$DATA"/smoke/.precomputed/decoded_videos/*.mp4 "$OUT/verification/" 2>/dev/null || true

echo "--- artifacts ---"
du -sh "$OUT"/*
echo "total: $(du -sh "$OUT" | cut -f1)"

#!/usr/bin/env bash
# Fetch MiniMaxAI/MiniMax-H3 into $HF_HOME and expose it at $H3_MODELS/MiniMax-H3.
#
# The HF repo ships the same weights twice: a diffusers-native flat layout at the
# repo root (transformer/, transformer_ref/, text_encoder/, vae/, audio_vae/, ...)
# and the original MiniMax packaging under FL2VA/ and Ref2VA/ (~144GB each, with a
# custom-code video_vae/audio_vae). Only the flat layout is usable from diffusers,
# so we skip the duplicates: ~210GB instead of ~500GB.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

REPO="MiniMaxAI/MiniMax-H3"
DEST="$H3_MODELS/MiniMax-H3"

mkdir -p "$DEST"
hf download "$REPO" \
    --local-dir "$DEST" \
    --exclude "FL2VA/*" "Ref2VA/*" "assets/*" \
    --max-workers 8

echo "--- downloaded ---"
du -sh "$DEST"
ls "$DEST"

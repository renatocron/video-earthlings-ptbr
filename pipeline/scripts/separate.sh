#!/usr/bin/env bash
# usage: separate.sh <project_dir> [src_lang=en]
set -euo pipefail
P=$(realpath "$1"); LANG_SRC=${2:-en}
if ! command -v demucs >/dev/null; then
  uv tool install --python 3.11 demucs --with soundfile --with "torch==2.5.1" --with "torchaudio==2.5.1" \
    --index https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match
fi
LOCK=${VIDEO_REDUB_GPU_LOCK:-$HOME/.cache/video-redub.gpu.lock}
flock "$LOCK" demucs -n htdemucs_ft --two-stems vocals -d cuda -o "$P/stems" "$P/source/video.wav" >/dev/null 2>&1
ffmpeg -v error -y -i "$P/stems/htdemucs_ft/video/vocals.wav" -ac 1 -ar 16000 "$P/work/vocals16k.wav"
flock "$LOCK" whisper "$P/work/vocals16k.wav" --model turbo --language "$LANG_SRC" --device cuda --word_timestamps True \
  --output_format json --output_dir "$P/work" >/dev/null 2>&1
ls -la "$P/stems/htdemucs_ft/video" "$P/work/vocals16k.json"

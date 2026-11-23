#!/usr/bin/env bash
# usage: fetch.sh <url> <project_dir> [sub_lang=en]
set -euo pipefail
URL=$1; P=$(realpath -m "$2"); SUB=${3:-en}
mkdir -p "$P/source" "$P/stems" "$P/work" "$P/out"
cd "$P/source"
yt-dlp --list-subs --skip-download "$URL" 2>/dev/null | grep -E "^(Language|$SUB[ -])" || echo "no manual '$SUB' subs listed"
yt-dlp -f "bv*+ba/b" --merge-output-format mkv --write-subs --sub-langs "$SUB,$SUB-.*" --convert-subs srt \
  --write-info-json -o "video.%(ext)s" "$URL" >/dev/null
if [ ! -f "video.$SUB.srt" ]; then f=$(ls video.$SUB-*.srt 2>/dev/null | head -1 || true); if [ -n "$f" ]; then mv "$f" "video.$SUB.srt"; fi; fi
ffmpeg -v error -y -i video.mkv -vn -ac 2 -ar 44100 video.wav
ls -la "$P/source"

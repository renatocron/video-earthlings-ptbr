#!/usr/bin/env bash
# Remux each out/video.por.mkv to mp4 (no re-encode) and convert subtitles to WebVTT.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p player/media
for mkv in */out/video.por.mkv; do
  id=${mkv%%/*}
  mp4=player/media/$id.mp4
  if [[ ! -s $mp4 || $mkv -nt $mp4 ]]; then
    ffmpeg -v error -y -i "$mkv" -map 0:v:0 -map 0:a:0 -c copy -movflags +faststart "$mp4"
  fi
  ffmpeg -v error -y -i "$id/out/video.por.srt" "player/media/$id.por.vtt"
  ffmpeg -v error -y -i "$id/out/video.eng.dubtimeline.srt" "player/media/$id.eng.vtt"
  echo "ok $id"
done

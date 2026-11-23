---
name: video-redub
description: Re-dubs a narrated video (YouTube URL or local file) with a new TTS voice (Chatterbox pt-BR clone or Kokoro), optionally translated (pt-br tested). Splits narration from music/SFX with demucs, times the script from the source subtitles plus whisper, translates, then syncs by splitting overflow between voice speed and video slow-mo. Use when the user asks to "redub", "re-voice", "replace the old TTS", "dub this video in pt-br", "make this video sound better", or to run that over a playlist.
---

# Video re-dub

Pipeline proven on the whole Earthlings 101 playlist (16 videos, 3-21 min each, pt-br). With Chatterbox, alignment cleanup and `--voice-first` the dubs run +0% to +5% longer, max slow-mo 1.56x. Scripts are in `scripts/` next to this file; call them with absolute paths and the project dir as an argument (never run code from inside the downloaded folder).

Project layout (one dir per video): `source/` (video.mkv, video.<lang>.srt, video.wav), `stems/`, `work/`, `out/`.

Needs: yt-dlp, ffmpeg, `whisper` CLI, system python3 with `kokoro` + `soundfile` + `numpy`, `uv` (installs demucs on first run). GPU steps take a flock (`~/.cache/video-redub.gpu.lock`) so parallel runs on an 8 GB card queue instead of OOM.

## Steps

S=`pipeline/scripts` in this repo (or the skill's `scripts/` when installed as a Claude Code skill), P=project dir.

1. **Fetch**: `bash $S/fetch.sh <url> $P en`. Takes `en` or any `en-*` variant (en-GB is common) and saves it as `video.en.srt`. If only auto-captions exist, skip `--srt` in step 5 (whisper text is used).
2. **Separate + transcribe**: `bash $S/separate.sh $P en` (demucs htdemucs_ft two-stems, whisper turbo word timestamps on the vocal stem).
3. **Find music/SFX windows**: `python3 $S/analyze.py $P`. Two lists: background-stem activity and vocal-stem sound with no word on it. Demucs often puts melodies and voice-like SFX (Earthlings "microbes" sting) in the *vocals* stem, so the second list matters. Grab frames (`ffmpeg -ss T -frames:v 1`) at candidates to see what is on screen. Pick windows that are music/SFX and contain no narration (start a window after the last word before it, end it before the next word). Quiet beds under narration in the background stem stay automatically, they need no window.
4. **Bed**: `python3 $S/build_bed.py $P --windows a-b,c-d,e-end`. Writes `out/bgm_sfx_bed.wav`, per-window clips, `work/windows.txt`.
5. **Script**: `python3 $S/build_script.py $P --srt $P/source/video.en.srt [--fix 'typo=fixed' ...]`. Text from the SRT (corrupt timestamp lines are dropped), timing from whisper. Read `out/script_src.json`; fix typos via `--fix` and re-run. Check UNMATCHED/OVERLAP output. UNMATCHED usually means whisper turbo skipped a stretch of narration (it happens on long videos): find the gap between the neighbor sentences' times and run `python3 $S/retranscribe.py $P <start> <end>` (splices words into vocals16k.json, keeps a .orig copy), then re-run build_script. An unmatched closing "Thank you for watching" can also be a line that is in the subs but not spoken: check the vocals stem before forcing it. Subtitles sometimes differ from the audio (re-recorded passages): `--fix` them to what is actually spoken.
6. **Translate** (you do this, no MT model): write `work/tgt_texts.txt`, exactly one line per sentence of `script_src.json`, same order. Keep humor, keep it concise (pt runs ~25-30% longer than EN). Numbers: digits are fine ("101"), but fractions and symbols are not ("1/10" reads "um dez", spell them out). Reuse a glossary across episodes of a series. Kokoro chunks at 510 phonemes; very long list sentences are split at commas automatically.
7. **Dub**: `python3 $S/dub.py $P --lang-tag por --kokoro-lang p --voice pm_alex`. Optional `--lexicon lex.json` (word -> IPA), `--base-speed 1.05 --max-speed 1.2`. Output `out/video.<tag>.mkv` with target + source subtitle tracks, `out/dub_report_<tag>.json`. Cached per sentence+speed, so re-runs after a text fix are quick.
7b. **Chatterbox engine (pt-BR, recommended)**: one-time `~/.cache/chatterbox-venv/bin/python $S/setup_chatterbox_ptbr.py` (builds `~/.cache/chatterbox-ptbr`). Render with the chatterbox venv: `~/.cache/chatterbox-venv/bin/python $S/chatterbox_tts.py $P --ref voice.wav` (caches `work/tts_<tag>_cbx/`, aligns every take with MMS_FA and retries runaways with calmer settings). For a cache rendered before alignment existed: `.../python $S/cbx_align.py $P`. Then `python3 $S/dub.py $P --engine chatterbox --max-speed 1.35 --voice-first`. dub.py cuts each clip at the end of the last aligned word (drops Chatterbox's trailing clicks/hallucinated words), shortens silent gaps between words, normalizes clip loudness (LUFS) and times subtitles from the alignment. Re-render bad lines with `--redo 12,57`.
8. **Verify**: `flock ~/.cache/video-redub.gpu.lock whisper $P/out/narration_<tag>.wav --model turbo --language pt --output_format json` and diff against `tgt_texts.txt` (expect ~1-2% WER, mostly spelling, digits vs words, and proper names). Recurring mispronounced words (names, English terms) go into a series-wide lexicon, then re-run dub.py. Whisper hallucinates a closing phrase over trailing silence; check the wav is silent there before believing it. Report the top slow-mo spots (k > 1.6) the dub prints; a line with k > 2 is the first candidate to shorten.

## Gotchas

- Lexicon IPA: use combining diacritics (`u` + U+0303), never precomposed `ũ`/`ẽ`: Kokoro's vocab drops them silently ("101" became "cetu i").
- Kokoro lang codes: p pt-br, e es, f fr, i it. Only pt-br is tested. English target (`a`) uses misaki, subtitle cues fall back to one per sentence if word counts mismatch.
- Old TTS narration is often ~4 words/s; natural voices are ~2.6-3. Sync anchors every sentence at its original start; never stretch music/SFX windows.
- demucs needs torch 2.5.1+cu124 on the 2070 (driver CUDA 12.5); the default uv torch build fails CUDA init.
- Whisper turbo needs ~4 GB; always wrap ad-hoc whisper runs in the flock or parallel jobs OOM.
- Series runs: one subagent per video works. Give them the shared glossary and lexicon; reconcile `work/glossary-new.md` files afterwards, since parallel agents coin different terms for the same thing.
- Waiting on a background step: never poll with `until ! pgrep -f "<pattern>"`, the pattern matches the polling shell itself and the loop never ends (the playlist run left 15 of these). Use the background-task notification, or `pgrep -f "[d]ub.py <dir>"`.
- Chatterbox: no duration control (rubberband time-stretch), ~20-35% of takes have junk or extra words after the last word, and it sometimes runs 2-4x long; alignment-based cut and take scoring handle both. The aligner's min word score is low for one-letter words ("o", "a", "é") and numbers, so gate re-renders on whisper WER of the cleaned clip, not on alignment score alone.
- build_bed.py snaps window edges that fall inside an English word (whisper word times, padded 0.15 s) to the quietest nearby point, and caps each window at +1 LU over the narration. `--no-snap` keeps edges as given.
- Do not publish, upload or commit outputs unless asked.

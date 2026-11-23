"""Find music/SFX candidates: background-stem activity and vocal-stem sounds with no transcribed word.

usage: python3 analyze.py <project_dir>
"""
import json, subprocess, sys
import numpy as np

P = sys.argv[1]
HOP = 0.1


def env_db(path, sr=8000):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    h = int(sr * HOP)
    a = a[: len(a) // h * h].reshape(-1, h)
    return 20 * np.log10(np.sqrt((a ** 2).mean(1)) + 1e-9)


def runs(mask, min_len=0.3, merge_gap=0.6):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            a, b = i * HOP, j * HOP
            if out and a - out[-1][1] <= merge_gap:
                out[-1][1] = b
            else:
                out.append([a, b])
            i = j
        else:
            i += 1
    return [(a, b) for a, b in out if b - a >= min_len]


fmt = lambda t: f"{int(t // 60)}:{t % 60:05.2f}"
bg = env_db(f"{P}/stems/htdemucs_ft/video/no_vocals.wav")
vox = env_db(f"{P}/stems/htdemucs_ft/video/vocals.wav")
covered = np.zeros(len(vox), bool)
words = [w for s in json.load(open(f"{P}/work/vocals16k.json"))["segments"] for w in s["words"]]
for w in words:
    covered[int(w["start"] / HOP):int(w["end"] / HOP) + 2] = True

print("background stem > -45 dB (music beds, ambience, or separation bleed):")
for a, b in runs(bg > -45):
    print(f"  {fmt(a)} - {fmt(b)}  ({b - a:.1f}s, peak {bg[int(a / HOP):int(b / HOP)].max():.1f} dB)")
print("vocal stem > -35 dB with no transcribed word (melodies or voice-like SFX that demucs put in vocals):")
for a, b in runs((vox > -35) & ~covered):
    print(f"  {fmt(a)} - {fmt(b)}  ({b - a:.1f}s, peak {vox[int(a / HOP):int(b / HOP)].max():.1f} dB)")
print(f"narration: {len(words)} words, {words[0]['start']:.1f}s - {words[-1]['end']:.1f}s")

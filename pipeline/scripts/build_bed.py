"""Background bed = no_vocals stem + vocals stem gated to the given music/SFX windows.

usage: python3 build_bed.py <project_dir> --windows 5.0-39.4,241.3-243.9,447.6-end
"""
import argparse, json, os
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("project")
ap.add_argument("--windows", required=True)
ap.add_argument("--no-snap", action="store_true", help="keep window edges exactly as given")
ap.add_argument("--max-over-db", type=float, default=1.0, help="cap each window this many dB above the narrated parts")
args = ap.parse_args()
P = args.project

vox, sr = sf.read(f"{P}/stems/htdemucs_ft/video/vocals.wav")
bg, _ = sf.read(f"{P}/stems/htdemucs_ft/video/no_vocals.wav")
end = len(vox) / sr
windows = [tuple(end if x == "end" else float(x) for x in w.split("-")) for w in args.windows.split(",")]


def snap(x, words, pad=0.15, reach=0.3, max_shift=0.6):
    """Move an edge out of any (padded) English word, then to the quietest vocals point nearby."""
    busy = [(w["start"] - pad, w["end"] + pad) for w in words]
    inside = lambda t: any(a < t < b for a, b in busy)
    if not inside(x):
        return x
    y = x
    for a, b in busy:
        if a < y < b:
            y = a if y - a < b - y else b
    if abs(y - x) > max_shift or inside(y):
        print(f"  edge {x:.2f}: inside speech, left as is")
        return x
    hop = sr // 100
    lo, hi = max(0, int((y - reach) * 100)), min(len(vox) // hop, int((y + reach) * 100))
    cand = [k for k in range(lo, hi) if not inside(k / 100)]
    if not cand:
        return round(y, 2)
    e = [np.sqrt((vox[k * hop:(k + 1) * hop] ** 2).mean()) for k in cand]
    z = cand[int(np.argmin(e))] / 100
    if abs(z - x) > 0.01:
        print(f"  edge {x:.2f} -> {z:.2f}")
    return z


wj = f"{P}/work/vocals16k.json"
if not args.no_snap and os.path.exists(wj):
    words = [w for s in json.load(open(wj))["segments"] for w in s.get("words", [])]
    snapped = [(a if a <= 0 else snap(a, words), b if b >= end else snap(b, words)) for a, b in windows]
    windows = [(x, y) if y - x >= 0.5 * (b - a) else (a, b) for (a, b), (x, y) in zip(windows, snapped)]

mask = np.zeros(len(vox))
fade = int(0.05 * sr)
for a, b in windows:
    i, j = int(a * sr), min(int(b * sr), len(vox))
    m = np.ones(j - i)
    m[:fade] = np.linspace(0, 1, fade)
    m[-fade:] = np.linspace(1, 0, fade)
    mask[i:j] = m
bed = bg + vox * mask[:, None]


def level(x):
    l = pyln.Meter(sr).integrated_loudness(np.concatenate([x, np.zeros((sr,) + x.shape[1:])]))
    return l if np.isfinite(l) else -99


mix = bg + vox
speech = level(mix[mask == 0])
gain = np.ones(len(vox))
for a, b in windows:
    i, j = int(a * sr), min(int(b * sr), len(vox))
    g = min(1.0, 10 ** ((speech + args.max_over_db - level(bed[i:j])) / 20))
    ramp = min(sr, (j - i) // 2)
    gain[i:j] = g
    gain[i:i + ramp] = np.linspace(1, g, ramp)
    gain[j - ramp:j] = np.linspace(g, 1, ramp)
    print(f"window {a:.1f}-{b:.1f}: {20 * np.log10(g):+.1f} dB")
bed *= gain[:, None]
sf.write(f"{P}/out/bgm_sfx_bed.wav", bed, sr)
sf.write(f"{P}/out/narration_old.wav", vox * (1 - mask[:, None]), sr)
for n, (a, b) in enumerate(windows, 1):
    sf.write(f"{P}/out/window{n}_{a:.1f}-{b:.1f}.wav", bed[int(a * sr):int(b * sr)], sr)
open(f"{P}/work/windows.txt", "w").write(",".join(f"{a}-{b}" for a, b in windows))
print("windows:", windows)

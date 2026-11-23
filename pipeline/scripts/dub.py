"""Synthesize the translated script and render the re-dubbed video.

Each sentence starts where the original did. Overflow is split (sqrt) between voice speed (capped)
and video slow-mo over that sentence's span; music/SFX windows are never stretched.

usage: python3 dub.py <project_dir> --lang-tag por [--kokoro-lang p] [--voice pm_alex]
                      [--base-speed 1.05] [--max-speed 1.2] [--lexicon lex.json] [--engine chatterbox]
--engine chatterbox reads clips pre-rendered by chatterbox_tts.py and speeds them up with rubberband.
reads out/script_src.json, work/tgt_texts.txt (one line per sentence), work/windows.txt
"""
import argparse, fcntl, hashlib, json, math, os, re, subprocess, sys
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kokoro_tts import KokoroTTS, SR, VERSION as TTS_VERSION

ap = argparse.ArgumentParser()
ap.add_argument("project")
ap.add_argument("--lang-tag", default="por", help="ISO 639-2 tag for the audio/sub tracks")
ap.add_argument("--kokoro-lang", default="p")
ap.add_argument("--voice", default="pm_alex")
ap.add_argument("--engine", choices=["kokoro", "chatterbox"], default="kokoro")
ap.add_argument("--base-speed", type=float)
ap.add_argument("--max-speed", type=float, default=1.2)
ap.add_argument("--voice-first", action="store_true", help="speed the voice up to --max-speed before slowing the video")
ap.add_argument("--pause", type=float, default=0.35)
ap.add_argument("--lexicon")
ap.add_argument("--src-lang-tag", default="eng")
args = ap.parse_args()
P = os.path.abspath(args.project)
SRC, TAG = f"{P}/source/video.mkv", args.lang_tag
CBX = args.engine == "chatterbox"
if args.base_speed is None:
    args.base_speed = 1.0 if CBX else 1.05
TTS_DIR = f"{P}/work/tts_{TAG}_{'cbx' if CBX else args.voice}"
os.makedirs(TTS_DIR, exist_ok=True)

script = json.load(open(f"{P}/out/script_src.json"))
tgt = [l.strip() for l in open(f"{P}/work/tgt_texts.txt", encoding="utf-8") if l.strip()]
assert len(tgt) == len(script), f"{len(tgt)} translated lines for {len(script)} sentences"
for c, t in zip(script, tgt):
    c["text_tgt"] = t
total = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                       "-of", "csv=p=0", SRC]))
fps = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                               "stream=r_frame_rate", "-of", "csv=p=0", SRC]).decode().strip()
windows = [tuple(float(x) for x in w.split("-")) for w in open(f"{P}/work/windows.txt").read().split(",") if w]

lex = json.load(open(args.lexicon)) if args.lexicon else {}
LOCK = os.environ.get("VIDEO_REDUB_GPU_LOCK", os.path.expanduser("~/.cache/video-redub.gpu.lock"))
lockf = open(LOCK, "w")
if CBX:
    cbx_index = {int(k): v for k, v in json.load(open(f"{TTS_DIR}/index.json")).items()}
else:
    fcntl.flock(lockf, fcntl.LOCK_EX)
    tts = KokoroTTS(args.kokoro_lang, args.voice, lex)


def _fade(seg, inn=0.005, out=0.005):
    seg = seg.copy()
    i, o = min(len(seg), int(inn * SR)), min(len(seg), int(out * SR))
    if i:
        seg[:i] *= np.linspace(0, 1, i)
    if o:
        seg[-o:] *= np.linspace(1, 0, o)
    return seg


def clean_cbx(a, max_gap=0.35, keep=0.25, burst=0.4, burst_gap=0.12, lead=0.05, tail=0.25):
    # Chatterbox holds long silences mid-sentence and often emits a click/thump after the last word
    hop = SR // 100
    n = len(a) // hop
    db = 20 * np.log10(np.sqrt((a[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
    quiet = db < db.max() - 40
    runs, i = [], 0
    while i < n:
        j = i
        while j < n and quiet[j] == quiet[i]:
            j += 1
        runs.append([i, j, bool(quiet[i])])
        i = j
    while runs and runs[-1][2]:
        runs.pop()
    while runs and runs[0][2]:
        runs.pop(0)
    def junk(r, gap):
        # a click/thump, a hiss, or a short sound after a long silence; a voiced final word after a comma stays
        t, seg = (r[1] - r[0]) / 100, a[r[0] * hop:r[1] * hop]
        s = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) + 1e-9
        flat = np.exp(np.log(s).mean()) / s.mean()
        return gap >= burst_gap and (t < 0.13 or flat > 0.3 or (t < burst and gap >= 0.25))

    for _ in range(2):
        if len(runs) >= 3 and junk(runs[-1], (runs[-2][1] - runs[-2][0]) / 100):
            runs = runs[:-2]
        if len(runs) >= 3 and (runs[0][1] - runs[0][0]) / 100 < burst_gap and (runs[1][1] - runs[1][0]) / 100 >= burst_gap:
            runs = runs[2:]
    if not runs:
        return a
    runs[0][0] = max(0, runs[0][0] - int(lead * 100))
    runs[-1][1] = min(n, runs[-1][1] + int(tail * 100))
    out = []
    for s, e, q in runs:
        seg = a[s * hop:e * hop]
        if q and (e - s) / 100 > max_gap:
            h = int(keep * SR / 2)
            out += [_fade(seg[:h], 0), _fade(seg[-h:], out=0)]
        else:
            out.append(seg)
    return _fade(np.concatenate(out), 0.005, 0.04)


def lufs(x, sr=SR):
    pad = max(0, int(0.5 * sr) - len(x))
    return pyln.Meter(sr).integrated_loudness(np.concatenate([x, np.zeros((pad,) + x.shape[1:])]))


def clip_loudness(a, target=-20.0):
    l = lufs(a)
    return a * 10 ** ((target - l) / 20) if np.isfinite(l) else a


def clean_aligned(a, al, max_gap=0.35, keep=0.25, lead=0.05, tail=0.06):
    """Keep first word to end of voice, shorten long gaps between words. Returns audio and word times."""
    w = al["words"]
    hop = SR // 100
    n = len(a) // hop
    db = 20 * np.log10(np.sqrt((a[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
    loud = np.flatnonzero(db > db.max() - 40)
    # short words ("a", "D") can align late, so never start after the first real sound
    onset = loud[0] / 100 if loud.size else al["voice_start"]
    first = onset if al["voice_start"] - 0.6 < onset < al["voice_start"] else al["voice_start"]
    cuts = [max(0.0, first - lead)]
    for x, y in zip(w, w[1:]):
        s, e = x["end"] + keep / 2, y["start"] - keep / 2
        if e - s > max_gap - keep and (db[int(s * 100):int(e * 100)] < db.max() - 35).all():
            cuts += [s, e]
    cuts.append(min(len(a) / SR, al["voice_end"] + tail))
    spans = list(zip(cuts[::2], cuts[1::2]))
    out = [_fade(a[int(s * SR):int(e * SR)]) for s, e in spans]

    def remap_t(t):
        acc = 0.0
        for s, e in spans:
            if t <= e:
                return acc + max(0.0, t - s)
            acc += e - s
        return acc

    words = [{"text": x["text"], "start_ts": remap_t(x["start"]), "end_ts": remap_t(x["end"])} for x in w]
    return _fade(np.concatenate(out), 0.005, 0.04), words


def synth_cbx(c, speed):
    base = cbx_index[c["id"]]
    al = json.load(open(base + ".align.json")) if os.path.exists(base + ".align.json") else None
    if al:
        path = base[:-4] + f"_al_{speed:.3f}.wav"
        if not os.path.exists(path + ".json"):
            a, _ = sf.read(base)
            a, words = clean_aligned(a, al)
            a = clip_loudness(a)
            tmp = path + ".tmp.wav"
            sf.write(tmp, a, SR)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-af", f"rubberband=tempo={speed}",
                            "-ar", str(SR), "-ac", "1", path], check=True)
            os.remove(tmp)
            json.dump([{"text": x["text"], "start_ts": round(x["start_ts"] / speed, 3),
                        "end_ts": round(x["end_ts"] / speed, 3)} for x in words], open(path + ".json", "w"),
                      ensure_ascii=False)
        return path, json.load(open(path + ".json"))
    words = json.load(open(base + ".json"))
    path = base[:-4] + f"_cl_{speed:.3f}.wav"
    if not os.path.exists(path):
        a, _ = sf.read(base)
        tmp = path + ".tmp.wav"
        sf.write(tmp, clean_cbx(a), SR)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-af", f"rubberband=tempo={speed}",
                        "-ar", str(SR), "-ac", "1", path], check=True)
        os.remove(tmp)
    dur = sf.info(path).duration
    span, lead = max(0.1, dur - 0.3), 0.05
    w = np.cumsum([0] + [len(t["text"]) + 1 for t in words])
    return path, [{"text": t["text"], "start_ts": lead + w[k] / w[-1] * span, "end_ts": lead + w[k + 1] / w[-1] * span}
                  for k, t in enumerate(words)]


def synth(c, speed):
    if CBX:
        return synth_cbx(c, speed)
    key = hashlib.sha1((c["text_tgt"] + json.dumps(lex, sort_keys=True) + f"v{TTS_VERSION}").encode()).hexdigest()[:8]
    path = f"{TTS_DIR}/s{c['id']:03d}_{speed:.3f}_{key}.wav"
    meta = path + ".json"
    if not os.path.exists(meta):
        json.dump(tts.synth(c["text_tgt"], path, speed), open(meta, "w"))
    return path, json.load(open(meta))


for i, c in enumerate(script):
    nxt = script[i + 1]["start"] if i + 1 < len(script) else total
    c["limit"] = min([nxt] + [a for a, _ in windows if c["start"] < a])
    avail = c["limit"] - c["start"]
    speed = args.base_speed
    path, tok = synth(c, speed)
    dur = sf.info(path).duration
    over = (dur + args.pause) / avail
    if over > 1:
        speed = round(min(args.max_speed, args.base_speed * (over if args.voice_first else math.sqrt(over))), 3)
        path, tok = synth(c, speed)
        dur = sf.info(path).duration
    c.update(wav=path, tokens=tok, dur=dur, speed=speed, k=max(1.0, (dur + args.pause) / avail))
    print(f"s{c['id']:03d} speed={speed} k={c['k']:.2f}", flush=True)

if not CBX:
    del tts
    import torch
    torch.cuda.empty_cache()
    fcntl.flock(lockf, fcntl.LOCK_UN)

segs, t = [], 0.0
for c in script:
    if c["start"] > t:
        segs.append((t, c["start"], 1.0))
    segs.append((c["start"], c["limit"], c["k"]))
    t = c["limit"]
segs.append((t, total, 1.0))


def remap(x):
    acc = 0.0
    for a, b, k in segs:
        if x <= b:
            return acc + (max(x, a) - a) * k
        acc += (b - a) * k
    return acc


new_total = remap(total)
narr = np.zeros(int(new_total * SR) + SR)
for c in script:
    c["new_start"] = remap(c["start"])
    a, _ = sf.read(c["wav"])
    i = int(c["new_start"] * SR)
    narr[i:i + len(a)] += a[: len(narr) - i]


old, osr = sf.read(f"{P}/out/narration_old.wav")
narr *= 10 ** ((lufs(old, osr) - lufs(narr)) / 20)
sf.write(f"{P}/out/narration_{TAG}.wav", np.clip(narr, -1, 1), SR)


def ts(x):
    ms = int(round(x * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def chunk_cues(c, maxlen=84):
    words, tok = c["text_tgt"].split(), c["tokens"]
    if len(tok) != len(words):
        return [(c["new_start"], c["new_start"] + c["dur"], c["text_tgt"])]
    cues, cur, t0, prev = [], [], tok[0]["start_ts"], 0.0
    for w, tk in zip(words, tok):
        if cur and len(" ".join(cur + [w])) > maxlen:
            cues.append((t0, prev, " ".join(cur)))
            cur, t0 = [], tk["start_ts"]
        cur.append(w)
        prev = tk["end_ts"]
    cues.append((t0, prev, " ".join(cur)))
    return [(c["new_start"] + a, c["new_start"] + b, txt) for a, b, txt in cues]


def wrap2(txt, width=42):
    if len(txt) <= width:
        return txt
    cut = min((m.start() for m in re.finditer(" ", txt)), key=lambda p: abs(p - len(txt) // 2))
    return txt[:cut] + "\n" + txt[cut + 1:]


cues = [q for c in script for q in chunk_cues(c)]
with open(f"{P}/out/video.{TAG}.srt", "w", encoding="utf-8") as f:
    for n, (a, b, txt) in enumerate(cues, 1):
        nxt = cues[n][0] if n < len(cues) else b + 1
        f.write(f"{n}\n{ts(a)} --> {ts(min(b + 0.15, nxt - 0.02))}\n{wrap2(txt)}\n\n")
with open(f"{P}/out/video.{args.src_lang_tag}.dubtimeline.srt", "w", encoding="utf-8") as f:
    for c in script:
        f.write(f"{c['id']}\n{ts(c['new_start'])} --> {ts(remap(c['end']))}\n{c['text_src']}\n\n")

parts, vlab, alab = [], [], []
for j, (a, b, k) in enumerate(segs):
    if b - a < 0.01:
        continue
    r = 1 / math.sqrt(k)
    parts.append(f"[0:v]trim={a:.3f}:{b:.3f},setpts=(PTS-STARTPTS)*{k:.4f}[v{j}]")
    parts.append(f"[1:a]atrim={a:.3f}:{b:.3f},asetpts=PTS-STARTPTS,atempo={r:.4f},atempo={r:.4f}[a{j}]")
    vlab.append(f"[v{j}]")
    alab.append(f"[a{j}]")
parts.append(f"{''.join(vlab)}concat=n={len(vlab)}:v=1:a=0,fps={fps}[vout]")
parts.append(f"{''.join(alab)}concat=n={len(alab)}:v=0:a=1[bed]")
parts.append("[2:a]aresample=44100,pan=stereo|c0=c0|c1=c0[nar]")
parts.append("[bed][nar]amix=inputs=2:normalize=0:duration=first,alimiter=limit=0.95[aout]")
open(f"{P}/work/filter_{TAG}.txt", "w").write(";\n".join(parts))

subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", SRC, "-i", f"{P}/out/bgm_sfx_bed.wav",
                "-i", f"{P}/out/narration_{TAG}.wav", "-i", f"{P}/out/video.{TAG}.srt",
                "-i", f"{P}/out/video.{args.src_lang_tag}.dubtimeline.srt",
                "-filter_complex_script", f"{P}/work/filter_{TAG}.txt",
                "-map", "[vout]", "-map", "[aout]", "-map", "3:s", "-map", "4:s",
                "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-c:a", "aac", "-b:a", "192k",
                "-c:s", "srt", f"-metadata:s:a:0", f"language={TAG}", "-metadata:s:s:0", f"language={TAG}",
                "-metadata:s:s:1", f"language={args.src_lang_tag}", "-disposition:s:0", "default",
                f"{P}/out/video.{TAG}.mkv"], check=True)

json.dump([{k: c[k] for k in ("id", "start", "limit", "new_start", "dur", "speed", "k", "text_tgt")}
           for c in script], open(f"{P}/out/dub_report_{TAG}.json", "w"), indent=1, ensure_ascii=False)
ks = [c["k"] for c in script]
print(f"\n{total:.1f}s -> {new_total:.1f}s (+{(new_total / total - 1) * 100:.0f}%), "
      f"stretched {sum(k > 1 for k in ks)}/{len(ks)}, max k {max(ks):.2f}")
for c in sorted(script, key=lambda c: -c["k"])[:6]:
    m = int(c["new_start"])
    print(f"  {m // 60}:{m % 60:02d} s{c['id']:03d} k={c['k']:.2f} speed={c['speed']} {c['text_tgt'][:50]}")
print(f"-> {P}/out/video.{TAG}.mkv")

"""Pre-render every sentence with Chatterbox (pt-BR finetune) into the cache dub.py --engine chatterbox reads.

Runs in its own venv (torch 2.6 cu124), not the system python:
  ~/.cache/chatterbox-venv/bin/python chatterbox_tts.py <project_dir> --ref voice.wav [--ckpt ~/.cache/chatterbox-ptbr]
Writes work/tts_<tag>_cbx/s<id>_<key>.wav at natural pace plus .json with proportional word timings.
"""
import argparse, fcntl, hashlib, json, os, re, sys, time
import numpy as np
import perth
import torch
import torchaudio
from num2words import num2words

perth.PerthImplicitWatermarker = perth.DummyWatermarker
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cbx_align import Aligner

VERSION = "cbx1"
SR = 24000
MAX_CHARS = 250
# (seed, temperature, cfg_weight); later takes are calmer, which curbs runaways
TAKES = [(0, 0.8, 0.5), (1, 0.8, 0.5), (2, 0.65, 0.4), (3, 0.65, 0.3), (4, 0.5, 0.3), (5, 0.5, 0.3)]

ap = argparse.ArgumentParser()
ap.add_argument("project")
ap.add_argument("--ref", required=True, help="reference voice wav to clone")
ap.add_argument("--ckpt", default=os.path.expanduser("~/.cache/chatterbox-ptbr"))
ap.add_argument("--lang-tag", default="por")
ap.add_argument("--lang", default="pt")
ap.add_argument("--respell", help="json word -> spelling Chatterbox reads right")
ap.add_argument("--redo", default="", help="comma-separated sentence ids to render again")
ap.add_argument("--min-score", type=float, default=0.5, help="worst aligned word score to accept a take")
args = ap.parse_args()
P = os.path.abspath(args.project)
OUT = f"{P}/work/tts_{args.lang_tag}_cbx"
os.makedirs(OUT, exist_ok=True)
respell = json.load(open(args.respell)) if args.respell else {}
ref_id = hashlib.sha1(open(args.ref, "rb").read()).hexdigest()[:8]

script = json.load(open(f"{P}/out/script_src.json"))
tgt = [l.strip() for l in open(f"{P}/work/tgt_texts.txt", encoding="utf-8") if l.strip()]
assert len(tgt) == len(script)


def spoken(text):
    for w, r in respell.items():
        text = re.sub(rf"\b{re.escape(w)}\b", r, text)
    text = re.sub(r"(\d+)\s*°", r"\1 graus", text)
    text = re.sub(r"(\d+)\s*%", r"\1 por cento", text)
    text = re.sub(r"\d+(?:,\d+)?", lambda m: num2words(float(m.group().replace(",", ".")) if "," in m.group()
                                                       else int(m.group()), lang="pt_BR"), text)
    return text


def pieces(text):
    if len(text) <= MAX_CHARS:
        return [text]
    out, cur = [], ""
    for part in re.split(r"(?<=[.;:!?,])\s+", text):
        if cur and len(cur) + len(part) + 1 > MAX_CHARS:
            out.append(cur)
            cur = part
        else:
            cur = f"{cur} {part}".strip()
    return out + [cur] if cur else out


def trim(a, lead=0.05, tail=0.25):
    loud = np.flatnonzero(np.abs(a) > 0.01)
    if not loud.size:
        return a
    return a[max(0, loud[0] - int(lead * SR)): loud[-1] + int(tail * SR)]


def tokens(words, dur, lead=0.05, tail=0.25):
    # no alignment from Chatterbox: spread words over the voiced span by character count
    span = max(0.1, dur - lead - tail)
    w = np.array([len(x) + 1 for x in words], float)
    edges = lead + np.concatenate([[0], np.cumsum(w)]) / w.sum() * span
    return [{"text": x, "start_ts": round(float(a), 3), "end_ts": round(float(b), 3)}
            for x, a, b in zip(words, edges[:-1], edges[1:])]


redo = {int(x) for x in args.redo.split(",") if x}
todo, index = [], {}
for c, t in zip(script, tgt):
    key = hashlib.sha1((t + json.dumps(respell, sort_keys=True) + ref_id + VERSION).encode()).hexdigest()[:8]
    path = f"{OUT}/s{c['id']:03d}_{key}.wav"
    index[c["id"]] = path
    if c["id"] in redo:
        for f in os.listdir(OUT):
            if f.startswith(os.path.basename(path)[:-4]):
                os.remove(f"{OUT}/{f}")
    if not os.path.exists(path + ".json"):
        todo.append((c, t, path))
json.dump(index, open(f"{OUT}/index.json", "w"), indent=0)
print(f"{P}: {len(todo)}/{len(script)} sentences to render", flush=True)
if not todo:
    sys.exit()

lockf = open(os.path.expanduser("~/.cache/video-redub.gpu.lock"), "w")
fcntl.flock(lockf, fcntl.LOCK_EX)
model = ChatterboxMultilingualTTS.from_local(args.ckpt, "cuda")
model.prepare_conditionals(args.ref)
aligner = Aligner("cpu")

for n, (c, t, path) in enumerate(todo, 1):
    t0 = time.time()
    parts = []
    for piece in pieces(spoken(t)):
        best, best_q = None, -9.0
        for seed, temp, cfg in (TAKES[1:] if c["id"] in redo else TAKES):
            torch.manual_seed(seed)
            a = trim(model.generate(piece, language_id=args.lang, temperature=temp, cfg_weight=cfg)
                     .squeeze(0).cpu().numpy())
            r = aligner(a, piece.split())
            score = r["min_score"] if r else 0.0
            voiced = (r["voice_end"] - r["voice_start"]) if r else len(a) / SR
            ratio = voiced / max(0.4, len(piece) / 15)
            q = score - abs(np.log(ratio))
            if q > best_q:
                best, best_q = a[max(0, int((r["voice_start"] - 0.05) * SR)):int((r["voice_end"] + 0.25) * SR)] if r else a, q
            if score >= args.min_score and 0.6 < ratio < 1.6:
                break
            print(f"  s{c['id']:03d} take {seed}: score {score:.2f} ratio {ratio:.2f}, retrying", flush=True)
        parts += [best, np.zeros(int(0.15 * SR))]
    a = np.concatenate(parts[:-1])
    torchaudio.save(path, torch.from_numpy(a).unsqueeze(0).float(), SR)
    json.dump(tokens(t.split(), len(a) / SR), open(path + ".json", "w"), ensure_ascii=False)
    json.dump(aligner(a, t.split()), open(path + ".align.json", "w"), ensure_ascii=False)
    print(f"s{c['id']:03d} {len(a) / SR:.1f}s in {time.time() - t0:.1f}s ({n}/{len(todo)})", flush=True)

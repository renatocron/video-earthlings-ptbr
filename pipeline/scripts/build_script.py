"""Sentence script with timings: text from the source SRT (or whisper if none), timing from whisper words.

usage: python3 build_script.py <project_dir> [--srt source/video.en.srt] [--fix 'hydic=hydric' ...]
"""
import argparse, difflib, json, os, re

ap = argparse.ArgumentParser()
ap.add_argument("project")
ap.add_argument("--srt")
ap.add_argument("--fix", action="append", default=[], help="typo fix old=new, repeatable")
args = ap.parse_args()
P = args.project

words = [w for s in json.load(open(f"{P}/work/vocals16k.json"))["segments"] for w in s["words"]]
if args.srt and os.path.exists(args.srt):
    raw = open(args.srt, encoding="utf-8").read()
    lines = [l for l in raw.splitlines() if l.strip() and "-->" not in l and not l.strip().isdigit()
             and not re.match(r"^\d+:\d\d:\d\d[.,]\d+,", l)]
    text = re.sub(r"<[^>]+>", "", " ".join(l.strip() for l in lines))
else:
    print("no SRT, using whisper text")
    text = "".join(w["word"] for w in words)
for f in args.fix:
    a, b = f.split("=", 1)
    text = text.replace(a, b)
text = re.sub(r"\s+", " ", text).strip()
sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z"¿¡])', text) if s.strip()]

norm = lambda w: re.sub(r"[^\w]", "", w.lower())
wt = [norm(w["word"]) for w in words]
toks, owner = [], []
for i, s in enumerate(sents):
    for t in s.split():
        if norm(t):
            toks.append(norm(t))
            owner.append(i)
pos = {}
for blk in difflib.SequenceMatcher(None, toks, wt, autojunk=False).get_matching_blocks():
    for k in range(blk.size):
        pos[blk.a + k] = blk.b + k

out, missing = [], []
for i, s in enumerate(sents):
    idx = [pos[k] for k in range(len(toks)) if owner[k] == i and k in pos]
    if not idx:
        missing.append(s)
        continue
    out.append({"id": len(out) + 1, "start": round(words[min(idx)]["start"], 2),
                "end": round(words[max(idx)]["end"], 2), "text_src": s})
json.dump(out, open(f"{P}/out/script_src.json", "w"), indent=1, ensure_ascii=False)
for s in missing:
    print("UNMATCHED (dropped):", s)
for a, b in zip(out, out[1:]):
    if b["start"] < a["end"]:
        print(f"OVERLAP s{a['id']}/s{b['id']} at {b['start']}")
rate = sorted(len(c["text_src"].split()) / max(c["end"] - c["start"], .01) for c in out)[len(out) // 2]
print(f"{len(out)} sentences timed, median {rate:.1f} words/s -> out/script_src.json")

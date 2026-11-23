"""Re-transcribe a time range whisper skipped and splice its words into work/vocals16k.json.

usage: python3 retranscribe.py <project_dir> <start_s> <end_s> [--model small] [--lang en]
"""
import argparse, json, os, shutil, subprocess, tempfile

ap = argparse.ArgumentParser()
ap.add_argument("project")
ap.add_argument("start", type=float)
ap.add_argument("end", type=float)
ap.add_argument("--model", default="small")
ap.add_argument("--lang", default="en")
args = ap.parse_args()
P = os.path.abspath(args.project)
src = f"{P}/work/vocals16k.json"
lock = os.path.expanduser("~/.cache/video-redub.gpu.lock")

with tempfile.TemporaryDirectory() as tmp:
    clip = f"{tmp}/clip.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(args.start), "-to", str(args.end),
                    "-i", f"{P}/work/vocals16k.wav", clip], check=True)
    subprocess.run(["flock", lock, "whisper", clip, "--model", args.model, "--language", args.lang,
                    "--word_timestamps", "True", "--output_format", "json", "--output_dir", tmp],
                   check=True, capture_output=True)
    new = json.load(open(f"{tmp}/clip.json"))["segments"]

for s in new:
    s["start"] += args.start
    s["end"] += args.start
    for w in s["words"]:
        w["start"] += args.start
        w["end"] += args.start

d = json.load(open(src))
if not os.path.exists(src.replace(".json", ".orig.json")):
    shutil.copy(src, src.replace(".json", ".orig.json"))
for s in d["segments"]:
    s["words"] = [w for w in s["words"] if not (args.start <= w["start"] < args.end)]
norm = lambda w: "".join(c for c in w["word"].lower() if c.isalnum())
edge = [w for s in d["segments"] for w in s["words"]
        if abs(w["start"] - args.start) < 1.0 or abs(w["start"] - args.end) < 1.0]
for s in new:
    s["words"] = [w for w in s["words"]
                  if not any(norm(w) == norm(o) and abs(w["start"] - o["start"]) < 0.5 for o in edge)]
new = [s for s in new if s["words"]]
d["segments"] = sorted([s for s in d["segments"] if s["words"]] + new, key=lambda s: s["start"])
json.dump(d, open(src, "w"), ensure_ascii=False)
print(f"spliced {sum(len(s['words']) for s in new)} words in {args.start}-{args.end}s:")
print(" ".join(s["text"].strip() for s in new))

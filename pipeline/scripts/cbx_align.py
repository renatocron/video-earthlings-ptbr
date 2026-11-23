"""CTC forced alignment (torchaudio MMS_FA) of TTS clips against their text.

Library for chatterbox_tts.py, and a pass over an existing cache (chatterbox venv):
  ~/.cache/chatterbox-venv/bin/python cbx_align.py <project_dir> [--lang-tag por]
Writes <clip>.wav.align.json: per original word start/end/score, plus where the voice really ends.
"""
import argparse, fcntl, json, os, re, sys, unicodedata
import numpy as np
import torch
import torchaudio
from num2words import num2words

SR = 24000
FA_SR = 16000
FRAME = 0.02


def spoken(text):
    text = re.sub(r"(\d+)\s*°", r"\1 graus", text)
    text = re.sub(r"(\d+)\s*%", r"\1 por cento", text)
    return re.sub(r"\d+(?:,\d+)?", lambda m: num2words(float(m.group().replace(",", ".")) if "," in m.group()
                                                       else int(m.group()), lang="pt_BR"), text)


class Aligner:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        b = torchaudio.pipelines.MMS_FA
        self.model = b.get_model(with_star=True).to(self.device).eval()
        self.dict = b.get_dict(star="*")
        self.align_fn = b.get_aligner()

    def norm(self, word):
        s = unicodedata.normalize("NFKD", spoken(word).lower())
        return "".join(c for c in s if c in self.dict and c not in "-*")

    @torch.inference_mode()
    def __call__(self, audio, words):
        """audio: float array at 24 kHz. words: text.split(). Returns dict or None."""
        wav = torch.from_numpy(np.asarray(audio, np.float32)).unsqueeze(0)
        wav = torchaudio.functional.resample(wav, SR, FA_SR).to(self.device)
        emission, _ = self.model(wav)
        keep = [(i, n) for i, n in enumerate(self.norm(w) for w in words) if n]
        if not keep:
            return None
        seq = ["*"] + [n for _, n in keep] + ["*"]
        spans = self.align_fn(emission[0], [[self.dict[c] for c in w] for w in seq])
        ratio = wav.shape[1] / emission.shape[1] / FA_SR
        out = [None] * len(words)
        for (i, _), sp in zip(keep, spans[1:-1]):
            sc = float(np.mean([t.score for t in sp]))
            out[i] = {"text": words[i], "start": round(sp[0].start * ratio, 3), "end": round(sp[-1].end * ratio, 3),
                      "score": round(sc, 3)}
        for i in range(len(out)):
            if out[i] is None:
                prev = next((o for o in reversed(out[:i]) if o), None)
                t = prev["end"] if prev else (spans[1][0].start * ratio)
                out[i] = {"text": words[i], "start": t, "end": t, "score": None}
        scores = [o["score"] for o in out if o["score"] is not None]
        return {"words": out, "dur": len(audio) / SR, "min_score": round(min(scores), 3),
                "mean_score": round(float(np.mean(scores)), 3),
                "voice_start": out[0]["start"], "voice_end": voice_end(audio, out[-1]["end"])}


def voice_end(audio, end, max_ext=0.3, quiet_db=40):
    """Where the last word's sound decays, searching from the aligned end, capped at end+max_ext."""
    hop = SR // 100
    n = len(audio) // hop
    db = 20 * np.log10(np.sqrt((audio[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
    q = db < db.max() - quiet_db
    i, stop = int(end * 100), min(n, int((end + max_ext) * 100))
    while i < stop and not q[i:i + 5].all():
        i += 1
    return i / 100


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--lang-tag", default="por")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    import soundfile as sf
    P = os.path.abspath(args.project)
    d = f"{P}/work/tts_{args.lang_tag}_cbx"
    index = json.load(open(f"{d}/index.json"))
    script = json.load(open(f"{P}/out/script_src.json"))
    tgt = [l.strip() for l in open(f"{P}/work/tgt_texts.txt", encoding="utf-8") if l.strip()]
    text = {str(c["id"]): t for c, t in zip(script, tgt)}
    todo = [(k, v) for k, v in index.items() if args.force or not os.path.exists(v + ".align.json")]
    if not todo:
        sys.exit()
    lockf = open(os.path.expanduser("~/.cache/video-redub.gpu.lock"), "w")
    fcntl.flock(lockf, fcntl.LOCK_EX)
    al = Aligner()
    for k, v in todo:
        a, _ = sf.read(v)
        r = al(a, text[k].split())
        json.dump(r, open(v + ".align.json", "w"), ensure_ascii=False)
    print(f"{P}: aligned {len(todo)}", flush=True)

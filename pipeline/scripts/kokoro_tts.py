"""Kokoro for non-English targets: phonemize each sentence in context, word timestamps from pred_dur.

Lexicon IPA must use combining diacritics (u + U+0303), precomposed chars like ũ are not in the vocab.
"""
import re
import numpy as np
import soundfile as sf
from kokoro import KPipeline

SR, FPS = 24000, 40
TAIL_KEEP = 0.25
TAIL_LIFT_DB = 12
VERSION = 2
_PUNCT = ".,;:!?\"'()"
_PLACEHOLDER = "papa"


class KokoroTTS:
    def __init__(self, lang="p", voice="pm_alex", lexicon=None):
        self.pipe = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M")
        self.voice = self.pipe.load_voice(voice)
        self.lex = lexicon or {}
        self._cache = {}

    def _g2p(self, s):
        if s not in self._cache:
            self._cache[s] = self.pipe.g2p(s)[0].split()
        return self._cache[s]

    def words(self, text):
        vocab = self.pipe.model.vocab
        runs = []
        for surface in re.split(r"\s+", text.strip()):
            core = surface.strip(_PUNCT)
            if core and core in self.lex:
                i = surface.index(core)
                runs.append((surface, surface[:i] + self.lex[core] + surface[i + len(core):],
                             surface[:i] + _PLACEHOLDER + surface[i + len(core):]))
            else:
                runs.append((surface, None, surface))
        ctx = self._g2p(" ".join(r[2] for r in runs))
        counts = [1 if k is not None else len(self._g2p(s)) for s, k, _ in runs]
        if sum(counts) != len(ctx):
            ctx = [w for s, k, _ in runs for w in ([k] if k is not None else self._g2p(s))]
        out, i = [], 0
        for (surface, known, _), n in zip(runs, counts):
            ps = known if known is not None else " ".join(ctx[i:i + n])
            i += n
            out.append((surface.strip(_PUNCT), "".join(c for c in ps if c in vocab)))
        return out

    def synth(self, text, path, speed=1.0):
        words = self.words(text)
        chunks, cur, n = [], [], 0
        for w in words:
            cur.append(w)
            n += len(w[1]) + 1
            if (n > 380 and w[1][-1:] in ".!?;:") or (n > 440 and w[1][-1:] == ",") or n > 480:
                chunks.append(cur)
                cur, n = [], 0
        if cur:
            chunks.append(cur)
        audio, tokens, offset = [], [], 0.0
        for ch in chunks:
            ps = " ".join(p for _, p in ch)
            assert len(ps) <= 510, f"chunk of {len(ps)} phonemes, split the sentence"
            res = KPipeline.infer(self.pipe.model, ps, self.voice, speed)
            dur = res.pred_dur.tolist()
            i, frames = 1, dur[0]
            for surface, p in ch:
                start = frames
                frames += sum(dur[i:i + len(p)])
                i += len(p)
                if surface:
                    tokens.append({"text": surface, "start_ts": round(offset + start / FPS, 3),
                                   "end_ts": round(offset + frames / FPS, 3)})
                if i < len(dur) - 1:
                    frames += dur[i]
                    i += 1
            a = res.audio.cpu().numpy()
            audio.append(a)
            offset += len(a) / SR
        full = np.concatenate(audio)
        loud = np.flatnonzero(np.abs(full) > 0.005)
        if loud.size:
            full = full[:min(len(full), loud[-1] + 1 + int(TAIL_KEEP * SR))]
        full = self._lift_tail(full, tokens)
        sf.write(path, full, SR)
        return tokens

    @staticmethod
    def _lift_tail(full, tokens):
        # Kokoro fades the last word to a whisper; ramp it up, never above the sentence peak
        if not tokens or TAIL_LIFT_DB <= 0:
            return full
        a = int(tokens[-1]["start_ts"] * SR)
        b = min(len(full), int(tokens[-1]["end_ts"] * SR))
        if b - a < 2:
            return full
        gain = np.ones(len(full))
        gain[a:b] = 10 ** (np.linspace(0, TAIL_LIFT_DB, b - a) / 20)
        gain[b:] = gain[b - 1]
        peak = np.abs(full).max()
        return np.clip(full * gain, -peak, peak)

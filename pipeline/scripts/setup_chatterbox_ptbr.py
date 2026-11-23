"""Assemble the Chatterbox Multilingual pt-BR checkpoint folder that chatterbox_tts.py loads.

Run with the chatterbox venv (chatterbox-tts 0.1.7, torch 2.6 cu124):
  python setup_chatterbox_ptbr.py [--out ~/.cache/chatterbox-ptbr]
The pt-BR finetune ships only T3 and a V3 S3Gen. ChatterboxMultilingualTTS.from_local expects the base layout,
and s3gen_v3 lacks two tokenizer buffers that the base s3gen.pt has, so they are copied over.
"""
import argparse, os
import torch
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.expanduser("~/.cache/chatterbox-ptbr"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

base = snapshot_download("ResembleAI/chatterbox", allow_patterns=["ve.pt", "conds.pt", "s3gen.pt"])
ptbr = snapshot_download("ResembleAI/Chatterbox-Multilingual-pt-br",
                         allow_patterns=["t3_pt_br.safetensors", "s3gen_v3.pt", "grapheme_mtl_merged_expanded_v1.json"])

links = {"ve.pt": f"{base}/ve.pt", "conds.pt": f"{base}/conds.pt",
         "t3_mtl23ls_v2.safetensors": f"{ptbr}/t3_pt_br.safetensors",
         "grapheme_mtl_merged_expanded_v1.json": f"{ptbr}/grapheme_mtl_merged_expanded_v1.json"}
for name, src in links.items():
    dst = f"{args.out}/{name}"
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.realpath(src), dst)

v3 = torch.load(f"{ptbr}/s3gen_v3.pt", map_location="cpu", weights_only=True)
b = torch.load(f"{base}/s3gen.pt", map_location="cpu", weights_only=True)
for k in ("tokenizer._mel_filters", "tokenizer.window"):
    v3[k] = b[k]
torch.save(v3, f"{args.out}/s3gen.pt")
print("ready:", args.out)

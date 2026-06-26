#!/usr/bin/env python
"""LoRA training-health check: is the adapter FIRING (learning) and STORING (saved) well?

Loads every checkpoint-*/pytorch_lora_weights.safetensors under a results dir and reports,
per checkpoint:
  #layers, layers with lora_B==0 (dead),  mean/max/total ||ΔW||_F   (ΔW = B·A)
Optionally, with --base <Wan transformer dir>, also the RELATIVE update ||ΔW||/||W_base||
(the metric with an interpretable healthy band ~1–20%).

CPU-only by design (won't touch a GPU running other work):
  CUDA_VISIBLE_DEVICES="" python scripts/lora_health_check.py EgoX/results/EgoX_cooking_r128
  CUDA_VISIBLE_DEVICES="" python scripts/lora_health_check.py <results_dir> --base <transformer_dir>
"""
import argparse, glob, os, re, torch
from safetensors.torch import safe_open


def load_lora(path):
    A, B = {}, {}
    with safe_open(path, framework="pt", device="cpu") as g:
        for k in g.keys():
            if k.endswith("lora_A.weight"):
                A[k[:-len("lora_A.weight")]] = g.get_tensor(k).float()
            elif k.endswith("lora_B.weight"):
                B[k[:-len("lora_B.weight")]] = g.get_tensor(k).float()
    return A, B


def base_norms(base_dir):
    """||W||_F for each base linear, keyed to match a LoRA prefix (…attn1.to_k.)."""
    norms = {}
    for f in glob.glob(os.path.join(base_dir, "*.safetensors")):
        with safe_open(f, framework="pt", device="cpu") as g:
            for k in g.keys():
                if k.endswith(".weight") and "lora" not in k:
                    norms[k[:-len("weight")]] = g.get_tensor(k).float().norm().item()
    return norms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--base", default=None, help="Wan transformer dir for relative ||ΔW||/||W|| ratio")
    a = ap.parse_args()

    ckpts = sorted(glob.glob(os.path.join(a.results_dir, "checkpoint-*")),
                   key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)))
    bn = base_norms(a.base) if a.base else None
    hdr = f"{'ckpt':>6} {'#lyr':>5} {'B==0':>5} {'mean||ΔW||':>11} {'max||ΔW||':>10} {'total':>8}"
    if bn:
        hdr += f" {'rel%(med)':>9}"
    print(hdr); print("-" * len(hdr))
    for c in ckpts:
        step = int(re.search(r"checkpoint-(\d+)", c).group(1))
        A, B = load_lora(os.path.join(c, "pytorch_lora_weights.safetensors"))
        zero = sum(1 for k in B if B[k].abs().max() == 0)
        dW, rel = [], []
        for k in A:
            if k in B:
                d = (B[k] @ A[k]).norm().item(); dW.append(d)
                if bn and k in bn and bn[k] > 0:
                    rel.append(d / bn[k])
        dW = torch.tensor(dW)
        line = f"{step:6d} {len(B):5d} {zero:5d} {dW.mean():11.3f} {dW.max():10.3f} {dW.sum():8.1f}"
        if bn:
            r = torch.tensor(rel)
            line += f" {100*r.median():8.2f}%" if len(r) else f" {'n/a':>9}"
        print(line)


if __name__ == "__main__":
    main()

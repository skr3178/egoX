#!/usr/bin/env python
"""Build a CLEAN train/val split from Stage-1 outputs:
  1. merge per-clip meta_entry_*.json (complete Stage-1 only),
  2. EXCLUDE black/dark/missing priors (generated/bad_prior_clips.txt),
  3. stratified-by-TAKE split (hold out whole takes for val -> no scene leakage),
  4. write meta_train_clean.json + meta_val_clean.json.

Usage: build_clean_split.py [--val-clips 20] [--seed 42]
"""
import argparse, json, glob, os, re, random

CT = "/media/skr/SeagateHub1/egoexo4d/cooking_train"
BAD = "/media/skr/storage/paper_reproduction/egoX/generated/bad_prior_clips.txt"
NF = 49

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-clips", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    bad = set()
    if os.path.exists(BAD):
        bad = {l.strip() for l in open(BAD) if l.strip() and not l.startswith("#")}

    merged = {}
    for f in sorted(glob.glob(f"{CT}/meta_entry_*.json")):
        for clip, m in json.load(open(f)).items():
            dm = f"{CT}/depth_maps/{clip}"
            complete = (os.path.isdir(dm) and len([x for x in os.listdir(dm) if x.endswith('.npy')]) == NF
                        and os.path.exists(m.get('ego_prior_path', ''))
                        and os.path.exists(f"{CT}/vipe_results/{clip}/intrinsics/{clip}.npz"))
            if complete and clip not in bad:
                merged[clip] = m
    print(f"clean clips: {len(merged)} (excluded {len(bad)} bad-prior)")

    # group by take, stratified val = whole takes until ~val-clips reached
    bytake = {}
    for c in merged:
        bytake.setdefault(re.sub(r"_\d+_\d+$", "", c), []).append(c)
    takes = sorted(bytake)
    random.Random(a.seed).shuffle(takes)
    val_clips, val_takes = [], []
    for t in takes:
        if len(val_clips) >= a.val_clips:
            break
        val_clips += bytake[t]; val_takes.append(t)
    val_set = set(val_clips)
    train = {c: m for c, m in merged.items() if c not in val_set}
    val = {c: m for c, m in merged.items() if c in val_set}

    json.dump(train, open(f"{CT}/meta_train_clean.json", "w"), indent=2)
    json.dump(val, open(f"{CT}/meta_val_clean.json", "w"), indent=2)
    n_train_takes = len({re.sub(r"_\d+_\d+$", "", c) for c in train})
    print(f"TRAIN: {len(train)} clips ({n_train_takes} takes) -> meta_train_clean.json")
    print(f"VAL  : {len(val)} clips from {len(val_takes)} held-out takes -> meta_val_clean.json")
    print("  val takes: " + ", ".join(val_takes))

if __name__ == "__main__":
    main()

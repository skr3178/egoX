#!/usr/bin/env python
"""Scan every ego_Prior.mp4 in cooking_train and classify by coverage (non-black pixel fraction,
averaged over frames). Reproduces generated/empty_priors_list.txt and EXTENDS to newly-generated
clips. Black/darkened/missing priors carry ~no conditioning signal and must be excluded from train+val.

  coverage <1%   -> BLACK    (actively harmful)
  1-10%          -> DARKENED  (nearly blank)
  >=10%          -> OK
  no file        -> MISSING

Writes generated/bad_prior_clips.txt (one clip id per line; black+dark+missing) for meta assembly to
exclude. Usage: python scripts/pipeline/scan_priors.py [--ct <cooking_train>] [--ok-thresh 0.10]
"""
import argparse, os, glob
import cv2, numpy as np

def coverage(mp4):
    cap = cv2.VideoCapture(mp4); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(float((f.max(axis=2) > 12).mean()))   # non-near-black fraction
    cap.release()
    return float(np.mean(fr)) if fr else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct", default="/media/skr/SeagateHub1/egoexo4d/cooking_train")
    ap.add_argument("--black", type=float, default=0.01)
    ap.add_argument("--ok-thresh", type=float, default=0.10)
    ap.add_argument("--out", default="/media/skr/storage/paper_reproduction/egoX/generated/bad_prior_clips.txt")
    a = ap.parse_args()

    clips = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{a.ct}/depth_maps/*"))
    clips = sorted(os.path.basename(d) for d in glob.glob(f"{a.ct}/depth_maps/*") if os.path.isdir(d))
    black, dark, ok, missing = [], [], [], []
    for c in clips:
        p = f"{a.ct}/videos/{c}/ego_Prior.mp4"
        if not os.path.exists(p):
            missing.append(c); continue
        cov = coverage(p)
        (black if cov < a.black else dark if cov < a.ok_thresh else ok).append((c, cov))

    bad = sorted([c for c, _ in black] + [c for c, _ in dark] + missing)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(bad) + ("\n" if bad else ""))
    print(f"scanned {len(clips)} clips | OK={len(ok)} BLACK={len(black)} DARK={len(dark)} MISSING={len(missing)}")
    print(f"  -> {len(bad)} bad (black+dark+missing) written to {a.out}")
    # bad-by-take summary (whole-take failures are the actionable signal)
    import re
    from collections import defaultdict
    tot, badc = defaultdict(int), defaultdict(int)
    for c in clips:
        t = re.sub(r"_\d+_\d+$", "", c); tot[t] += 1
    for c in bad:
        t = re.sub(r"_\d+_\d+$", "", c); badc[t] += 1
    full = [t for t in badc if badc[t] == tot[t]]
    if full:
        print(f"  whole-take failures ({len(full)}): " + ", ".join(sorted(full)))

if __name__ == "__main__":
    main()

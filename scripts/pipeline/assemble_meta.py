#!/usr/bin/env python
"""Merge per-clip meta_entry_*.json -> one training-schema meta.json (flat dict keyed by clip).
Only includes clips with COMPLETE Stage-1 outputs (49 depth npy + ego_Prior + intrinsics npz).

Usage: assemble_meta.py [out_meta.json]   (default: cooking_train/meta_train_cooking.json)
"""
import json, glob, os, sys
OUT = "/media/skr/SeagateHub1/egoexo4d/cooking_train"
NF = 49
out_path = sys.argv[1] if len(sys.argv) > 1 else f"{OUT}/meta_train_cooking.json"

merged, skipped = {}, 0
for f in sorted(glob.glob(f"{OUT}/meta_entry_*.json")):
    for clip, m in json.load(open(f)).items():
        dm = f"{OUT}/depth_maps/{clip}"
        ok = (os.path.isdir(dm) and len([x for x in os.listdir(dm) if x.endswith('.npy')]) == NF
              and os.path.exists(m.get('ego_prior_path', ''))
              and os.path.exists(f"{OUT}/vipe_results/{clip}/intrinsics/{clip}.npz"))
        if ok:
            merged[clip] = m
        else:
            skipped += 1

json.dump(merged, open(out_path, "w"), indent=2)
print(f"assembled {len(merged)} complete clips ({skipped} incomplete skipped) -> {out_path}")

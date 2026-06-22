#!/usr/bin/env python
"""Integrity check for a downloaded HF snapshot.

Verifies, for Wan-AI/Wan2.1-I2V-14B-480P-Diffusers:
  1. no leftover *.incomplete files in the cache
  2. every LFS file's on-disk SHA256 == the authoritative sha256 from the Hub
     (and byte size matches)
  3. every *.safetensors file actually parses (catches silent truncation)

Exit code 0 = all good, 1 = problems found.
"""
import hashlib
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

REPO = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"


def sha256_file(path, chunk=16 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    api = HfApi()
    info = api.model_info(REPO, files_metadata=True)

    # locate the local snapshot dir without triggering any download
    snap = snapshot_download(REPO, local_files_only=True)
    snap = Path(snap)
    cache_root = Path(HF_HUB_CACHE) / ("models--" + REPO.replace("/", "--"))
    print(f"snapshot: {snap}")

    # 1) leftover .incomplete
    incompletes = list(cache_root.rglob("*.incomplete"))
    print(f"\n[1] leftover .incomplete files: {len(incompletes)}")
    for p in incompletes:
        print("    !!", p)

    # 2) + 3) per-file checks
    lfs_files, ok, bad, missing = [], [], [], []
    for s in info.siblings:
        if s.lfs is not None:
            lfs_files.append(s)

    print(f"\n[2] verifying SHA256 of {len(lfs_files)} LFS files against the Hub...")
    for s in sorted(lfs_files, key=lambda x: x.rfilename):
        fpath = snap / s.rfilename
        name = s.rfilename
        if not fpath.exists():
            print(f"    MISSING   {name}")
            missing.append(name)
            continue
        exp_sha = s.lfs.sha256
        exp_size = s.size
        act_size = fpath.stat().st_size
        if exp_size is not None and act_size != exp_size:
            print(f"    SIZE-BAD  {name}  (have {act_size} want {exp_size})")
            bad.append(name)
            continue
        act_sha = sha256_file(fpath)
        if act_sha == exp_sha:
            print(f"    ok        {name}  ({act_size/1e9:.2f} GB)")
            ok.append(name)
        else:
            print(f"    SHA-BAD   {name}\n              have {act_sha}\n              want {exp_sha}")
            bad.append(name)

    # 3) safetensors parse test
    print("\n[3] safetensors parse test...")
    st_bad = []
    try:
        from safetensors import safe_open
        for s in sorted(lfs_files, key=lambda x: x.rfilename):
            if not s.rfilename.endswith(".safetensors"):
                continue
            fpath = snap / s.rfilename
            if not fpath.exists():
                continue
            try:
                with safe_open(fpath, framework="pt") as f:
                    _ = list(f.keys())
            except Exception as e:
                print(f"    PARSE-BAD {s.rfilename}: {type(e).__name__}: {e}")
                st_bad.append(s.rfilename)
        print(f"    parsed {len([s for s in lfs_files if s.rfilename.endswith('.safetensors')]) - len(st_bad)} files cleanly")
    except ImportError:
        print("    (safetensors not importable here — skipped)")

    # summary
    print("\n" + "=" * 60)
    problems = len(incompletes) + len(bad) + len(missing) + len(st_bad)
    print(f"SUMMARY: {len(ok)} ok | {len(bad)} hash/size-bad | "
          f"{len(missing)} missing | {len(st_bad)} parse-bad | "
          f"{len(incompletes)} incomplete")
    if problems == 0:
        print("RESULT: ✅ ALL FILES VERIFIED — snapshot is intact.")
        return 0
    print("RESULT: ❌ PROBLEMS FOUND — re-run `hf download` to repair "
          "(it will re-fetch only the bad/missing files).")
    return 1


if __name__ == "__main__":
    sys.exit(main())

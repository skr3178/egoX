#!/usr/bin/env python3
"""Selective Ego-Exo4D take_trajectory downloader.

Pulls ONLY the small calibration/metadata files from each take's trajectory/
folder, skipping the large open_loop/closed_loop pose CSVs (~46 GB for the
cooking subset). Uses the same v2 manifest the official `egoexo` CLI uses, so
S3 keys/sizes are authoritative.

Usage:
    python scripts/fetch_trajectory_calib_only.py \
        --uids cooking_uids.txt --out egoexo_dl --profile egoexo

KEEP set: online_calibration.jsonl (ego intrinsics),
          gopro_calibs.csv (exo cam01-04 intrinsics+extrinsics),
          summary.json (timing meta).
"""
import argparse, os
from iopath.common.s3 import S3PathHandler
from iopath.common.file_io import PathManager
from ego4d.internal.download.manifest import manifest_loads

MANIFEST = "s3://ego4d-consortium-sharing/egoexo-public/v2/take_trajectory/manifest.json"
KEEP = {"online_calibration.jsonl", "gopro_calibs.csv", "summary.json"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uids", required=True, help="file with one take_uid per line")
    ap.add_argument("--out", default="egoexo_dl", help="output dir (mirrors takes/<take>/...)")
    ap.add_argument("--profile", default="egoexo", help="AWS profile in ~/.aws/credentials")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pm = PathManager()
    pm.register_handler(S3PathHandler(profile=args.profile))

    want = {l.strip() for l in open(args.uids) if l.strip()}
    ms = manifest_loads(pm.open(MANIFEST).read())
    entries = [m for m in ms if m.uid in want]
    print(f"matched {len(entries)}/{len(want)} uids in manifest")

    todo = []
    for m in entries:
        for p in m.paths:
            if p.relative_path.split("/")[-1] in KEEP:
                todo.append(p)
    total = sum(p.size or 0 for p in todo)
    print(f"{len(todo)} files, {total/1e9:.2f} GB to fetch")
    if args.dry_run:
        return

    done = 0
    for i, p in enumerate(todo, 1):
        dst = os.path.join(args.out, p.relative_path)
        if os.path.exists(dst) and os.path.getsize(dst) == (p.size or -1):
            done += 1
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with pm.open(p.source_path, "rb") as src, open(dst, "wb") as f:
            f.write(src.read())
        done += 1
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}  {dst}")
    print(f"done: {done} files -> {args.out}")


if __name__ == "__main__":
    main()

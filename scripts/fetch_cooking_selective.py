#!/usr/bin/env python3
"""Selectively download the cooking-subset Ego-Exo4D files EgoX uses (best_exo + ego),
at 448p, resumable, to the Seagate drive.

Run with the .venv-egoexo interpreter (has boto3 + ego4d):
  /media/skr/storage/paper_reproduction/hands/.venv-egoexo/bin/python \
      scripts/fetch_cooking_selective.py [--dry-run]

Why selective: the official egoexo CLI can only filter by --views {ego,exo} (all exo cams),
not by camera id. The S3 manifest lists every camera as its own object, so we fetch exactly
the two files EgoX consumes per take: the ego RGB (aria*_214-1.mp4) + the chosen best_exo cam.

Note: each take's media lives in a PER-UNIVERSITY bucket (e.g. s3://ego4d-minnesota/...),
each in its own region; source_path in the manifest is the authoritative s3:// location.

Auth: boto3 profile "egoexo" (~/.aws). Resumable: skip files whose local size matches the
manifest size (same check the egoexo CLI uses).
"""
import argparse
import csv
import json
import os
import re
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# Ego RGB stream = Aria camera 214 (the others: 1201-* = SLAM mono, 211-* = eye tracking).
EGO_RGB_RE = re.compile(r"^aria\d+_214-1\.mp4$")
EXO_CAM_RE = re.compile(r"^cam\d+\.mp4$")

REPO = Path(__file__).resolve().parent.parent

MANIFEST_BUCKET = "ego4d-consortium-sharing"
MANIFEST_KEY = "egoexo-public/v2/downscaled_takes/448/manifest.json"

OUT = Path("/media/skr/SeagateHub1/egoexo4d")
PROFILE = "egoexo"
_CFG = Config(retries={"max_attempts": 10, "mode": "standard"})


class S3:
    """Per-bucket client cache; each bucket gets a client in its own region."""

    def __init__(self, profile):
        self.session = boto3.session.Session(profile_name=profile)
        self._boot = self.session.client("s3", config=_CFG)  # for get_bucket_location
        self._clients = {}

    def client(self, bucket):
        if bucket not in self._clients:
            self._clients[bucket] = self.session.client(
                "s3", region_name=self._region(bucket), config=_CFG
            )
        return self._clients[bucket]

    def _region(self, bucket):
        """Region from the x-amz-bucket-region header (present even on a denied/redirect
        response) — avoids GetBucketLocation, which these credentials don't allow."""
        try:
            hdr = self._boot.head_bucket(Bucket=bucket)["ResponseMetadata"]["HTTPHeaders"]
        except ClientError as e:
            hdr = e.response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
        return hdr.get("x-amz-bucket-region") or "us-east-1"

    def get_json(self, bucket, key):
        return json.loads(self.client(bucket).get_object(Bucket=bucket, Key=key)["Body"].read())

    def download(self, source_path, dest):
        bucket, key = source_path.split("s3://", 1)[1].split("/", 1)
        self.client(bucket).download_file(bucket, key, str(dest))


def load_targets(csv_path):
    """uid -> best_exo ('ALL' for takes with no catalog best_exo)."""
    with Path(csv_path).open() as f:
        return {r["take_uid"]: r["best_exo"] for r in csv.DictReader(f)}


def select_paths(entry, best_exo):
    """Keep the ego RGB + the one best_exo cam (or all exo cams when best_exo == ALL).

    The 448p manifest leaves `views` unset, so select by basename:
    ego = aria*_214-1.mp4 (RGB stream), exo = the chosen camNN.mp4.
    """
    keep = []
    for p in entry["paths"]:
        base = os.path.basename(p["relative_path"])
        is_ego = bool(EGO_RGB_RE.match(base))
        if best_exo == "ALL":
            if is_ego or EXO_CAM_RE.match(base):
                keep.append(p)
        elif is_ego or base == f"{best_exo}.mp4":
            keep.append(p)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(REPO / "egoexo4D" / "cooking_takes.csv"),
                    help="subset CSV with take_name,take_uid,best_exo (default: cooking)")
    ap.add_argument("--dry-run", action="store_true", help="list selected files, download nothing")
    args = ap.parse_args()

    targets = load_targets(args.csv)
    print(f"{Path(args.csv).stem}: {len(targets)} uids", flush=True)

    s3 = S3(PROFILE)
    print(f"fetching manifest s3://{MANIFEST_BUCKET}/{MANIFEST_KEY}", flush=True)
    manifest = s3.get_json(MANIFEST_BUCKET, MANIFEST_KEY)
    by_uid = {m["uid"]: m for m in manifest}

    selected = []      # (source_path, dest, size)
    missing_uids = []
    for uid, best_exo in targets.items():
        entry = by_uid.get(uid)
        if entry is None:
            missing_uids.append(uid)
            continue
        for p in select_paths(entry, best_exo):
            selected.append((p["source_path"], OUT / p["relative_path"], p.get("size")))

    print(f"selected files: {len(selected)}  (expect ~532 = 266 x 2)", flush=True)
    if missing_uids:
        print(f"WARNING: {len(missing_uids)} uids absent from manifest: {missing_uids[:5]}", flush=True)

    if args.dry_run:
        for src, dest, size in selected[:6]:
            print(f"  {src}  ->  {dest}  ({size} bytes)")
        print("  ... (dry-run, nothing downloaded)")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    done = skipped = failed = 0
    total_bytes = 0
    for i, (src, dest, size) in enumerate(selected, 1):
        try:
            if dest.exists() and size is not None and dest.stat().st_size == size:
                skipped += 1
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix + ".part")
                s3.download(src, tmp)
                tmp.replace(dest)
                done += 1
                total_bytes += dest.stat().st_size
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL [{i}/{len(selected)}] {src}: {e}", flush=True)
        if i % 25 == 0 or i == len(selected):
            print(f"progress {i}/{len(selected)}  downloaded={done} skipped={skipped} "
                  f"failed={failed}  +{total_bytes/1e9:.2f}GB", flush=True)

    print(f"DONE downloaded={done} skipped={skipped} failed={failed} "
          f"bytes={total_bytes/1e9:.2f}GB", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

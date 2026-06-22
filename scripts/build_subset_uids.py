#!/usr/bin/env python3
"""Build the EgoX take list for a given Ego-Exo4D domain (canonical parent_task_name),
mapping each take to its take_uid + best_exo camera.

Usage (run with any python that can read JSON):
  python scripts/build_subset_uids.py "Basketball"
  python scripts/build_subset_uids.py "Bike Repair" --slug bike

Writes (under gitignored egoexo4D/):
  egoexo4D/<slug>_uids.txt    one take_uid per line
  egoexo4D/<slug>_takes.csv   take_name,take_uid,best_exo

Domain membership uses takes.json `parent_task_name` (canonical), restricted to the takes
EgoX actually references in meta_*.json. best_exo missing in the catalog -> "ALL".
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
META = REPO / "egoexo4D"
TAKE_RE = re.compile(r"/([^/]+)/exo\.mp4$")


def base_take_name(t: str) -> str:
    parts = t.split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return "_".join(parts[:-2])
    return t


def egox_takes():
    """Distinct base take_names EgoX references, with clip counts."""
    takes, clips = set(), 0
    for fname, key in [("meta_train.json", "train_datasets"),
                       ("meta_seen.json", "val_datasets"),
                       ("meta_unseen.json", "test_datasets")]:
        for e in json.loads((META / fname).read_text())[key]:
            takes.add(base_take_name(TAKE_RE.search(e["exo_path"]).group(1)))
            clips += 1
    return takes, clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", help='canonical parent_task_name, e.g. "Basketball", "Dance"')
    ap.add_argument("--slug", help="output filename stem (default: domain lowercased)")
    args = ap.parse_args()
    slug = args.slug or args.domain.lower().replace(" ", "_")

    takes, _ = egox_takes()
    catalog = {t["take_name"]: t for t in json.loads((META / "raw" / "takes.json").read_text())}

    rows, none_best, missing = [], [], []
    for take in sorted(takes):
        info = catalog.get(take)
        if info is None:
            missing.append(take)
            continue
        if (info.get("parent_task_name") or "") != args.domain:
            continue
        best = info.get("best_exo") or "ALL"
        if best == "ALL":
            none_best.append(take)
        rows.append((take, info["take_uid"], best))

    if missing:
        print(f"ERROR: {len(missing)} EgoX takes absent from takes.json: {missing[:5]}", file=sys.stderr)
        return 1
    if not rows:
        names = sorted({c.get("parent_task_name") for c in catalog.values() if c.get("parent_task_name")})
        print(f"ERROR: no takes for domain {args.domain!r}. Known domains: {names}", file=sys.stderr)
        return 1

    (META / f"{slug}_uids.txt").write_text("\n".join(r[1] for r in rows) + "\n")
    with (META / f"{slug}_takes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["take_name", "take_uid", "best_exo"])
        w.writerows(rows)

    print(f"domain={args.domain!r} slug={slug}: {len(rows)} takes "
          f"(best_exo=ALL: {len(none_best)})")
    print(f"wrote egoexo4D/{slug}_uids.txt and egoexo4D/{slug}_takes.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

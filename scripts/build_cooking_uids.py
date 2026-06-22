#!/usr/bin/env python3
"""Derive the cooking-domain take list EgoX needs and map each to its Ego-Exo4D
take_uid + best_exo camera.

Reads (relative to repo root):
  egoexo4D/meta_train.json | meta_seen.json | meta_unseen.json   (the EgoX recipe)
  egoexo4D/raw/takes.json                                        (official catalog)

Writes (under gitignored egoexo4D/):
  egoexo4D/cooking_uids.txt   one take_uid per line
  egoexo4D/cooking_takes.csv  take_name,take_uid,best_exo

Take-name rule: exo_path = ./videos/<base>_<startframe>_<endframe>/exo.mp4
  -> strip the trailing two numeric underscore fields to get the base take_name.
Keep takes whose base name contains "cooking".
The 5 takes with no best_exo in the catalog are recorded as best_exo=ALL
(the fetch step keeps every exo cam for those rather than guessing).
"""
import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
META = REPO / "egoexo4D"
TAKE_RE = re.compile(r"/([^/]+)/exo\.mp4$")


def base_take_name(take_with_frames: str) -> str:
    parts = take_with_frames.split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return "_".join(parts[:-2])
    return take_with_frames


def main() -> int:
    splits = [
        ("meta_train.json", "train_datasets"),
        ("meta_seen.json", "val_datasets"),
        ("meta_unseen.json", "test_datasets"),
    ]

    cooking = set()
    clip_count = 0
    for fname, key in splits:
        path = META / fname
        data = json.loads(path.read_text())
        for entry in data[key]:
            m = TAKE_RE.search(entry["exo_path"])
            if not m:
                continue
            base = base_take_name(m.group(1))
            if "cooking" in base.lower():
                cooking.add(base)
                clip_count += 1

    catalog = {t["take_name"]: t for t in json.loads((META / "raw" / "takes.json").read_text())}

    missing = sorted(t for t in cooking if t not in catalog)
    if missing:
        print(f"ERROR: {len(missing)} cooking takes not found in takes.json: {missing[:5]}", file=sys.stderr)
        return 1

    rows = []
    none_best = []
    for take in sorted(cooking):
        info = catalog[take]
        best = info.get("best_exo")
        if not best:
            best = "ALL"
            none_best.append(take)
        rows.append((take, info["take_uid"], best))

    uids_path = META / "cooking_uids.txt"
    csv_path = META / "cooking_takes.csv"
    uids_path.write_text("\n".join(r[1] for r in rows) + "\n")
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["take_name", "take_uid", "best_exo"])
        w.writerows(rows)

    print(f"cooking takes: {len(rows)}  (clips: {clip_count})")
    print(f"best_exo=ALL (no catalog best_exo): {len(none_best)} -> {none_best}")
    print(f"wrote {uids_path.relative_to(REPO)} ({len(rows)} uids)")
    print(f"wrote {csv_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

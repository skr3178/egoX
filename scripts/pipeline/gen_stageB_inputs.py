#!/usr/bin/env python
"""Generate the missing Stage B (training) input files for the 5 shipped example clips.

The EgoX example/ dir ships exo/ego_GT/ego_Prior videos + 49 depth .npy/clip + camera
params, but is MISSING two things the training dataloader (wan_dataset.py) needs:

  1. ViPE depth-intrinsics  ->  <vipe_results_path>/intrinsics/<best_camera>.npz
     Format (confirmed via iproj_disp in core/finetune/datasets/utils.py:99):
       npz key 'data', shape (N,4) = [fx, fy, cx, cy]; loader reads data[0:1,:].
     These intrinsics are camera-calibration (GeoCalib/ViPE) and are DEPTH-MODEL
     INDEPENDENT, so for the well-calibrated Ego-Exo4D clips we derive them by scaling
     the meta camera_intrinsics (at ~4K) down to the depth-map resolution.

  2. A training-schema meta.json. The shipped meta is INFERENCE format
     ({"test_datasets":[...]}, fields exo_path/ego_gt_path/ego_prior_path). The loader
     iterates meta_data.items() and reads exo_video_path/ego_video_path/take_name/
     vipe_results_path/best_camera/... -> we convert.

Outputs (under example/egoexo4D, so data_root = that dir):
  - vipe_results/<take>/intrinsics/<take>.npz      (one per clip)
  - meta_train_smoke.json                          (training-schema, 5 clips)
"""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path("/media/skr/storage/paper_reproduction/egoX/EgoX/example/egoexo4D")
SRC_META = ROOT / "meta.json"
OUT_META = ROOT / "meta_train_smoke.json"


def take_name_from_path(p: str) -> str:
    # ".../videos/<take>/ego_Prior.mp4" -> "<take>"
    return Path(p).parent.name


def main():
    src = json.load(open(SRC_META))["test_datasets"]
    out = {}
    print(f"source clips: {len(src)}")

    for e in src:
        take = take_name_from_path(e["ego_prior_path"])
        vid_dir = ROOT / "videos" / take
        depth_dir = ROOT / "depth_maps" / take

        # --- depth resolution (from an actual depth map) ---
        d0 = sorted(depth_dir.glob("*.npy"))[0]
        Hd, Wd = np.load(d0).shape  # (H, W)

        # --- scale meta camera_intrinsics (~4K) to depth resolution ---
        K = np.array(e["camera_intrinsics"], dtype=np.float64)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        W0, H0 = 2.0 * cx, 2.0 * cy           # implied source resolution from principal pt
        sx, sy = Wd / W0, Hd / H0
        fx_d, fy_d = fx * sx, fy * sy
        cx_d, cy_d = cx * sx, cy * sy          # == Wd/2, Hd/2

        # --- write npz: {'data': (1,4)=[fx,fy,cx,cy]} at depth resolution ---
        vipe_dir = ROOT / "vipe_results" / take / "intrinsics"
        vipe_dir.mkdir(parents=True, exist_ok=True)
        npz_path = vipe_dir / f"{take}.npz"
        data = np.array([[fx_d, fy_d, cx_d, cy_d]], dtype=np.float64)  # (1,4)
        np.savez(npz_path, data=data)

        # --- training-schema meta entry ---
        out[take] = {
            "exo_video_path": str(vid_dir / "exo.mp4"),
            "ego_video_path": str(vid_dir / "ego_GT.mp4"),
            "ego_prior_path": str(vid_dir / "ego_Prior.mp4"),
            "prompt": e["prompt"],
            "take_name": take,
            "vipe_results_path": str(ROOT / "vipe_results" / take),
            "best_camera": take,
            "camera_extrinsics": e["camera_extrinsics"],
            "camera_intrinsics": e["camera_intrinsics"],
            "ego_extrinsics": e["ego_extrinsics"],
            "ego_intrinsics": e["ego_intrinsics"],
        }
        print(f"  {take:36s} depth {Hd}x{Wd}  K_depth=[fx {fx_d:.1f}, fy {fy_d:.1f}, "
              f"cx {cx_d:.1f}, cy {cy_d:.1f}]  -> {npz_path.name}")

    json.dump(out, open(OUT_META, "w"), indent=2)
    print(f"\nwrote training meta: {OUT_META}  ({len(out)} clips)")
    print(f"data_root for the dataloader: {ROOT}")


if __name__ == "__main__":
    main()

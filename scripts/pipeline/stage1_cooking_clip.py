#!/usr/bin/env python
"""Stage-1 for ONE cooking clip — reuses the existing validated scripts (no rewrites of the
core ViPE/convert/render logic). Pulls poses/intrinsics/prompt/frame-range from meta_train.json.

Per clip (e.g. sfu_cooking_011_5_2451_2499 -> take sfu_cooking_011_5, frames 2451..2499):
  1. ffmpeg trim  cam04[start:end] -> exo.mp4   (49 frames, native 796x448)
     ffmpeg trim  aria01_214-1[start:end] -> ego_GT.mp4 (448x448)
  2. vipe infer exo.mp4 -p lyra (vit-L, GeoCalib intrinsics)  -> depth.zip + intrinsics.npz
  3. convert_depth_zip_to_npy.py -> depth_maps/<clip>/*.npy
  4. render_vipe_pointcloud.py (--fish_eye_rendering --use_mean_bg --only_bg) + meta poses -> ego_Prior.mp4
  5. place ViPE intrinsics .npz + return the training-schema meta entry

Resolution reduction to 49x256x704 is NOT done here — it happens at train time (loader/precompute).

Usage:  egox-egoprior/bin/python stage1_cooking_clip.py <clip_name> [--out <dir>]
Env:    run with the egox-egoprior python; needs PYTHONNOUSERSITE=1, HF_HUB_ENABLE_HF_TRANSFER=0.
"""
import argparse, json, os, re, subprocess, sys, shutil
from pathlib import Path

ENV = "/media/skr/storage/conda_envs/egox-egoprior"
RENDERER = "/media/skr/storage/paper_reproduction/egoX/EgoX/EgoX-EgoPriorRenderer"
SEAGATE = "/media/skr/SeagateHub1/egoexo4d/takes"
META_TRAIN = "/media/skr/storage/paper_reproduction/egoX/egoexo4D/meta_train.json"
TAKES_JSON = "/media/skr/storage/paper_reproduction/hands/trihands/egoexo_data/takes.json"
DEF_OUT = "/media/skr/SeagateHub1/egoexo4d/cooking_train"

_BEST_EXO = None
def best_exo(take):
    """best_exo cam name (e.g. 'cam04') from takes.json; None if unknown."""
    global _BEST_EXO
    if _BEST_EXO is None:
        _BEST_EXO = {t["take_name"]: t.get("best_exo") for t in json.load(open(TAKES_JSON))}
    return _BEST_EXO.get(take)


def sh(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def find_meta_entry(clip):
    for e in json.load(open(META_TRAIN))["train_datasets"]:
        if e["exo_path"].split("/")[-2] == clip:
            return e
    raise KeyError(f"{clip} not in meta_train")


def trim(src, start, end, dst):
    """frame-accurate trim of frames [start, end] inclusive -> dst (49 frames)."""
    sh(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
        "-vf", f"select=between(n\\,{start}\\,{end}),setpts=N/FRAME_RATE/TB",
        "-vsync", "0", "-an", dst])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--pipeline", default="lyra",
                    help="vipe config: lyra (vit-L, faithful) or lyra_svda (vit-S, ~11GB)")
    ap.add_argument("--skip-render", action="store_true",
                    help="skip ego_prior render (needs online_calibration.jsonl); do trim+vipe+depth only")
    a = ap.parse_args()
    clip = a.clip
    take = re.sub(r"_\d+_\d+$", "", clip)
    start, end = map(int, clip.rsplit("_", 2)[-2:])
    nframes = end - start + 1
    print(f"=== clip {clip} | take {take} | frames {start}..{end} ({nframes}) ===")
    assert nframes == 49, f"expected 49 frames, got {nframes}"

    e = find_meta_entry(clip)
    src = f"{SEAGATE}/{take}/frame_aligned_videos/downscaled/448"
    cams = sorted(f for f in os.listdir(src) if re.match(r"cam\d+\.mp4$", f))
    assert cams, f"no exo cam in {src}"
    best = best_exo(take)                       # e.g. 'cam04' (varies per take)
    camfile = f"{best}.mp4" if best and f"{best}.mp4" in cams else cams[0]
    exo_src, aria = f"{src}/{camfile}", f"{src}/aria01_214-1.mp4"
    print(f"  exo cam: {camfile} (best_exo={best})")
    assert os.path.exists(aria), f"missing {aria}"

    out = Path(a.out)
    vid = out / "videos" / clip; vid.mkdir(parents=True, exist_ok=True)
    depth_dir = out / "depth_maps" / clip
    vipe_out = out / "vipe_results"            # vipe writes vipe_out/<clip>/...
    intr_dir = out / "vipe_results" / clip / "intrinsics"; intr_dir.mkdir(parents=True, exist_ok=True)

    # resumable: skip if trim+vipe+depth already done for this clip
    done = (len(list(depth_dir.glob("*.npy"))) == nframes
            and any(intr_dir.glob("*.npz"))
            and (vid / "exo.mp4").exists() and (vid / "ego_GT.mp4").exists()
            and (a.skip_render or (vid / "ego_Prior.mp4").exists()))
    if done:
        print(f"=== SKIP {clip}: already has {nframes} depth + intrinsics + trims"
              f"{'' if a.skip_render else ' + ego_Prior'} ===")
        return

    # 1) trim exo + ego_GT
    print("[1/5] trim exo + ego_GT")
    trim(exo_src, start, end, str(vid / "exo.mp4"))
    trim(aria,  start, end, str(vid / "ego_GT.mp4"))

    # 2) vipe infer (vit-L lyra) with GT exo intrinsics from meta_train (faithful, GT-frame-consistent)
    print("[2/5] vipe infer (lyra / vit-L, GT intrinsics from meta_train)")
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0"}
    # scale meta_train camera_intrinsics (≈4K) to the actual cam04 video resolution
    import cv2
    cap = cv2.VideoCapture(str(vid / "exo.mp4")); W = int(cap.get(3)); H = int(cap.get(4)); cap.release()
    K = [[float(x) for x in row] for row in e["camera_intrinsics"]]
    sx, sy = W / (2.0 * K[0][2]), H / (2.0 * K[1][2])
    Kgt = [[K[0][0]*sx, 0.0, W/2.0], [0.0, K[1][1]*sy, H/2.0], [0.0, 0.0, 1.0]]
    # name the vipe input so its stem == clip (vipe uses the file stem as the result dir name)
    exo_named = vid / f"{clip}.mp4"; shutil.copy(vid / "exo.mp4", exo_named)
    sh([f"{ENV}/bin/vipe", "infer", str(exo_named), "-o", str(vipe_out),
        "-p", a.pipeline, "--assume_fixed_camera_pose", "--end_frame", str(nframes - 1),
        "--use_exo_intrinsic_gt", json.dumps(Kgt)], env=env)
    exo_named.unlink(missing_ok=True)
    vres = vipe_out / clip

    # 3) depth zip -> npy
    print("[3/5] convert depth -> npy")
    depth_dir.mkdir(parents=True, exist_ok=True)
    sh([f"{ENV}/bin/python", f"{RENDERER}/scripts/convert_depth_zip_to_npy.py",
        "--depth_path", str(vres / "depth"), "--egox_depthmaps_path", str(out / "depth_maps")],
       env=env, cwd=RENDERER)

    # 4) render ego_prior (fisheye) using meta_train poses — needs online_calibration.jsonl
    if a.skip_render:
        print("[4/5] render ego_prior — SKIPPED (--skip-render; do after calib lands)")
    else:
        print("[4/5] render ego_prior (fisheye)")
        render_meta = {"test_datasets": [{
            "exo_path": f"./videos/{clip}/exo.mp4",
            "ego_prior_path": f"./videos/{clip}/ego_Prior.mp4",
            "prompt": e["prompt"],
            "camera_intrinsics": e["camera_intrinsics"], "camera_extrinsics": e["camera_extrinsics"],
            "ego_intrinsics": e["ego_intrinsics"], "ego_extrinsics": e["ego_extrinsics"],
        }]}
        rmeta = out / f"render_meta_{clip}.json"; json.dump(render_meta, open(rmeta, "w"))
        render_out = out / "ego_prior_render"
        # faithful: pass the take's per-take Aria fisheye calib if downloaded (else generic fallback)
        calib = f"{SEAGATE}/{take}/trajectory/online_calibration.jsonl"
        rcmd = [f"{ENV}/bin/python", f"{RENDERER}/scripts/render_vipe_pointcloud.py",
                "--input_dir", str(vres), "--meta_json_path", str(rmeta), "--out_dir", str(render_out),
                "--start_frame", "0", "--end_frame", str(nframes - 1), "--point_size", "5.0",
                "--fish_eye_rendering", "--use_mean_bg", "--only_bg"]
        if os.path.exists(calib):
            rcmd += ["--online_calibration_path", calib]
            print(f"  using per-take Aria calib: {calib}")
        else:
            print("  no per-take calib found -> generic Aria coeffs")
        sh(rcmd, env=env, cwd=RENDERER)
        rendered = render_out / clip / "ego_Prior.mp4"
        if rendered.exists():
            shutil.copy(rendered, vid / "ego_Prior.mp4")

    # 5) place ViPE intrinsics .npz (already in vres/intrinsics from vipe) + build meta entry
    print("[5/5] intrinsics + meta entry")
    src_npz = next((vres / "intrinsics").glob("*.npz"), None)
    dst_npz = intr_dir / f"{clip}.npz"
    if src_npz and src_npz.resolve() != dst_npz.resolve():
        shutil.copy(src_npz, dst_npz)

    entry = {clip: {
        "exo_video_path": str(vid / "exo.mp4"),
        "ego_video_path": str(vid / "ego_GT.mp4"),
        "ego_prior_path": str(vid / "ego_Prior.mp4"),
        "prompt": e["prompt"],
        "take_name": clip,
        "vipe_results_path": str(out / "vipe_results" / clip),
        "best_camera": clip,
        "camera_extrinsics": e["camera_extrinsics"], "camera_intrinsics": e["camera_intrinsics"],
        "ego_extrinsics": e["ego_extrinsics"], "ego_intrinsics": e["ego_intrinsics"],
    }}
    json.dump(entry, open(out / f"meta_entry_{clip}.json", "w"), indent=2)
    print(f"=== DONE {clip}: exo+ego_GT+ego_Prior, {len(list(depth_dir.glob('*.npy')))} depth npy, intrinsics, meta entry ===")


if __name__ == "__main__":
    main()

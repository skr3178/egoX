#!/usr/bin/env python
"""Group-by-take Stage-1 batch — faithful vit-L, FASTER via fewer model reloads.

Per take: trim all its 49-frame clips, then run ViPE ONCE in DIRECTORY mode over them
(models load once per take instead of once per clip -> ~3.4x fewer reloads, GPU stays
saturated). GT intrinsics from meta_train (same cam across a take). Render deferred
(needs online_calibration.jsonl). Resumable; continues on per-take errors.

Run detached: PYTHONNOUSERSITE=1 egox-egoprior/bin/python stage1_batch_bytake.py
"""
import json, os, re, shutil, subprocess
from pathlib import Path
import cv2

ENV = "/media/skr/storage/conda_envs/egox-egoprior"
RENDERER = "/media/skr/storage/paper_reproduction/egoX/EgoX/EgoX-EgoPriorRenderer"
SEAGATE = "/media/skr/SeagateHub1/egoexo4d/takes"
META = "/media/skr/storage/paper_reproduction/egoX/egoexo4D/meta_train.json"
TAKES_JSON = "/media/skr/storage/paper_reproduction/hands/trihands/egoexo_data/takes.json"
OUT = Path("/media/skr/SeagateHub1/egoexo4d/cooking_train")
ENVRN = {**os.environ, "PYTHONNOUSERSITE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0"}


def sh(cmd, **kw):
    print("  $", " ".join(map(str, cmd))[:300], flush=True)
    return subprocess.run(cmd, env=ENVRN, **kw)


def trim(src, s, e, dst):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                    "-vf", f"select=between(n\\,{s}\\,{e}),setpts=N/FRAME_RATE/TB",
                    "-vsync", "0", "-an", dst], check=True)


def clip_done(c):
    d = OUT / "depth_maps" / c
    return (d.exists() and len(list(d.glob("*.npy"))) == 49
            and any((OUT / "vipe_results" / c / "intrinsics").glob("*.npz")))


best_exo = {t["take_name"]: t.get("best_exo") for t in json.load(open(TAKES_JSON))}
seg = set(os.listdir(SEAGATE))
by_take, entry = {}, {}
for e in json.load(open(META))["train_datasets"]:
    c = e["exo_path"].split("/")[-2]; t = re.sub(r"_\d+_\d+$", "", c)
    if "cooking" in c and t in seg:
        by_take.setdefault(t, []).append(c); entry[c] = e
takes = sorted(by_take)
nclips = sum(len(v) for v in by_take.values())
print(f"=== by-take Stage-1: {len(takes)} takes, {nclips} clips (vit-L, GT intrinsics, render deferred) ===", flush=True)

ok = fail = 0
for ti, take in enumerate(takes):
    clips = sorted(by_take[take])
    todo = [c for c in clips if not clip_done(c)]
    if not todo:
        ok += len(clips); continue
    print(f"\n[take {ti+1}/{len(takes)}] {take}: {len(todo)}/{len(clips)} clips", flush=True)
    src = f"{SEAGATE}/{take}/frame_aligned_videos/downscaled/448"
    cams = sorted(f for f in os.listdir(src) if re.match(r"cam\d+\.mp4$", f)) if os.path.isdir(src) else []
    aria = f"{src}/aria01_214-1.mp4"
    if not cams or not os.path.exists(aria):
        print(f"  !! missing cam/ego, skip take"); fail += len(todo); continue
    be = best_exo.get(take); camfile = f"{be}.mp4" if be and f"{be}.mp4" in cams else cams[0]
    exo_src = f"{src}/{camfile}"
    # trim all todo clips; collect vipe inputs (stem == clip)
    tin = OUT / "_vipe_input" / take
    if tin.exists(): shutil.rmtree(tin)
    tin.mkdir(parents=True, exist_ok=True)
    for c in todo:
        s, e = map(int, c.rsplit("_", 2)[-2:])
        vd = OUT / "videos" / c; vd.mkdir(parents=True, exist_ok=True)
        trim(exo_src, s, e, str(vd / "exo.mp4"))
        trim(aria, s, e, str(vd / "ego_GT.mp4"))
        shutil.copy(vd / "exo.mp4", tin / f"{c}.mp4")
    # GT intrinsics (same cam/res across the take) scaled to video res
    cap = cv2.VideoCapture(str(tin / f"{todo[0]}.mp4")); W = int(cap.get(3)); H = int(cap.get(4)); cap.release()
    K = [[float(x) for x in r] for r in entry[todo[0]]["camera_intrinsics"]]
    sx, sy = W / (2 * K[0][2]), H / (2 * K[1][2])
    Kgt = [[K[0][0]*sx, 0, W/2], [0, K[1][1]*sy, H/2], [0, 0, 1]]
    print(f"  cam={camfile} {W}x{H} | {len(todo)} clips -> ViPE directory mode (models load once)", flush=True)
    sh([f"{ENV}/bin/vipe", "infer", str(tin), "-o", str(OUT / "vipe_results"), "-p", "lyra",
        "--assume_fixed_camera_pose", "--end_frame", "48", "--use_exo_intrinsic_gt", json.dumps(Kgt)])
    # depth->npy + meta entry per clip
    for c in todo:
        vres = OUT / "vipe_results" / c
        if not (vres / "depth").exists():
            print(f"  !! no vipe output for {c}"); fail += 1; continue
        sh([f"{ENV}/bin/python", f"{RENDERER}/scripts/convert_depth_zip_to_npy.py",
            "--depth_path", str(vres / "depth"), "--egox_depthmaps_path", str(OUT / "depth_maps")], cwd=RENDERER)
        e = entry[c]; vd = OUT / "videos" / c
        json.dump({c: {
            "exo_video_path": str(vd/"exo.mp4"), "ego_video_path": str(vd/"ego_GT.mp4"),
            "ego_prior_path": str(vd/"ego_Prior.mp4"), "prompt": e["prompt"], "take_name": c,
            "vipe_results_path": str(vres), "best_camera": c,
            "camera_extrinsics": e["camera_extrinsics"], "camera_intrinsics": e["camera_intrinsics"],
            "ego_extrinsics": e["ego_extrinsics"], "ego_intrinsics": e["ego_intrinsics"]}},
            open(OUT / f"meta_entry_{c}.json", "w"))
        ok += 1 if clip_done(c) else 0
    shutil.rmtree(tin, ignore_errors=True)
    print(f"  --- take {take} done | total {ok} ok, {fail} failed ---", flush=True)

print(f"\n=== BY-TAKE BATCH DONE: {ok} ok, {fail} failed of {nclips} ===", flush=True)

#!/usr/bin/env python
"""Optimized single-process Stage-1 ViPE driver — loads vipe/torch/CUDA + the pipeline ONCE,
then loops over many clips calling pipeline.run() (mutating out_path + GT intrinsics per clip).
Kills the per-clip python/torch/JIT/CUDA-context startup that dominated the per-subprocess ~45s.

vitS by default (`lyra_svda`); shardable across machines via --shard i/N (split by take).

Usage:
  vipe_driver.py [--pipeline lyra_svda] [--shard 0/2] [--limit N] [--no-depth]
Env: egox-egoprior python, PYTHONNOUSERSITE=1 HF_HUB_ENABLE_HF_TRANSFER=0
"""
import argparse, json, os, re, shutil, subprocess, time
from pathlib import Path
import cv2
import hydra
import torch

from vipe import get_config_path, make_pipeline
from vipe.streams.base import ProcessedVideoStream
from vipe.streams.raw_mp4_stream import RawMp4Stream
from vipe.utils.logging import configure_logging

SEAGATE = "/media/skr/SeagateHub1/egoexo4d/takes"
META = "/media/skr/storage/paper_reproduction/egoX/egoexo4D/meta_train.json"
TAKES_JSON = "/media/skr/storage/paper_reproduction/hands/trihands/egoexo_data/takes.json"
RENDERER = "/media/skr/storage/paper_reproduction/egoX/EgoX/EgoX-EgoPriorRenderer"
ENVPY = "/media/skr/storage/conda_envs/egox-egoprior/bin/python"
OUT = Path("/media/skr/SeagateHub1/egoexo4d/cooking_train")
NF = 49


def trim(src, s, e, dst):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                    "-vf", f"select=between(n\\,{s}\\,{e}),setpts=N/FRAME_RATE/TB",
                    "-vsync", "0", "-an", dst], check=True)


def clip_done(c):
    d = OUT / "depth_maps" / c
    return (d.exists() and len(list(d.glob("*.npy"))) == NF
            and any((OUT / "vipe_results" / c / "intrinsics").glob("*.npz")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default="lyra_svda", help="vipe pipeline config (lyra_svda=vit-S, lyra=vit-L)")
    ap.add_argument("--shard", default="0/1", help="i/N — process takes where take_index %% N == i")
    ap.add_argument("--limit", type=int, default=0, help="cap #clips (0=all) — for timing")
    ap.add_argument("--no-depth", action="store_true", help="skip depth zip->npy (timing ViPE only)")
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))
    log = configure_logging()

    best_exo = {t["take_name"]: t.get("best_exo") for t in json.load(open(TAKES_JSON))}
    seg = set(os.listdir(SEAGATE))
    by_take, entry = {}, {}
    for e in json.load(open(META))["train_datasets"]:
        c = e["exo_path"].split("/")[-2]; t = re.sub(r"_\d+_\d+$", "", c)
        if "cooking" in c and t in seg:
            by_take.setdefault(t, []).append(c); entry[c] = e
    takes = [t for i, t in enumerate(sorted(by_take)) if i % sn == si]
    nclips = sum(len(by_take[t]) for t in takes)
    print(f"=== vipe_driver [{a.pipeline}] shard {si}/{sn}: {len(takes)} takes, {nclips} clips ===", flush=True)

    # ---- build the pipeline ONCE (models still lazy-load inside run(), but python/torch/JIT/CUDA
    #      context + import cost is paid a single time for the whole shard) ----
    overrides = [f"pipeline={a.pipeline}", f"pipeline.output.path={OUT/'vipe_results'}",
                 "pipeline.output.save_artifacts=true", "pipeline.output.save_viz=false",
                 "pipeline.assume_fixed_camera_pose=true", "pipeline.slam.optimize_intrinsics=false"]
    with hydra.initialize_config_dir(config_dir=str(get_config_path()), version_base=None):
        args = hydra.compose("default", overrides=overrides)
    pipeline = make_pipeline(args.pipeline)

    ok = fail = 0; done_n = 0; t0 = time.time()
    tin_root = OUT / "_drv_in"; tin_root.mkdir(parents=True, exist_ok=True)
    for take in takes:
        clips = sorted(by_take[take])
        todo = [c for c in clips if not clip_done(c)]
        if not todo:
            ok += len(clips); continue
        src = f"{SEAGATE}/{take}/frame_aligned_videos/downscaled/448"
        cams = sorted(f for f in os.listdir(src) if re.match(r"cam\d+\.mp4$", f)) if os.path.isdir(src) else []
        aria = f"{src}/aria01_214-1.mp4"
        if not cams or not os.path.exists(aria):
            print(f"  !! {take}: missing cam/ego, skip"); fail += len(todo); continue
        be = best_exo.get(take); camfile = f"{be}.mp4" if be and f"{be}.mp4" in cams else cams[0]
        exo_src = f"{src}/{camfile}"
        for c in todo:
            s, e = map(int, c.rsplit("_", 2)[-2:])
            vd = OUT / "videos" / c; vd.mkdir(parents=True, exist_ok=True)
            trim(exo_src, s, e, str(vd / "exo.mp4"))
            trim(aria, s, e, str(vd / "ego_GT.mp4"))
            clip_in = tin_root / f"{c}.mp4"; shutil.copy(vd / "exo.mp4", clip_in)
            # GT intrinsics scaled to this clip's res
            cap = cv2.VideoCapture(str(clip_in)); W = int(cap.get(3)); H = int(cap.get(4)); cap.release()
            K = [[float(x) for x in r] for r in entry[c]["camera_intrinsics"]]
            sx, sy = W / (2 * K[0][2]), H / (2 * K[1][2])
            Kgt = [[K[0][0]*sx, 0, W/2], [0, K[1][1]*sy, H/2], [0, 0, 1]]
            # --- run ViPE on this clip with the SAME loaded pipeline ---
            pipeline.use_exo_intrinsic_gt = Kgt
            pipeline.out_path = OUT / "vipe_results" / c
            pipeline.out_path.mkdir(parents=True, exist_ok=True)
            try:
                stream = ProcessedVideoStream(RawMp4Stream(clip_in, seek_range=range(0, NF)), []).cache(desc=f"read {c}")
                pipeline.run(stream)
            except Exception as ex:
                print(f"  !! {c}: vipe failed: {ex}"); fail += 1; clip_in.unlink(missing_ok=True); continue
            clip_in.unlink(missing_ok=True)
            torch.cuda.empty_cache()
            # depth zip -> npy
            if not a.no_depth:
                vres = OUT / "vipe_results" / c
                subprocess.run([ENVPY, f"{RENDERER}/scripts/convert_depth_zip_to_npy.py",
                                "--depth_path", str(vres / "depth"),
                                "--egox_depthmaps_path", str(OUT / "depth_maps")],
                               cwd=RENDERER, env={**os.environ, "PYTHONNOUSERSITE": "1"})
                e2 = entry[c]
                json.dump({c: {
                    "exo_video_path": str(vd/"exo.mp4"), "ego_video_path": str(vd/"ego_GT.mp4"),
                    "ego_prior_path": str(vd/"ego_Prior.mp4"), "prompt": e2["prompt"], "take_name": c,
                    "vipe_results_path": str(vres), "best_camera": c,
                    "camera_extrinsics": e2["camera_extrinsics"], "camera_intrinsics": e2["camera_intrinsics"],
                    "ego_extrinsics": e2["ego_extrinsics"], "ego_intrinsics": e2["ego_intrinsics"]}},
                    open(OUT / f"meta_entry_{c}.json", "w"))
            ok += 1; done_n += 1
            rate = (time.time() - t0) / done_n
            print(f"  [{done_n}] {c} done | {rate:.1f}s/clip avg | {ok} ok {fail} fail", flush=True)
            if a.limit and done_n >= a.limit:
                print(f"=== limit {a.limit} reached; avg {rate:.1f}s/clip ===", flush=True)
                shutil.rmtree(tin_root, ignore_errors=True); return
    shutil.rmtree(tin_root, ignore_errors=True)
    print(f"\n=== DRIVER DONE shard {si}/{sn}: {ok} ok, {fail} failed; avg {(time.time()-t0)/max(done_n,1):.1f}s/clip ===", flush=True)


if __name__ == "__main__":
    main()

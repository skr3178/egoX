#!/usr/bin/env python
"""Full cooking Stage-1 batch — vitS (lyra_svda) + faithful per-take Aria calib render.
Runs the validated per-clip orchestrator over every eligible cooking clip. Resumable
(skips clips already complete incl. ego_Prior), continues on per-clip failure.

Sharding for multi-machine / multi-worker:  --shard i/N  (processes clips where idx % N == i)
  e.g. 24 GB box (2 vitS workers): run --shard 0/3 and --shard 1/3 ; 12 GB box: --shard 2/3

Calib policy (default = faithful): only process clips whose take already has
online_calibration.jsonl on disk; re-run later to pick up newly-downloaded takes.
Pass --allow-generic to also process takes without calib (generic Aria coeffs).

Usage: PYTHONNOUSERSITE=1 egox-egoprior/bin/python stage1_full.py --shard 0/3 [--allow-generic] [--pipeline lyra_svda]
"""
import argparse, json, os, re, subprocess
SEA = "/media/skr/SeagateHub1/egoexo4d/takes"
META = "/media/skr/storage/paper_reproduction/egoX/egoexo4D/meta_train.json"
ORCH = "/media/skr/storage/paper_reproduction/egoX/local/stage1_cooking_clip.py"
ENVPY = "/media/skr/storage/conda_envs/egox-egoprior/bin/python"
OUT = "/media/skr/SeagateHub1/egoexo4d/cooking_train"


def clip_done(c, skip_render):
    d = f"{OUT}/depth_maps/{c}"
    ok = (os.path.isdir(d) and len([f for f in os.listdir(d) if f.endswith('.npy')]) == 49
          and os.path.isdir(f"{OUT}/vipe_results/{c}/intrinsics")
          and any(f.endswith('.npz') for f in os.listdir(f"{OUT}/vipe_results/{c}/intrinsics")))
    if ok and not skip_render:
        ok = os.path.exists(f"{OUT}/videos/{c}/ego_Prior.mp4")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1", help="i/N — process clips where idx %% N == i")
    ap.add_argument("--pipeline", default="lyra_svda", help="lyra_svda (vit-S) | lyra (vit-L)")
    ap.add_argument("--allow-generic", action="store_true", help="also process takes lacking online_calibration.jsonl")
    ap.add_argument("--skip-render", action="store_true", help="trim+ViPE+depth only (defer render)")
    ap.add_argument("--limit", type=int, default=0, help="cap #clips this shard processes (0=all) — for subset tests")
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))

    calib_takes = {d for d in os.listdir(SEA) if os.path.exists(f"{SEA}/{d}/trajectory/online_calibration.jsonl")}
    clips = []
    for e in json.load(open(META))["train_datasets"]:
        c = e["exo_path"].split("/")[-2]; t = re.sub(r"_\d+_\d+$", "", c)
        if "cooking" not in c:
            continue
        src = f"{SEA}/{t}/frame_aligned_videos/downscaled/448"
        if not (os.path.isdir(src) and os.path.exists(f"{src}/aria01_214-1.mp4")):
            continue
        if not a.skip_render and not a.allow_generic and t not in calib_takes:
            continue
        clips.append(c)
    clips = sorted(set(clips))
    mine = [c for i, c in enumerate(clips) if i % sn == si]
    if a.limit:
        mine = mine[:a.limit]
    print(f"=== stage1_full [{a.pipeline}] shard {si}/{sn}: {len(mine)}/{len(clips)} eligible clips "
          f"({len(calib_takes)} takes have calib){' [generic allowed]' if a.allow_generic else ' [calib-only]'} ===", flush=True)

    env = {**os.environ, "PYTHONNOUSERSITE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0"}

    # --- Skip corrupt takes: a take whose ego_prior renders BLACK (point cloud misses the ego FOV)
    # yields no conditioning signal; generating its other windows wastes compute. Shared across shards
    # via bad_takes.txt (pre-seeded with known whole-take failures, grown on the fly).
    import cv2, numpy as np
    BAD_TAKES_FILE = "/media/skr/storage/paper_reproduction/egoX/generated/bad_takes.txt"
    def load_bad_takes():
        if not os.path.exists(BAD_TAKES_FILE): return set()
        return {l.strip() for l in open(BAD_TAKES_FILE) if l.strip() and not l.startswith("#")}
    def mark_bad_take(t):
        if t in load_bad_takes(): return
        with open(BAD_TAKES_FILE, "a") as f:
            f.write(t + "\n")
    def prior_cov(c):
        p = f"{OUT}/videos/{c}/ego_Prior.mp4"
        if not os.path.exists(p): return 0.0
        cap = cv2.VideoCapture(p); fr = []
        while True:
            okk, frm = cap.read()
            if not okk: break
            fr.append(float((frm.max(axis=2) > 12).mean()))
        cap.release(); return float(np.mean(fr)) if fr else 0.0

    ok = skip = fail = skipbad = 0; failed = []
    for i, c in enumerate(mine):
        t = re.sub(r"_\d+_\d+$", "", c)
        if t in load_bad_takes():               # corrupt take -> don't waste time rendering it
            skipbad += 1; continue
        if clip_done(c, a.skip_render):
            skip += 1; continue
        print(f"\n[{i+1}/{len(mine)}] {c}", flush=True)
        cmd = [ENVPY, ORCH, c, "--pipeline", a.pipeline] + (["--skip-render"] if a.skip_render else [])
        r = subprocess.run(cmd, env=env)
        if r.returncode == 0:
            ok += 1
            if not a.skip_render:
                cov = prior_cov(c)
                if cov < 0.01:                    # BLACK prior -> mark take corrupt, skip its other windows
                    mark_bad_take(t)
                    print(f"  !! BLACK ego_prior (cov={cov:.2%}) -> take '{t}' corrupt; skipping its remaining windows", flush=True)
        else:
            fail += 1; failed.append(c); print(f"  !! FAILED {c} (rc={r.returncode})", flush=True)
        if (i + 1) % 25 == 0:
            print(f"  --- {ok} ok, {skip} done, {skipbad} skip-corrupt, {fail} fail, {i+1}/{len(mine)} ---", flush=True)
    print(f"\n=== SHARD {si}/{sn} DONE: {ok} processed, {skip} already-done, {skipbad} skip-corrupt, {fail} failed ===", flush=True)
    if failed:
        open(f"/media/skr/storage/paper_reproduction/egoX/local/stage1_failed_{si}_{sn}.txt", "w").write("\n".join(failed) + "\n")


if __name__ == "__main__":
    main()

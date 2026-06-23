#!/usr/bin/env python
"""Produce 5 fully-ready cooking samples with the vitS pipeline (trim + ViPE lyra_svda + depth +
faithful ego-prior render w/ per-take Aria calib + meta entry). Sequential. Resumable.

After inspection, the SAME orchestrator call (--pipeline lyra_svda) scales to the full set.
"""
import json, os, re, subprocess, sys
SEA = "/media/skr/SeagateHub1/egoexo4d/takes"
META = "/media/skr/storage/paper_reproduction/egoX/egoexo4D/meta_train.json"
ORCH = "/media/skr/storage/paper_reproduction/egoX/local/stage1_cooking_clip.py"
ENVPY = "/media/skr/storage/conda_envs/egox-egoprior/bin/python"
OUTDM = "/media/skr/SeagateHub1/egoexo4d/cooking_train/depth_maps"
N = 5

calib_takes = {d for d in os.listdir(SEA) if os.path.exists(f"{SEA}/{d}/trajectory/online_calibration.jsonl")}
picks = []
for e in json.load(open(META))["train_datasets"]:
    c = e["exo_path"].split("/")[-2]; t = re.sub(r"_\d+_\d+$", "", c)
    if "cooking" not in c or t not in calib_takes:
        continue
    src = f"{SEA}/{t}/frame_aligned_videos/downscaled/448"
    if not (os.path.isdir(src) and os.path.exists(f"{src}/aria01_214-1.mp4")):
        continue
    # prefer fresh clips (no ego_Prior yet) so the full pipeline incl. render runs
    if os.path.exists(f"/media/skr/SeagateHub1/egoexo4d/cooking_train/videos/{c}/ego_Prior.mp4"):
        continue
    picks.append(c)
    if len(picks) >= N:
        break

print(f"=== 5-sample vitS test set ({len(calib_takes)} takes have calib) ===")
for c in picks:
    print("  ", c)
print()

env = {**os.environ, "PYTHONNOUSERSITE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0"}
ok = 0
for i, c in enumerate(picks):
    print(f"\n########## [{i+1}/{len(picks)}] {c} ##########", flush=True)
    r = subprocess.run([ENVPY, ORCH, c, "--pipeline", "lyra_svda"], env=env)
    if r.returncode == 0:
        ok += 1
    else:
        print(f"  !! FAILED {c} (rc={r.returncode})", flush=True)

print(f"\n=== 5-SAMPLE TEST DONE: {ok}/{len(picks)} clips ready ===", flush=True)

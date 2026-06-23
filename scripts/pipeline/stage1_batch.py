#!/usr/bin/env python
"""Batch Stage-1 (trim + ViPE + depth, NO render) over all cooking clips in meta_train whose
take is on the Seagate disk. Resumable (orchestrator skips done clips), continues on per-clip
failure. ego_prior render is deferred (needs online_calibration.jsonl) -> run with --skip-render.

Run detached:  PYTHONNOUSERSITE=1 egox-egoprior/bin/python stage1_batch.py
"""
import json, os, re, subprocess, time

SEAGATE = "/media/skr/SeagateHub1/egoexo4d/takes"
META = "/media/skr/storage/paper_reproduction/egoX/egoexo4D/meta_train.json"
ENVPY = "/media/skr/storage/conda_envs/egox-egoprior/bin/python"
ORCH = "/media/skr/storage/paper_reproduction/egoX/local/stage1_cooking_clip.py"

seg = set(os.listdir(SEAGATE))
clips = []
for e in json.load(open(META))["train_datasets"]:
    c = e["exo_path"].split("/")[-2]
    t = re.sub(r"_\d+_\d+$", "", c)
    if "cooking" in c and t in seg:
        clips.append(c)
clips = sorted(set(clips))
print(f"=== Stage-1 batch: {len(clips)} cooking clips (trim+ViPE+depth, render deferred) ===", flush=True)

env = {**os.environ, "PYTHONNOUSERSITE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0"}
ok = fail = 0
failed = []
for i, c in enumerate(clips):
    print(f"\n[{i+1}/{len(clips)}] {c}", flush=True)
    r = subprocess.run([ENVPY, ORCH, c, "--skip-render"], env=env)
    if r.returncode == 0:
        ok += 1
    else:
        fail += 1; failed.append(c); print(f"  !! FAILED {c} (rc={r.returncode})", flush=True)
    if (i + 1) % 25 == 0:
        print(f"  --- progress: {ok} ok, {fail} failed, {i+1}/{len(clips)} ---", flush=True)

print(f"\n=== STAGE-1 BATCH DONE: {ok} ok, {fail} failed of {len(clips)} ===", flush=True)
if failed:
    open("/media/skr/storage/paper_reproduction/egoX/local/stage1_failed.txt", "w").write("\n".join(failed) + "\n")
    print("failed clips -> local/stage1_failed.txt", flush=True)

# EgoX Cooking — QLoRA Training Plan & Stage-1 Run Commands

Single 24 GB RTX PRO 4000 Blackwell (+ optional 12 GB second machine).
Last updated 2026-06-23.

---

## 0. TL;DR

1. **Stage-1 data prep** (per clip): trim → **vitS** ViPE depth → faithful per-take-calib ego-prior render → meta entry.
   Run `local/stage1_full.py` sharded across workers. Resumable.
2. **Resolution reduction is NOT a prep step** — it's a *training flag* `--train_resolution 49x256x704`.
   The loader downsamples (exo/ego/prior/depth) on decode + caches reduced latents on epoch 1. Originals stay native.
3. **Then**: assemble cooking meta → QLoRA train (NF4 14B + LoRA) at `49x256x704`.

All launches need `PYTHONNOUSERSITE=1` (a `~/.local` torch 2.11+cu130 leak otherwise shadows the env);
training benefits from `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

---

## 1. Stage-1 — full cooking run (vitS + faithful calib)

### Components (validated on 5 samples, 5/5 OK)
- `local/stage1_cooking_clip.py --pipeline lyra_svda` — per-clip orchestrator:
  trim exo+ego_GT → `vipe infer -p lyra_svda` (vitS, GT exo intrinsics from meta_train) →
  depth zip→npy → ego-prior render (`--fish_eye_rendering --use_mean_bg --only_bg`
  + `--online_calibration_path <take>/trajectory/online_calibration.jsonl`) → meta entry.
  Resumable (skips clips already complete **incl. ego_Prior**).
- `local/stage1_full.py` — batch wrapper over all eligible cooking clips:
  `--shard i/N` (clips where idx % N == i), calib-only by default, `--allow-generic` to include
  takes without calib, `--skip-render` to defer render. Continues on per-clip failure
  (logs `local/stage1_failed_<i>_<N>.txt`).

### Launch — 2-machine / 3-worker (24 GB fits 2× vitS @ ~11 GB; 12 GB fits 1)

```bash
cd /media/skr/storage/paper_reproduction/egoX
EP=/media/skr/storage/conda_envs/egox-egoprior/bin/python

# --- 24 GB machine: two workers in parallel ---
PYTHONNOUSERSITE=1 HF_HUB_ENABLE_HF_TRANSFER=0 $EP local/stage1_full.py --shard 0/3   # worker A
PYTHONNOUSERSITE=1 HF_HUB_ENABLE_HF_TRANSFER=0 $EP local/stage1_full.py --shard 1/3   # worker B

# --- 12 GB machine: one worker ---
PYTHONNOUSERSITE=1 HF_HUB_ENABLE_HF_TRANSFER=0 $EP local/stage1_full.py --shard 2/3

# detached example (per worker):
setsid nohup bash -c 'PYTHONNOUSERSITE=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
  /media/skr/storage/conda_envs/egox-egoprior/bin/python local/stage1_full.py --shard 0/3' \
  > local/stage1_full_0_3.log 2>&1 < /dev/null &
```

- ~45 s/clip; ~259 clips/worker. The 24 GB box's two workers share compute (~1.6–1.8×, not 2×).
  Expect **~3.5–4 h wall** for the full set across the 3 workers.
- Single-machine fallback (24 GB, 2 workers): `--shard 0/2` and `--shard 1/2`.

### Calib policy
- Default = **calib-only** (faithful): only clips whose take already has `online_calibration.jsonl`.
  Re-run later to pick up newly-downloaded takes (resumable).
- `--allow-generic`: also process takes lacking calib (renderer's built-in generic Aria coeffs).
- Status at last check: **235/266 takes have calib** → 776 calib-only clips / 889 total; 6 already done.
  Even 3-way shard split (calib-only): [259, 259, 258].

### Outputs (per clip) → `/media/skr/SeagateHub1/egoexo4d/cooking_train/`
- `videos/<clip>/{exo,ego_GT,ego_Prior}.mp4` (49 frames each)
- `depth_maps/<clip>/*.npy` (49, **vitS**)
- `vipe_results/<clip>/intrinsics/<clip>.npz` `{data:(N,4)=[fx,fy,cx,cy]}`
- `meta_entry_<clip>.json` (training-schema entry)

### vitS vs vitL
- vitS (`lyra_svda`) = MoGe-2 + **VDA-Small**; peak **~11 GB** (fits 12 GB; 2 fit on 24 GB).
- Only the **depth** stage differs (VDA backbone). Intrinsics identical; per-clip speed ~same as vitL.
- The win is **memory → parallelism**, not per-clip speed. Cost: slightly more temporal depth flicker.
- Checkpoints local: `~/.cache/torch/hub/checkpoints/video_depth_anything_vits.pth` (+ dinov2_vits14, MoGe-2).

### Inspection previews (built for the 5-sample test)
`previews/vits5_grid.png` (5× ego_GT|ego_Prior), `previews/vits5/<clip>_cmp.mp4` (exo|ego_GT|ego_Prior),
`previews/REF_cooking57_mid.png` (authors' reference — confirms sparse+black ego_Prior is normal).

---

## 2. Resolution — handled at TRAIN time (no prep step)

`core/finetune/datasets/wan_dataset.py`:
- Videos resized on decode (`preprocess_video_with_resize`); depth/point-map `F.interpolate` to exo size (:381).
- Cache keyed by `train_resolution_str` (:168) → each resolution caches separately.
- **Frames hard-locked at 49**; only spatial H×W is reducible. Token N = 13·(H/16)·(W/16); GGA buffer = N²·4 B.
- **Chosen `49x256x704`** (N≈9,152, GGA buffer ~0.33 GB). Paper `49x448x1232` won't fit (3.14 GB GGA buffer).
- Phase A validated this resolution end-to-end (peak 22.8 GB / 24 GB).
- **Never pre-resize files** — pass `--train_resolution 49x256x704`; loader downsamples + pre-encodes epoch 1.

---

## 3. After Stage-1 → QLoRA training

1. **Assemble training meta**: merge all `meta_entry_*.json` → one training-schema `meta.json`
   keyed by clip (the dict the loader reads). Point `--data_root` at `.../cooking_train`. (script TBD)
2. **Train** (`scripts/finetune_phaseA.sh` pattern):
   - `EGOX_NF4=1 PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   - `configs_acc/1gpu.yaml`, `--training_type lora`, `--train_resolution 49x256x704`
   - real run: `--rank 256 --lora_alpha 256`, more epochs/steps (Phase A used rank 64 / 20 steps as a smoke).
   - first launch precomputes latents/embeds/GGA → unloads encoders → only transformer resident.
- **Loss** = flow-matching MSE on the **ego region only**: predicted velocity vs target `(noise − latent)`
  (`sft_trainer.py:932,951,953`). Exo region is clean in-context conditioning, excluded from loss.
- **Never `fuse_lora`** on the NF4 base (unfused adapter only).

---

## 4. EgoX submodule code changes (local, uncommitted)
- `core/finetune/models/wan_i2v/sft_trainer.py` — `import os` + gated NF4 load (`EGOX_NF4=1`).
- `core/finetune/datasets/wan_dataset.py` — repo bug fix: added missing `self.prompts = []`.
- `EgoX-EgoPriorRenderer/scripts/render_vipe_pointcloud.py` — 4→3 return-tuple fix; supports `--online_calibration_path`.
- `configs_acc/1gpu.yaml` — gpu_ids → "0".
- `local/stage1_cooking_clip.py` — added `--pipeline` (vitS) + auto per-take calib in render + render-aware resume.

---

## 5. Trajectory dataset — keep/discard (settled)
Keep: `online_calibration.jsonl` (ego fisheye distortion — needed for render),
`gopro_calibs.csv` (tiny), `summary.json` (tiny).
Discard (~43 GB): `open_loop_trajectory.csv`, `closed_loop_trajectory.csv`
— ego pose already baked into meta_train `ego_extrinsics (49,3,4)`; render reads poses from meta, not the CSVs.
(Holds because clips use meta_train's exact frame ranges.)

---

## 6. Phase A reference (DONE — single-GPU QLoRA loop validated)

PASSED on 1 ready cooking clip (`iiith_cooking_57_2_2451_2499`) at `49x256x704`:
precompute → encoder-unload → 20 steps, exit 0, peak 22.8 GB, ckpts at `results/EgoX_phaseA/checkpoint-{10,20}`.

### NF4 injection (gated) — `sft_trainer.py` (keeps 4-GPU bf16 path intact)
```python
if os.environ.get("EGOX_NF4") == "1":
    from diffusers import BitsAndBytesConfig
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    components.transformer = WanTransformer3DModel.from_pretrained(
        model_path, subfolder="transformer", quantization_config=qcfg, torch_dtype=torch.bfloat16)
else:
    components.transformer = WanTransformer3DModel.from_pretrained(model_path, subfolder="transformer")
```
Safe: `_GGA` has `PeftAdapterMixin` → `from_pretrained(quantization_config=...)` + `add_adapter` on the
4-bit base (canonical QLoRA). `cast_training_params(fp32)` + `__move_components_to_device` (transformer
in ignore_list) both fine on the 4-bit base.

### Training args (smoke used)
`--training_type lora --rank 64 --lora_alpha 64 --gradient_checkpointing True --mixed_precision bf16
--batch_size 1 --gradient_accumulation_steps 1 --train_steps 20 --checkpointing_steps 10`.
Optional 8-bit AdamW: `use_8bit=True` (`trainer.py:274`) — LoRA optimizer state is tiny, so optional.

### Success signals
- Precompute completes → VAE + text_encoder unloaded; only transformer resident.
- Peak VRAM < 24 GB; `loss` finite & decreasing; `grad_norm` finite/non-zero (LoRA grads flow).
- `results/.../checkpoint-N/pytorch_lora_weights.safetensors` written.
- Sanity: `run_infer_nf4.sh --lora_path ... --use_GGA` → output moves toward `ego_GT` vs base. LoRA stays unfused on NF4.

### Risks / gotchas
- Frames locked at 49 — reduce only H/W.
- NF4 injection env-gated (keep 4-GPU bf16 path).
- Never `fuse_lora` on NF4 base.
- `PYTHONNOUSERSITE=1` mandatory; `expandable_segments:True` to recover fragmentation.
- Cache keyed by resolution → switching res re-precomputes (clean old cache dirs if disk-tight).

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

### Ego-prior quality — what signal it carries, and why it's sparse (analyzed 2026-06-24)
**Renderer:** `EgoX/EgoX-EgoPriorRenderer/scripts/render_vipe_pointcloud.py` — a **point-cloud splat**
(pytorch3d `FishEyeCameras`, `--point_size 5.0`) of a ViPE-depth point cloud, reprojected from the exo
view into the ego Aria-fisheye pose. **Depth model = ViPE vitS** (`lyra_svda` = MoGe-2 + VDA-Small).
There is **no mesh/surface/inpainting** — discrete points → black holes between them by construction.

**Why the prior looks poor (5 compounding reasons):**
1. **Background-only** (`build_background_pointcloud`, `static_mask = instance_mask == 0`): the renderer
   **drops all dynamic foreground** (hands, the person, manipulated objects). For a hand-object ego task
   the most important content — hands at the cutting board — is *deliberately excluded* from the prior.
   (Authors' design, not ours.)
2. **Subsample + splat** (`spatial_subsample=2`): keeps ~1/4 of pixels, rendered as discrete points → gaps.
3. **vitS depth** (our speed choice): noisier/lower-fidelity than vitL → more points fail
   `reliable_depth_mask_range` and get dropped; positions jitter.
4. **exo→ego reprojection occlusion:** points come from far third-person exo cameras reprojected into the
   close first-person ego; the large viewpoint gap means much of the ego FOV was never seen by exo → holes.
5. **SLAM-map path unused:** the renderer *prioritizes an Aria MPS semi-dense SLAM map if present*, else
   falls back to this sparse depth method. We have no MPS points → always on the sparse fallback. The
   paper's priors were likely **denser** (MPS semi-dense static cloud).

**Is the signal adequate?** The prior is a **coarse geometric scaffold by design** — sparse+black matches
the authors' reference (`previews/REF_cooking57_mid.png`), so it is *normal*, not a bug. The **dominant**
conditioning signal is the **exo in-context video** the transformer attends to (the whole IC-LoRA point);
the prior just hints scene geometry/color, and the model hallucinates dynamic detail. **But** our prior is
poorer than the paper's (vitS + no-SLAM + background-only) → it carries little signal for *foreground/hands*,
which plausibly limits ceiling on hand-object fidelity. ckpt-100's predicted ego (coarse warm field where
GT has food + gray blob where hands are) is consistent with "tracking the coarse prior, undertrained for detail."

**Levers if priors prove limiting (cost ↑):** (a) vitL depth; (b) feed Aria MPS semi-dense SLAM points
(renderer already prefers them → much denser); (c) larger `point_size` / no subsample / point dilation;
(d) keep foreground (drop the `instance==0` filter) so hands appear in the prior. Defer until a *mature*
checkpoint shows the prior — not training maturity — is the bottleneck.

---

## 2. Resolution — handled at TRAIN time (no prep step)

`core/finetune/datasets/wan_dataset.py`:
- Videos resized on decode (`preprocess_video_with_resize`); depth/point-map `F.interpolate` to exo size (:381).
- Cache keyed by `train_resolution_str` (:168) → each resolution caches separately.
- **Frames hard-locked at 49**; only spatial H×W is reducible. Token N = 13·(H/16)·(W/16); GGA buffer = N²·4 B.
- **Chosen `49x256x704`** (N≈9,152, GGA buffer ~0.33 GB). Paper `49x448x1232` won't fit (3.14 GB GGA buffer).
- Phase A validated this resolution end-to-end (peak 22.8 GB / 24 GB).
- **Never pre-resize files** — pass `--train_resolution 49x256x704`; loader downsamples + pre-encodes epoch 1.

### Cache contents (verified) — what precompute writes per clip
Inspected `cache/.../49x256x704/<clip>_exo_exo_ego_gt_prior.safetensors` etc. All present, no NaNs:

| Cache file | Key | Shape | Meaning |
|---|---|---|---|
| **video_latent** | `encoded_exo_ego_gt_video` | (16, 13, 32, 88) bf16 | Wan VAE: 16 ch, 49→13 latent frames, 256/8=32 H, 704/8=88 W |
| | `encoded_exo_ego_prior_video` | (16, 13, 32, 88) bf16 | exo+ego_prior concat, same shape |
| **attn_maps** (GGA) | `attn_maps` | (13, 32, 88, 3) f32, [-1,1] | geometry cos-sim directions |
| | `attn_masks` | (13, 32, 88, 3) f32, [0,1] | valid-region mask |
| | `cam_rays` | (13, 32, 32, 3) f32 | ego rays (ego 256×256 → 32×32 latent) |
| | `point_vecs` | (13, 13, 32, 56, 3) f32 | exo points (exo = 56 latent-W) |
| **image_embeddings** | `image_embedding` | (257, 1280) f32 | CLIP-H: 256 patches+CLS, 1280 dim |
| **prompt_embeddings** | `prompt_embedding` | (512, 4096) bf16 | UMT5: 512 seq, 4096 dim |

- Width split: 88 latent-W = **56 (exo) + 32 (ego)** → concat ratio preserved; ego region (last 32 cols) matches the loss slice (`-ego_latent_width`).
- **Cache key = clip name** (fixed `wan_dataset.py:182` from `[-4]`='vipe_results' which collided all clips onto one latent file).

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

### Official EgoX defaults vs our 104-clip config (criticality)

Official = `scripts/finetune.sh` + `core/finetune/schemas/args.py` argparse defaults (effective values;
finetune.sh doesn't override the optimizer/LR block). LR=2e-5 confirmed by Phase A's `lr` ramp
(2e-7@step1, 4e-6@step20 = 2e-5·step/100).

Legend — **Matches?**: ✅ identical · ⚙️ value differs but achieves the same effect · ✗ diverges (reason).

| Param | Official | Ours | Matches? | Critical? |
|---|---|---|---|---|
| GPUs | 4 | 1 (24 GB) | ✗ — only 1 GPU available | forced |
| **Effective batch** | **4** (1×1×4) | **4** (1 × accum 4) | ✅ yes | 🔴 |
| `gradient_accumulation_steps` | 1 | 4 | ⚙️ differs — recovers batch 4 on 1 GPU | 🔴 |
| **LoRA rank / alpha** | 256 / 256 | 256 / 256 | ✅ yes | 🔴 |
| **train_resolution** | 49×448×1232 | 49×256×704 | ✗ — reduced; GGA N²·4B buffer won't fit 24 GB | 🟠 forced |
| **base precision** | bf16 (full) | NF4 4-bit (QLoRA) | ✗ — quantized to fit 14B in 24 GB | 🟠 forced |
| **epochs** | 150 (~3510 clips) | 100 → 300 (104 clips) | ✗ — scaled to 104 clips (150 would overfit) | 🔴 |
| learning_rate | 2e-5 | 2e-5 | ✅ yes (valid *because* batch = 4) | 🟢 |
| optimizer | AdamW | AdamW | ✅ yes | 🟢 |
| β1/β2 (β3), eps, weight_decay | 0.9/0.95 (0.98), 1e-8, 1e-4 | same | ✅ yes | 🟢 |
| max_grad_norm | 1.0 | 1.0 | ✅ yes | 🟢 |
| lr_scheduler | constant_with_warmup | constant_with_warmup | ✅ yes | 🟢 |
| **lr_warmup_steps** | 100 | 30 | ✗ — intentional; early signal on small set (faithful = 100) | 🟡 |
| gradient_checkpointing | True | True | ✅ yes | 🟢 |
| target_modules | to_q,to_k,to_v,to_out.0 | same | ✅ yes | 🟢 |
| checkpointing_steps | 250 | 100 | ✗ — operational only; more frequent resumable ckpts | minor |

**Critical notes:**
- **Effective batch 4 + LR 2e-5 are one coupled decision** — official LR was tuned for batch 4; we recover
  batch 4 via grad_accum 4 (not a larger per-GPU micro-batch, which official never used), so 2e-5 stays valid.
- **Don't copy 150 epochs** — that was over ~3510 clips. On 104 clips, 100 ep ≈ 2,600 steps, 300 ep ≈ 7,800.
- **Forced divergences** (res 256×704, NF4) are unavoidable on 24 GB; everything else matches the paper.
- **warmup 30** (vs 100) is a deliberate choice for an early loss signal on the small set; faithful = 100.

**Chosen launch** (`scripts/finetune_cooking100.sh`, NOT yet run): rank/alpha 256, batch 1 × grad_accum 4,
LR 2e-5, warmup 30, constant_with_warmup, AdamW(0.9,0.95) wd 1e-4 clip 1.0, bf16, NF4, 49×256×704,
train_epochs 100, ckpt every 100 (limit 10), report_to tensorboard.
**Wall-clock:** ~84 s/optimizer-step (grad_accum 4 × ~21 s/micro-step) → 100 ep ≈ **~60 h** (300 ep ≈ ~180 h);
loss signal visible in ~30–60 min, first checkpoint ~2–3 h. Stoppable + resumable.

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

## 5a. RESOLVED (2026-06-24): inference GGA mismatch was REAL, not compensating — now FIXED
Found in the 2026-06-23 audit; **numerically verified + fixed 2026-06-24.** The Geometry-Guided-Attention
(cam_rays / point_vecs) was computed **differently** at inference vs at training-cache time — 4 differences:

| step | training-cache (`core/finetune/datasets/wan_dataset.py`) | OLD inference (`infer_nf4.py` / authors' `infer.py`) |
|---|---|---|
| ray unprojection | pinhole `inv_intrinsics @ pixel_coords` | `cv2.undistortPoints` (Aria fisheye coeffs) |
| ego rotation | `cam_rays @ R.transpose(-1,-2)` | `cam_rays @ R` (no transpose) |
| 90° rotation | matrix `[[0,1,0],[-1,0,0],[0,0,1]]` | `torch.rot90(k=-1, dims=[1,2])` |
| point_map | no crop (interpolate full) | center-crop to (H, W-H) |

**Verdict: NOT compensating — they diverge badly.** Numerical check (`local/gga_check.py`, one val clip,
sanity: training-replica ≡ cache cos=1.0000):

| GGA component | OLD inference vs training | after fix |
|---|---|---|
| cam_rays (ego) | cos **0.28** (16% of rays inverted, cos<0) | **1.0000** |
| point_vecs (exo) | cos **0.33** (26% inverted) | **1.0000** |

So OLD inference fed the LoRA **near-orthogonal (out-of-distribution) conditioning** → artifacts. The
LoRA learned the `wan_dataset.py` convention, so inference MUST match *that*.

**Resolution sensitivity (why it surfaced now):** the author defaults were ~OK at the paper's 448×1232
(cam_rays cos 0.79; point_vecs crop ≈ no-op so ~1.0) and our **resolution reduction to 176×704** is what
broke them (0.28 / 0.33). The center-crop crops to `(PIX_H, PIX_W-PIX_H)` pixels — at 448 that ≈ the
depth-map size (no-op), at 176 it destroys most exo geometry. Training never crops & always uses
pinhole+Rᵀ+matrix, so aligning inference to training is correct at **both** resolutions.

**Fix applied to `EgoX/infer_nf4.py`** (inference-only; no training/cache impact): (1) cam_rays →
pinhole + Rᵀ (matches `wan_dataset.py:329-340`); (2) 90° → vector matrix-rot, not `rot90`
(`:407`); (3) point_vecs → drop the center-crop (`:384`). Verified the corrected inference reproduces
cached GGA at cos=1.0000. Diagnostic kept at `local/gga_check.py`; re-run script
`local/run_infer_ckpt100_ggafix.sh`.

**ckpt-100 re-eval after fix:** the exo half decodes razor-sharp (proves mask/decode/crop sound); only
the rightmost 176×176 **ego** is transformer-generated and it is still a degenerate low-frequency blob —
attributable to undertraining (~6 epochs), NOT the pipeline. Eval format going forward: GT | Prior |
Predicted, **ego-cropped** (the output mp4 is a 704×176 `[exo|ego]` strip; crop `=176:176:528:0`).

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

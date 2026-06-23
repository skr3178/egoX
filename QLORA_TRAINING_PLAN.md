# Small-Scale EgoX QLoRA Training Plan (single 24 GB Blackwell GPU)

## Context

We want a small-scale EgoX training run on one 24 GB GPU. The released training path is
4-GPU bf16 (won't fit: bf16 14B = 28 GB > 24 GB), so we use **QLoRA** (NF4 4-bit base +
LoRA), which we already proved loads at 8.6 GB on this card. The user also wants the dataset
**pre-encoded early** — which EgoX already does (its trainer caches latents/embeds once, then
unloads the encoders, so only the transformer is resident during training).

Data reality: only **1 cooking clip is Stage-1-ready** today (`iiith_cooking_57_2_2451_2499`
in `EgoX/example/egoexo4D/`). The 266 cooking takes on Seagate are **downscaled frames only**
(no depth/ego_prior/intrinsics). So the plan is **phased**:
- **Phase A** — validate the single-GPU QLoRA loop end-to-end on the 1 ready cooking clip.
- **Phase B** — run vit-L Stage-1 on Seagate cooking takes to build a real cooking set, then train.

Hard constraint (verified): **frames are locked at 49** (`wan_dataset.py:128` assert +
`ego_extrinsic[::4]` hardwiring + 49 depth `.npy`). The only memory lever is **spatial H/W**.

All launches require `PYTHONNOUSERSITE=1` (a `~/.local` torch 2.11+cu130 leak otherwise shadows
the env) and benefit from `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

---

## Phase A — Validate QLoRA loop on 1 cooking clip

### A1. Inject NF4 (gated) — `core/finetune/models/wan_i2v/sft_trainer.py:719`
Replace the bf16 transformer load with an env-gated NF4 load (same pattern proven in
`infer_nf4.py:66-69`), so the 4-GPU bf16 path is untouched:
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
Verified safe: `_GGA` is `(ModelMixin, ConfigMixin, PeftAdapterMixin, ...)` so `from_pretrained`
takes `quantization_config` and `add_adapter` works on the 4-bit base (canonical QLoRA).
`cast_training_params(fp32)` (trainer.py:258) and `__move_components_to_device` (trainer.py:508,
transformer in ignore_list) are both safe on the 4-bit base — no change needed.

### A2. Single-GPU launch
Use existing `configs_acc/1gpu.yaml` (already `distributed_type: NO`, `num_processes: 1`); set
`gpu_ids` to a **free** GPU. Make a Phase-A copy of `scripts/finetune.sh`:
`--config_file configs_acc/1gpu.yaml`, `num_processes 1`, prepend
`EGOX_NF4=1 PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

### A3. Data wiring (1 clip) — reuse `egox_blackwell_local/gen_stageB_inputs.py`
It converts the inference-schema `example/egoexo4D/meta.json` → training-schema meta (the 11 keys
`wan_dataset.py:86-96` reads) and writes the ViPE intrinsics `.npz`
(`{'data':(N,4)=[fx,fy,cx,cy]}`) to `example/egoexo4D/vipe_results/<take>/intrinsics/<take>.npz`.
Produce a **1-entry** meta with just `iiith_cooking_57_2_2451_2499`. Cross-check the `.npz` against
the real GeoCalib values in `vitS/vipe_intrinsics_egoexo4d_5clips.zip` (cooking clip is included).
Set `--data_root ./example/egoexo4D` (depth at `depth_maps/<take>/*.npy`, 49 present) and
`--meta_data_file ./example/egoexo4D/meta_train_smoke_1clip.json`.

### A4. Training args (smoke)
`--training_type lora --rank 64 --lora_alpha 64` (smaller is fine for the smoke; 256 matches paper),
`--gradient_checkpointing True --mixed_precision bf16 --batch_size 1 --gradient_accumulation_steps 1`,
`--train_steps 20 --checkpointing_steps 10 --checkpointing_limit 2`.
**Resolution (frames locked at 49, vary spatial):** start `--train_resolution 49x224x616` (half the
paper's 448x1232, same exo:ego concat ratio so the `W-H` slicing stays valid; H divisible by 16).
If it fits with headroom, step up 49x320x880 → 49x384x1056 → 49x448x1232, watching `nvidia-smi`
peak. Cache is keyed by `train_resolution_str` (wan_dataset.py:167), so each res caches separately.
**Optional 8-bit AdamW:** pass `use_8bit=True` at `trainer.py:274` (LoRA optimizer state is tiny, so
optional).

### A5. Success signals
- Precompute loop completes (trainer.py:184-196) → "Done" → VAE + text_encoder unloaded (trainer.py:198-199).
- Peak VRAM < 24 GB (monitor via the `nvidia-smi` loop in `run_infer_nf4.sh`); expect ~8.6 GB + activations.
- `loss` (trainer.py:439) finite & generally decreasing over 20 steps; `grad_norm` (trainer.py:427) finite/non-zero (LoRA grads flow).
- `results/.../checkpoint-10/pytorch_lora_weights.safetensors` written (save hook trainer.py:544).

### A6. Sanity-check the trained LoRA
Run `run_infer_nf4.sh` with `--lora_path results/.../checkpoint-20` on the same cooking clip
(`--use_GGA`), compare generated ego frames vs `…/iiith_cooking_57_2_2451_2499/ego_GT.mp4`.
Overfit-on-1-clip should move output measurably toward ego_GT vs the base (confirms it *trained*,
not just *ran*). LoRA stays unfused on NF4 (infer_nf4.py:79).

---

## Phase B — vit-L Stage-1 on Seagate cooking, then train

Run in env `egox-egoprior` (`PYTHONNOUSERSITE=1 HF_HUB_ENABLE_HF_TRANSFER=0`), faithful `lyra`
(vit-L) config. **GPU contention:** vit-L ViPE needs ~21 GB; cannot coexist with the 12.6 GB
other-session process — run on a free GPU or serialize. Stage-1 (data gen) and Phase-A/training
run sequentially, not concurrently.

Per Seagate take (downscaled `frame_aligned_videos/downscaled/448/cam*.mp4` is usable — GeoCalib
estimates intrinsics from frames):
1. `vipe infer <exo cam.mp4> -o <out> -p lyra --assume_fixed_camera_pose` (template: `run_vipe_accurate.sh`) → depth zip + `intrinsics/*.npz` + poses.
2. `EgoX-EgoPriorRenderer/scripts/convert_depth_zip_to_npy.py` → `depth_maps/<take>/*.npy`.
3. `EgoX-EgoPriorRenderer/scripts/render_vipe_pointcloud.py` → `ego_Prior.mp4`.
4. Use ViPE `intrinsics/<best_camera>.npz` directly (already `{'data':(N,4)}` — no conversion).
5. **Trim** each take into 49-frame windows → per window: `exo.mp4`, `ego_GT.mp4`, `ego_Prior.mp4`
   (49 frames each), 49 depth `.npy`, `ego_extrinsics` `(49,3,4)`, `camera_extrinsics` `(3,4)`,
   `camera_intrinsics`/`ego_intrinsics` `(3,3)`.
6. Assemble a training-schema `meta.json` (dict keyed by clip; template = `gen_stageB_inputs.py:69-81`),
   one entry per window. Point `--data_root` at the new dir on Seagate (e.g.
   `/media/skr/SeagateHub1/egoexo4d/cooking_train/`); first launch runs precompute to fill
   `{data_root}/cache/`, then trains with encoders unloaded.
7. QLoRA-train as in Phase A (same NF4 + 1gpu + reduced-spatial config), now over the cooking set;
   raise `--train_steps`/epochs and `rank`→256 for a real run.

---

## Files to modify
- `core/finetune/models/wan_i2v/sft_trainer.py:719` — gated NF4 `BitsAndBytesConfig`.
- `core/finetune/trainer.py:274` — optional `use_8bit=True` for 8-bit AdamW.
- `configs_acc/1gpu.yaml` — free `gpu_ids`.
- `scripts/finetune.sh` — Phase-A variant (1gpu, EGOX_NF4=1, 1-clip meta, reduced `--train_resolution`, `--train_steps 20`).

## Reusable (no edit)
`egox_blackwell_local/gen_stageB_inputs.py` (1-clip data wiring), `infer_nf4.py` + `run_infer_nf4.sh`
(NF4+LoRA sanity check), `run_vipe_accurate.sh` (Stage-1 template),
`EgoX-EgoPriorRenderer/scripts/{convert_depth_zip_to_npy.py,render_vipe_pointcloud.py}`,
`vitS/vipe_intrinsics_egoexo4d_5clips.zip` (intrinsics cross-check).

## Risks / gotchas
- **Frames locked at 49** — reduce only H/W, never frames.
- NF4 injection must be env-gated (keep 4-GPU bf16 path intact).
- Never `fuse_lora` on the NF4 base (unfused adapter only).
- `PYTHONNOUSERSITE=1` mandatory; `expandable_segments:True` to recover fragmentation.
- Cache keyed by resolution → switching res re-precomputes (clean old cache dirs if disk-tight).
- GPU contention: Stage-1 (~21 GB) and the other-session process can't share 24 GB.

## Verification (end-to-end)
1. Phase A run reaches "Precomputing … Done", unloads encoders, trains 20 steps with decreasing
   loss, peak VRAM < 24 GB, and writes a LoRA checkpoint.
2. `infer_nf4.py` with that checkpoint produces an ego video that moved toward `ego_GT` vs base.
3. Phase B: at least one Seagate cooking take fully Stage-1-processed into ≥1 valid 49-frame clip
   that the dataloader accepts (passes the `num_frames==13` assert) and precomputes without error.

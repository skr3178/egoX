# Stage B (Wan 14B diffusion) — NF4 quantization findings (Blackwell)

Local-only notes (kept outside the git repo on purpose). Dated 2026-06-22.

## NF4 QUANT SMOKE TEST — PASSED (core QLoRA de-risk)

Goal: confirm bitsandbytes 4-bit (NF4) works on Blackwell (sm_120) and that the 14B Wan
transformer fits 24 GB once quantized. Script: `egox_blackwell_local/stageB_quant_smoke.py`
(run via `run_quant_smoke.sh`, which adds a peak-VRAM sampler).

Config (the documented QLoRA recipe):
```python
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                   bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
```

### Result
| metric | value |
|---|---|
| env | `egox` — torch 2.10.0+cu128, bnb 0.49.2 |
| GPU | RTX PRO 4000 Blackwell, cap (12,0) = sm_120 |
| load time | 35 s (reads 65 GB fp32 shards, quantizes on the fly) |
| device | cuda:0 |
| **VRAM allocated** | **8.62 GB** (reserved 9.22 GB, peak 9.22 GB / 24.5 GB) |
| in_channels | **36** (stock I2V channel-rich build confirmed) |
| Params4bit tensors | 486 (quantization genuinely happened) |
| verdict | ✅ NF4 4-bit works on sm_120 |

### What it validates
1. **bitsandbytes NF4 4-bit runs on Blackwell sm_120** (bnb 0.49.2 + torch 2.10+cu128) — was
   the single biggest unknown for the QLoRA plan.
2. **14B transformer fits easily:** 65 GB fp32 → ~8.6 GB NF4 → **~15 GB headroom** for
   activations, LoRA, optimizer, and the other (offloadable) components. Matches the QLoRA budget.
3. **in_channels=36** loads correctly under quantization → EgoX's 36-ch conditioning (16 noise +
   4 mask + 16 condition) will work on the quantized base.

### Gotchas (both required)
- **`PYTHONNOUSERSITE=1`** — the `egox` env ALSO suffers the `~/.local` torch 2.11+cu130 leak;
  without the guard the import picks the wrong torch and bnb (built for 2.10+cu128) breaks.
- **Load from the LOCAL snapshot path** via `snapshot_download(REPO, local_files_only=True)` +
  `/transformer`. Do NOT set `HF_HUB_OFFLINE=1`: for a SHARDED checkpoint diffusers calls the HF
  `model_info` API to resolve the shard index, which fails under offline mode. The local-path
  approach needs no network and sidesteps it entirely.

### Scope note / what's NOT yet tested
- Used the **stock** `diffusers.WanTransformer3DModel`. EgoX's `WanTransformer3DModel_GGA`
  subclass has the same parameters (adds GGA attention logic, not weights), so VRAM ≈ same — but a
  fully faithful Stage B pass should load the `_GGA` subclass + the EgoX LoRA
  (`/media/skr/SeagateHub1/egox_checkpoints/EgoX/pytorch_lora_weights.safetensors`).
- This was a load-only smoke test (no forward/denoise). Inference adds VAE + CLIP + (precompute-
  then-unload) UMT5; training adds LoRA params + 8-bit optimizer + activations (the real 24 GB
  pressure, which is why training runs at reduced frames/res).

## Next steps
1. Deeper Stage B pass: NF4 `_GGA` subclass + EgoX LoRA (faithful inference path).
2. QLoRA single-GPU training code: bnb 4-bit `_GGA` load + peft LoRA + grad-ckpt (keeps GGA on,
   see architecture.md gotcha) + 8-bit optim + single-GPU accelerate config (replace 4gpu.yaml).
3. Wire the satya `.npz` + training-schema meta (see STAGE1_FINDINGS.md) → complete the 5-clip
   Stage B dataset.

# EgoX Model Inventory & Quantization Notes

Complete list of models the EgoX reproduction depends on, plus the 24 GB / GGUF /
quantization analysis.

## 1. Generation stack (inference + training)

All Wan subcomponents ship **inside one snapshot** — `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`
(the **480P**, **-Diffusers**, **I2V** variant — all three qualifiers matter). Loaded as
subfolders ([core/finetune/models/wan_i2v/sft_trainer.py:99-105](EgoX/core/finetune/models/wan_i2v/sft_trainer.py#L99-L105)):

| Component | Class | Role | ~Size |
|---|---|---|---|
| `transformer` | `WanTransformer3DModel` → EgoX `_GGA` subclass | 14B diffusion backbone | ~28 GB (bf16) |
| `text_encoder` | `UMT5EncoderModel` (umt5-xxl) | text → conditioning | ~11 GB |
| `image_encoder` | `CLIPVisionModel` (CLIP ViT-H) | I2V image conditioning | ~2–4 GB |
| `vae` | `AutoencoderKLWan` | latent encode/decode | ~1 GB |
| tokenizer / image_processor / scheduler | — | no weights | negligible |
| **Full Wan snapshot** | | | **~60–70 GB** |

Plus the EgoX adapter (separate download):

| Model | Repo / file | Role | ~Size |
|---|---|---|---|
| **EgoX LoRA** | `DAVIAN-Robotics/EgoX` → `pytorch_lora_weights.safetensors` | rank-256 LoRA fused onto transformer | ~0.1–1 GB |

Same base for **both training and inference** — there is no separate "training variant" of
the Wan model. What is training-specific is the LoRA config (rank 256, lora_alpha 256,
resolution 49×448×1232, the `wan-i2v` custom trainer). See [scripts/finetune.sh](EgoX/scripts/finetune.sh).

### Actual download footprint — VERIFIED 2026-06-22 (HF API, `files_metadata=True`)

The "~60–70 GB / bf16" figures above were **estimates**. The official
`Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` repo actually ships everything in **fp32**, so the real
download is larger:

| Component | Size | Why |
|---|---|---|
| `transformer` | **65.6 GB** | 14 shards — stored in **fp32** (14B params × 4 bytes ≈ 56 GB + overhead) |
| `text_encoder` (UMT5-XXL) | **22.7 GB** | 5 shards, also **fp32** |
| `image_encoder` (CLIP) | 1.26 GB | |
| `vae` | 0.51 GB | |
| **Total** | **90.1 GB** | (46 files) |

Notes / implications:
- **fp32 on disk is transit waste, not a runtime problem** — we quantize the transformer to 4-bit
  *at load time* (NF4, see decision below) and `from_pretrained(torch_dtype=...)` downcasts on load.
  But you still pay the full 90 GB download. At a ~12 MB/s link this is **~2 hours** (link-bound;
  `hf_transfer` does not help — the pipe is the bottleneck, verified by saturated parallel test).
- **No safe way to shrink the download.** The nf4 pre-quantized repo is **empty** (see HF scan below);
  all four components are needed at least once. The only theoretical saving is a **bf16 Diffusers
  mirror** of the transformer (~28 GB vs 65.6 GB), but those mirrors are unverified against EgoX's
  pinned diffusers 0.34 + custom `WanTransformer3DModel_GGA` subclass → compatibility risk on the
  exact training path. Decision: download the official fp32 repo (faithful, low-risk).
- **text_encoder is precompute-then-unload.** EgoX encodes all prompts once and caches embeddings to
  `cache_dir/prompt_embeddings/<hash>.safetensors`
  ([trainer.py:183-199](EgoX/core/finetune/trainer.py#L183-L199),
  [wan_dataset.py:168-200](EgoX/core/finetune/datasets/wan_dataset.py#L168-L200)), then
  `unload_model(text_encoder)`. So the 22.7 GB text encoder is **not resident during the training
  loop** (good for the 24 GB budget) — but it must be downloaded + loaded once for the precompute
  pass, so this is a VRAM/runtime win, not a download saving.
- Download target / cache: default HF cache `~/.cache/huggingface/hub` (home partition, 321 GB free).
  Pull with `HF_HUB_ENABLE_HF_TRANSFER=1 hf download Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`.

#### Integrity verification after start/stop downloads — VERIFIED 2026-06-22 (✅ 33/33 ok)

This snapshot was pulled across **many interrupted (start/stop) downloads**, so it was
integrity-checked. The download passed cleanly: `33 ok | 0 hash/size-bad | 0 missing |
0 parse-bad | 0 incomplete` → **all bytes intact** (rev `b184e23`).

How to re-verify any HF snapshot (script: [verify_wan_integrity.py](verify_wan_integrity.py)):
- **Why it's trustworthy:** HF stores each LFS file under a blob name that **is** its SHA256, and
  the Hub API (`model_info(..., files_metadata=True)`) exposes the expected `lfs.sha256` + byte
  size per file. So recomputing the on-disk SHA256 and comparing catches a single flipped/truncated
  byte — stronger than size-only checks.
- **Three layers:** (1) no leftover `*.incomplete` files; (2) on-disk SHA256 == Hub `lfs.sha256`
  + size match, for every LFS file; (3) every `*.safetensors` parses via `safe_open` (catches
  silent truncation that still matches size).
- **Run:** `python verify_wan_integrity.py` (uses the egox env; hashing ~90 GB takes a few min).
  Exit 0 = intact; exit 1 = lists the bad/missing files. **Repair = re-run `hf download`** — it
  re-fetches only the bad/missing files, not the whole 90 GB.
- **Gotcha noted:** don't build a "wait for download then verify" watcher whose `pgrep -f
  "hf download…"` pattern appears in *its own* command line — it self-matches and loops forever.
  Match a more specific string or use the harness background-task notification instead.

## 2. Ego-prior preprocessing models

**Version 1 — shi3z standalone** ([EgoX-shi3z/generate_ego_prior.py](EgoX-shi3z/generate_ego_prior.py)),
runnable now, approximate:

| Model | Role | ~Size |
|---|---|---|
| Depth Anything V2 Large (`depth-anything/Depth-Anything-V2-Large-hf`) | monocular depth | ~1.3 GB |
| MiDaS DPT-Large (fallback) | monocular depth | ~1.4 GB |

**Version 2 — ViPE renderer** (faithful; submodule `kdh8156/EgoX-EgoPriorRenderer`, not yet
checked out). Install footprint ~15–20 GB; auto-downloads model weights on first run:

| Model | Role | ~Size |
|---|---|---|
| MoGE2 (`lyra` pipeline) | metric depth | ~1–2 GB |
| UniDepthV2 (`default` pipeline) | metric depth | ~0.5–1.5 GB |
| Video Depth Anything (VDA) | temporal depth consistency | ~1–2 GB |
| ViPE internal SLAM/pose models | camera poses | ~0.5–1 GB |

## 3. Captioning (training-data prep only)

| Model | How | Needed? |
|---|---|---|
| GPT-4o-mini via OpenAI API ([caption.py](EgoX/caption.py)) | API call, needs key | ❌ Not needed — captions already in `meta_train.json`. Only for re-captioning new/in-the-wild clips. |

## 4. Notes — depth/ViPE models **auto-download** from HuggingFace / torch.hub on first
run; you don't fetch them manually. The only big manual download is the Wan snapshot + LoRA.

---

# Quantization, GGUF, and the "inpainting variant" question

**Verified against the code — correcting a common misconception.**

## There is NO separate "inpainting variant" — stock I2V *is* the channel-rich build

EgoX loads the **stock** transformer straight from the Wan snapshot
([infer.py:60-65](EgoX/infer.py#L60-L65)):

```python
transformer_path = os.path.join(model_path, 'transformer')   # Wan2.1-I2V-14B-480P-Diffusers/transformer
transformer = WanTransformer3DModel_GGA.from_pretrained(transformer_path, ...)
```

Stock **Wan2.1-I2V** already declares **`in_channels=36`** in its `config.json` =
**16 noisy latent + 4 mask + 16 conditioning latent**. The I2V conditioning channels are
built into the I2V model. (EgoX's custom class defaults `in_channels=16` at
[custom_transformer.py:494](EgoX/core/finetune/models/wan_i2v/custom_transformer.py#L494),
but the checkpoint config overrides it to 36 on load.)

EgoX fills exactly those channels ([sft_trainer.py:193](EgoX/core/finetune/models/wan_i2v/sft_trainer.py#L193)):

```python
return latents, torch.concat([mask_lat_size, latent_condition], dim=1), ...
#                              4 ch mask      + 16 ch condition  → alongside 16 ch noise = 36
```

**Conclusion: EgoX uses stock `Wan2.1-I2V-14B-480P` unchanged.** In the Wan family, "I2V"
*is* the variant with the extra mask+condition channels — there is no distinct
"Wan2.1-I2V inpainting" checkpoint to find. (VACE is a separate editing model — different thing.)

Implications:
- EgoX does **not** ship a modified checkpoint, and the LoRA does **not** add input channels —
  it adapts existing weights of the 36-channel I2V transformer.
- The **city96 GGUF of `Wan2.1-I2V-14B-480P` is channel-correct** — it *is* the right base.
  The worry "the GGUF is base I2V, not the inpainting build" rests on a false split.

## The real catch with GGUF: stack incompatibility, not channels

- GGUF lives in the **ComfyUI ecosystem**. EgoX's [infer.py](EgoX/infer.py) is **HF Diffusers**
  (`from_pretrained` + `load_lora_weights` + `fuse_lora`), which does not load GGUF.
- Using a GGUF base on 24 GB means **reimplementing EgoX's custom conditioning** in ComfyUI:
  `WanWidthConcatImageToVideoPipeline`, GGA attention, the `[mask | exo | ego_prior]`
  width-concat, and `--use_GGA` / `cos_sim_scaling_factor` logic. That is the hard part.

## Recommended 24 GB route: in-framework quantization (not GGUF)

Keep EgoX's diffusers pipeline (all custom conditioning keeps working) and quantize the
transformer in-framework:

| Technique | 28 GB transformer → | Note |
|---|---|---|
| bitsandbytes 4-bit (QLoRA-style) | ~7–9 GB | some quality loss |
| optimum-quanto fp8 | ~14 GB | minimal loss, needs fp8 kernels |
| + `enable_model_cpu_offload` (encoders) | streams GPU↔CPU | needs 64 GB+ RAM |
| + VAE tiling | cuts decode spike | nearly free |

This preserves the released EgoX weights (no retraining) and runs the real model on 24 GB —
slow, but faithful. See also the smaller-backbone retrain options in [dataset.md](dataset.md) /
the repro discussion (Wan2.2-TI2V-5B, etc.) if a native-24 GB *trainable* model is the goal.

## Off-the-shelf pre-quantized 480P checkpoints (HF scan) — VERIFIED 2026-06-22

**Verdict: none of the candidates work, and you don't need them.** I inspected each repo's
actual file list against our requirements (stock base + **Diffusers format** + usable for
QLoRA). Results:

| Repo | Claimed | Verified verdict |
|---|---|---|
| `Meatfucker/Wan2.1-I2V-14B-480P-nf4-bnb` | bnb nf4, ~7–9 GB | ❌ **Empty repo** — only `.gitattributes`, no weights |
| `PJMixers-Images/wan2.1_i2v_480p_720p_14B_fp8_e4m3fn` | fp8, ~14 GB | ❌ Single **ComfyUI** `.safetensors` (transformer-only), **not Diffusers**; fp8 ≠ QLoRA format |
| `InsecureErasure/…-480P-…-NVFP4` | distilled + NVFP4, ~7 GB | ❌ **Distilled** — alters sampling regime; can't cleanly train EgoX's LoRA on it |
| `magespace/…-480P-Lightning-Diffusers` | distilled, bf16 | ❌ ~28 GB (doesn't fit 24 GB) **and** distilled |
| `fal/…-480P-FlashPack` | fast-load, ~28 GB | ❌ no quantization (just fast loading) |

### Key insight: QLoRA does NOT need a pre-quantized checkpoint

The standard, robust path is to download the **stock fp16 Diffusers repo** and quantize
**at load time** — full control over the quant config, works with EgoX's `_GGA` subclass:

```python
from diffusers import BitsAndBytesConfig
qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
transformer = WanTransformer3DModel_GGA.from_pretrained(transformer_path, quantization_config=qcfg, ...)
```

diffusers quantizes weights to NF4 as it streams them to the GPU. A pre-quantized repo would
only save a one-time ~few-minute quantization step — not worth the format/compatibility risk.
**Decision: use the stock `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` + load-time NF4.**
(Verified: bnb NF4 4-bit runs on this Blackwell GPU, cap 12.0, bnb 0.49.2, torch 2.10+cu128.)

**Traps (still relevant if scanning HF again):**
- `AaronHuangWei/…INT8FakeQuant` / `NVFP4FakeQuant` / `MXFP4FakeQuant` — **"FakeQuant" =
  simulated in fp16 for accuracy studies, NO real memory saving.** Do not use to fit 24 GB.
- `fal/…-480P-FlashPack` speeds *loading*, not memory (still ~28 GB).
- All `fp16` repos (denisbalon, gaga2210, IntervitensInc, wavespeed…) = bf16 size, no saving.
- 720P repos = wrong resolution (EgoX trains 49×448×1232).

**fuse_lora caveat for pre-quantized bases:** EgoX does `load_lora_weights` + `fuse_lora`
([infer.py](EgoX/infer.py)). You **cannot cleanly fuse a bf16 LoRA into already-quantized
(nf4/fp8) weights** — fusion needs the dequantized base. With a pre-quantized base you must
run EgoX's LoRA **unfused as a PEFT adapter** (small infer.py change). The fully faithful
alternative: load stock bf16 → `fuse_lora` → **then** quantize the fused model at load.

**Distillation caveat:** distilled bases (lightx2v / Lightning / NVFP4-distill) change the
sampling regime (4-step, no CFG, custom scheduler). EgoX's GGA bias + width-concat were
trained for the full multi-step sampler — stacking works mechanically but quality is
**unvalidated**, and you'd also swap the scheduler/`guidance_scale` in infer.py.

## Faithful 24 GB path with ZERO quality loss: block / CPU offload

Source: [Wan-Video/Wan2.1 issue #241](https://github.com/Wan-Video/Wan2.1/issues/241).
Streams the transformer's DiT blocks CPU↔GPU during the forward pass — only `N` blocks
live on the GPU at once. Native **bf16, no quantization → no quality loss**; cost is time.

| `offload_blocks_num` | Peak VRAM | Speed |
|---|---|---|
| 1 | ~11 GB (RTX 3090) | slowest |
| 7 | ~20 GB | ~3.5 h for 480P / 7 steps |

**That issue patches the official `generate.py` — NOT usable in EgoX**, which is HF Diffusers
(`WanWidthConcatImageToVideoPipeline` + `WanTransformer3DModel_GGA`). Diffusers ships the
same mechanism built-in — add to [infer.py](EgoX/infer.py) after building `pipe`:

```python
pipe.enable_sequential_cpu_offload()   # per-submodule streaming ≈ offload_blocks_num=1 → lowest VRAM, slow
# pipe.enable_model_cpu_offload()      # per-module; won't help alone (28 GB transformer won't fit at once)
pipe.vae.enable_tiling()               # caps the decode spike
```

GGA bias + width-concat run *inside* the block forwards, so accelerate's module-boundary
hooks don't interfere — works at native precision.

## Decision matrix (24 GB Blackwell, inference with released LoRA)

| Path | Quality | VRAM | Speed |
|---|---|---|---|
| Sequential CPU offload (issue #241 → Diffusers) | ✅ identical (bf16) | ~11–20 GB | ❌ hours/clip |
| In-framework / pre-quantized **fp8** | ~near-identical | ~14 GB | ✅ normal |
| **nf4 / NVFP4** | ⚠️ some loss | ~7–9 GB | ✅ fast |
| Distilled (lightx2v / Lightning) | ⚠️ unvalidated w/ GGA | ~7–14 GB | ✅✅ 4-step |

**Sweet spot:** fp8 transformer + `enable_model_cpu_offload` for the text/image encoders —
keeps the quantized transformer resident, offloads only the encoders. Faithful-ish, not
painfully slow.

## Status of this fact in upstream docs

- **EgoX repo**: implicit in code (loads stock I2V transformer, builds 36-ch input) but
  **not documented** — README only says "download Wan2.1-I2V-14B-480P-Diffusers."
- Recorded here because it is not stated anywhere else.

---

# DECISION (2026-06-22): Wan2.1-I2V-14B + 4-bit QLoRA

**Chosen path for the 24 GB repro: train EgoX's actual LoRA on a 4-bit-quantized
Wan2.1-I2V-14B-480P-Diffusers.** Rationale = minimum resistance + maximum faithfulness:
keeps the exact model, method, conditioning code, and dataset; no backbone port.

Why the alternatives were rejected:
- **TI2V-5B**: near-total rewrite — different VAE (z_dim 16→48, spatial 8→16), no CLIP path,
  no 36-ch mask+concat input, needs diffusers 0.35+ (breaks EgoX's 0.34 subclasses).
- **T2V-1.3B + convert**: must train I2V conditioning (channels + CLIP) from scratch — more
  than a LoRA, and weaker base.

### Backbone-fit summary (verified)

| Option | Size | Fits 24 GB? | Faithful? | Note |
|---|---|---|---|---|
| Wan2.1-I2V-14B **fp16** (original) | 14B | ❌ ~28 GB weights | ✅ reference | won't load |
| Wan2.1-I2V-14B **fp8** | 14B | ⚠️ ~14 GB | ✅ near-exact | inference yes; training too tight at full res |
| **Wan2.1-I2V-14B 4-bit/QLoRA** ← chosen | 14B | ✅ ~7–8 GB weights | ✅ exact method | inference easy; **training fits only at reduced res/frames** |
| T2V-1.3B + I2V conversion | 1.3B | ✅ ~2.6 GB | ⚠️ from scratch | trains at higher res than 14B-QLoRA |
| Wan2.2-TI2V-5B | 5B | ✅ ~10 GB | ❌ rewrite | different VAE/no CLIP/diffusers 0.35+ |

### Known constraints / risks for this path
- **Activations, not weights, are the 24 GB limit for training.** 14B hidden dim ≈ 5120; at the
  paper's 49×448×1232 (~112k tokens) activations exceed 24 GB even with grad checkpointing.
  → must reduce frames/resolution for training (e.g. ~13 frames, ~half res). "Faithful 14B" thus
  becomes "14B at reduced training resolution."
- **QLoRA is not built into the repo.** Trainer uses DeepSpeed + bf16. Need to add: bitsandbytes
  4-bit base load on the custom `WanTransformer3DModel_GGA`, peft LoRA on the quantized model,
  gradient checkpointing, 8-bit optimizer, single-GPU accelerate config (replace 4gpu.yaml).
- **Blackwell GPU (RTX PRO 4000, sm_121, cu128)** — bitsandbytes 4-bit needs a recent build with
  Blackwell support; torch must be cu128. See sibling hands project for cu128/Blackwell gotchas.
- diffusers stays at 0.34.0 (EgoX pin) — quantized loading of the custom transformer subclass
  must go through the bitsandbytes path that 0.34 supports, or manual Linear4bit replacement.

### Staged execution plan (de-risk before scaling)
0. **Quantized inference** of real EgoX on example data — validates env + Blackwell bnb + 24 GB
   fit + output quality. No dataset needed. (Uses the same quantization machinery as training.)
1. **Data pipeline on 7 local cooking takes** — ViPE → depth → ego-prior → width-concat GT.
2. **Backbone smoke-test** — QLoRA train on those few clips; confirm fit + loss drop; get real
   memory numbers (decides final training resolution).
3. **Scale** data + train.

#### Stage 0 readiness check — VERIFIED 2026-06-22 (it really is runnable with NO data prep)

Stage 0 = "quantized inference of the released model" is **our invented validation step** (no
equivalent in the paper's Stage A/B architecture — see [architecture.md](architecture.md)). The
worry "doesn't inference still need a rendered ego-prior?" is answered: **the repo ships
pre-rendered examples**, and `infer.py` *loads* the prior — it does not render it.

- `infer.py` reads `meta['ego_prior_path']` ([infer.py:43](EgoX/infer.py#L43)) from a
  `--meta_data_file`; it consumes a **pre-rendered** `ego_Prior.mp4`, never generates one. So
  Stage A / ViPE / the renderer are **not** needed for Stage 0.
- Shipped example data (no download, no preprocessing):
  - `EgoX/example/egoexo4D/` — `meta.json` + 3 clips (basketball, dance, cooking); each has
    `exo.mp4`, `ego_GT.mp4` (for comparison), and pre-rendered `ego_Prior.mp4`.
  - `EgoX/example/in_the_wild/` — `meta.json` + 4 clips (tabletennis, ironman, hulk, joker);
    each has `exo.mp4` + `ego_Prior.mp4`.

**All four Stage 0 inputs present:**

| Need | Status |
|---|---|
| Env (cu128, bnb 0.49.2, diffusers 0.34, peft) | ✅ built + verified |
| Wan base (90 GB) | ✅ downloaded + SHA256-verified (see §1 integrity) |
| EgoX LoRA | ✅ `/media/skr/SeagateHub1/egox_checkpoints/EgoX/pytorch_lora_weights.safetensors` |
| Example inputs (exo + pre-rendered ego_prior + meta) | ✅ `EgoX/example/` |

**The only Stage-0 work is a code change to `infer.py`:** stock `infer.py` loads the 14B
transformer in **bf16 (~28 GB) → won't fit 24 GB**. Add the NF4 `BitsAndBytesConfig` load.
**fuse_lora caveat** (from §"Off-the-shelf pre-quantized" above): a bf16 LoRA cannot be cleanly
`fuse_lora`'d into already-quantized weights → either run the LoRA **unfused as a PEFT adapter**,
or load bf16 → `fuse_lora` → **then** quantize the fused model.

---

# Alternatives — decision summary (24 GB GPU)

Three independent levers. They fix **different** problems and compose:

| Lever | Fixes | Reduces VRAM? | Reduces wall-clock? | Retrain? | Keeps real EgoX weights? |
|---|---|---|---|---|---|
| **Quantization** (4-bit / fp8, in-framework) | memory wall | ✅ 28 GB → 7–14 GB | ➖ a little | ❌ no | ✅ yes |
| **Distillation** (step distill: CausVid / DMD2 / Lightning LoRA) | speed | ❌ no (still full 14B) | ✅✅ 50 → 4–8 steps | ❌ no | ✅ yes |
| **Smaller backbone** (Wan2.2-TI2V-5B) | both, natively | ✅ | ✅ | ✅ yes | ❌ new model |

## Key facts

- **Quantization is mandatory** to fit the 14B on 24 GB (base alone is ~28 GB bf16, exceeds
  24 GB before any activation). Use in-framework (bitsandbytes 4-bit or optimum-quanto fp8),
  **not GGUF** — GGUF is ComfyUI-only and would require reimplementing EgoX's width-concat/GGA
  conditioning. See "The real catch with GGUF" above.
- **Distillation does NOT solve memory.** A step-distilled model is still the full 14B (~28 GB).
  It only cuts sampling steps. It must be **combined with** quantization to fit 24 GB.
- **No pre-distilled EgoX exists** (CVPR 2026, too new). You would stack a *generic* Wan
  distillation LoRA that matches `Wan2.1-I2V-14B-480P` specifically.

## Best INFERENCE setup on 24 GB (no retraining) — stack all three layers

```
quantized Wan2.1-I2V-14B base   (4-bit/fp8 → fits 24 GB)
  + EgoX LoRA                    (the real method; fuse_lora)
  + Wan distillation LoRA        (4–8 step sampling, fast)
```

EgoX's [infer.py](EgoX/infer.py) already does `load_lora_weights` + `fuse_lora`, so adding a
second (distill) LoRA is mechanically easy. Catches:
1. **Scheduler coupling** — distillation needs its own sampler/timestep schedule; EgoX uses
   `FlowMatchEulerDiscreteScheduler`. Swap/configure to the distill recipe.
2. **LoRA-stacking quality** — two fused LoRAs can interfere; tune `lora_scale` per adapter,
   and watch interaction with EgoX's `cos_sim_scaling_factor` / GGA under few-step sampling.

## Best TRAINING path on 24 GB (a model you control) — Family B

Distillation is **orthogonal** to training (it's an inference accelerator, not a training
shortcut). For a trainable repro use **Wan2.2-TI2V-5B** (smallest natively-I2V Wan; same
diffusers ecosystem). Optionally distill your *own* student **after** it works. Scope + porting
detail in [dataset.md](dataset.md).

## Recommended order

1. **Quantize the real EgoX** (Family A) → validate output quality in hours, zero dataset work.
2. Optionally **add a distill LoRA** → fast faithful inference on 24 GB.
3. **Only if quality is worth it** → port + retrain on **Wan2.2-TI2V-5B** with a cooking-only
   subset (7 takes already local).

# EgoX Architecture

Egocentric (first-person) video generation from a **single exocentric (third-person)
video** + a target ego camera trajectory. The method adapts a pretrained 14B video
diffusion model with a lightweight **LoRA** plus two conditioning tricks
(width-concat for implicit cross-view warping, channel-concat for pixel-aligned
guidance) and a training-time **Geometry-Guided Self-Attention (GGA)** bias.

Grounded in the released code:
[infer.py](EgoX/infer.py),
[sft_trainer.py](EgoX/core/finetune/models/wan_i2v/sft_trainer.py),
[custom_transformer.py](EgoX/core/finetune/models/wan_i2v/custom_transformer.py),
[wan.py](EgoX/core/inference/wan.py). Paper figures: `Fig 3.png` (pipeline),
`Fig 4.png` (GGA), `Fig 8.png` (in-the-wild ego cam), `Fig 9.png` (depth align).

---

## 1. Overall pipeline (ASCII)

```
=== STAGE A: OFFLINE PRE-PROCESSING ==========================================
    (separate repo: EgoX-EgoPriorRenderer -- NOT in this codebase)

  exo video (3rd person, fixed cam)          ego trajectory phi
         |                                    (Ego-Exo4D GT, or
         v                                     Viser hand-set, Fig 8)
  +------------------------+                          |
  | MoGe-2  (sharp depth)  |--+                       |
  +------------------------+  |                       |
  +------------------------+  +--> ViPE affine align  |
  | Video-Depth-Anything   |--+    (Fig 9)            |
  | (temporally smooth)    |          |               |
  +------------------------+          v               v
                              3D point cloud --> PyTorch3D rasterize
                              (lift depth to 3D)     from ego pose
                                                          |
                                                          v
                                       ego_Prior.mp4  +  depth_maps/*.npy
                                                          |
=== STAGE B: THIS REPO (EgoX) ============================================|====
                                                                          |
  exo.mp4 -------> [VAE encoder (frozen)] --> x0  (clean exo latent,16ch) |
  ego_Prior.mp4 -> [VAE encoder (frozen)] --> p0  (ego-prior latent,16ch) |
  noise ~ N(0,1) ---------------------------> zt  (noisy ego latent,16ch) |
         |                                                                |
         v                                                                |
  (W) WIDTH-CONCAT :  [ x0 clean | zt noisy ]  along width                |
  (C) CHANNEL-CONCAT: stack [ mask m1|m0 (4ch) + cond x0|p0 (16ch) ]      |
         |                                                                |
         v                                                                |
  36-channel DiT input  =  16 noise + 4 mask + 16 condition               |
         |                                                                |
         |   prompt (GPT-4o) --> [UMT5-XXL text enc (frozen)] --+         |
         |   exo frame0 -------> [CLIP-ViT-H img enc (frozen)] -+         |
         |   depth+cam params -> 3D ray dirs -> GGA cos-sim ----+----+----+
         v                                                      |    (Fig 4)
  +-------------------------------------------------+           |  ego<-exo
  |  WanTransformer3DModel_GGA   (DiT block x 40)   |<----------+
  |    1. Geometry-Guided Self-Attention  <---- LoRA rank 256 (TRAINABLE)
  |    2. Cross-Attention (text + image)                                  |
  |    3. Feed-Forward                                                    |
  +-------------------------------------------------+                     |
         |  50-step Flow-Match Euler denoise (exo half re-frozen / step)  |
         v                                                                |
  latent --> split --> [VAE decoder (frozen)] --> EGO VIDEO (ego half) <--+
==============================================================================

  Legend:  (W) width-wise concat   (C) channel-wise concat   | = side-by-side
           TRAINABLE = LoRA only      frozen = VAE, UMT5, CLIP, base DiT
```

### Tensor shapes (default config: 49 frames, 448×1232 width-concat)

| Stage | Pixel | Latent (÷8 spatial, ÷4 temporal) |
|---|---|---|
| exo | 448×784 | 56×98 |
| ego (prior / target) | 448×448 | 56×56 |
| width-concat (exo‖ego) | 448×1232 | 56×154 |
| frames | 49 | 13 |
| DiT token sequence | — | 13·28·77 = **28028** (after 1×2×2 patch) |
| transformer input channels | — | **36** = 16 noise + 4 mask + 16 condition |

Hard-coded in [infer.py:103](EgoX/infer.py#L103) (`C,F,H,W = 16,13,56,154`) and
[sft_trainer.py:884](EgoX/core/finetune/models/wan_i2v/sft_trainer.py#L884)
(`exo_width, ego_width = 784, 448`).

### Stage B — step-by-step (data flow)

```
                 STAGE B  —  EgoX generation (Wan2.1-I2V-14B + rank-256 LoRA)
 ============================================================================================

 INPUTS (from Stage A + dataset, per clip @ 49 frames):
   exo.mp4 (448x784)   ego_Prior.mp4 (448x448)   ego_GT.mp4*   prompt   depth+intrinsics+cams
        |                    |                       |            |              |
        |                    |                       |            |              |   (*train only)
        v                    v                       v            |              |
 ┌─[1] VAE ENCODE (frozen) ──────────────────────────────────┐   |              |
 |   exo      -> x0   clean exo latent      (16ch, 13x56x98)  |   |              |
 |   ego_prior-> p0   ego-prior latent      (16ch, 13x56x56)  |   |              |
 |   ego_GT   -> z0   then add noise -> zt  (16ch, noisy)     |   |              |
 └────────────────────────────────────────────────────────────┘  |              |
        |        |          |                                      |              |
        v        v          v                                      |              |
 ┌─[2] BUILD 36-CHANNEL DiT INPUT ──────────────────────────────┐ |              |
 |  (W) width-concat:  [ x0 | zt ]  side-by-side  -> 13x56x154  | |              |
 |  (C) channel-concat: 16 noise + 4 mask(m1|m0) + 16 cond(p0)  | |              |
 |                      = 36 channels                           | |              |
 └──────────────────────────────────────────────────────────────┘ |              |
        |                                                           v              v
        |                                          ┌─[3] CONDITION ENCODERS (frozen)──────────┐
        |                                          |  prompt -> UMT5-XXL  -> text embeds       |
        |                                          |           (precomputed+cached, then       |
        |                                          |            text encoder UNLOADED)          |
        |                                          |  exo frame -> CLIP-ViT-H -> image embeds   |
        |                                          └────────────────────────────────────────────┘
        |                                                           |
        |                  ┌─[4] GGA GEOMETRY ─────────────────────┐|
        |                  | depth + intrinsics(.npz) + cam/ego ext ||
        |                  |  -> 3D ray dirs -> cos-sim bias (4x16x16)|
        |                  └────────────────────────────────────────┘|
        v                                                            v
 ┌─[5] WanTransformer3DModel_GGA   (DiT block x 40)  [NF4 4-bit base + LoRA] ───────────────┐
 |     per block:                                                                            |
 |       5a. Geometry-Guided Self-Attention  <-- GGA bias   <== LoRA rank256 (TRAINABLE)     |
 |       5b. Cross-Attention (text embeds + image embeds)                                    |
 |       5c. Feed-Forward                                                                    |
 └──────────────────────────────────────────────────────────────────────────────────────────┘
        |
        v
 ┌─[6] FLOW-MATCH DENOISE (FlowMatchEulerDiscrete) ──────────────────────────────────────────┐
 |   TRAIN:  predict velocity; MSE loss on the EGO HALF ONLY; 1 step/sample; backprop -> LoRA  |
 |   INFER:  50-step loop; exo half re-frozen every step (noise_pred[...exo...]=0)             |
 └────────────────────────────────────────────────────────────────────────────────────────────┘
        |
        v
 ┌─[7] SPLIT + VAE DECODE (frozen) ──────────────┐
 |   take ego half of latent -> VAE decode        |
 |   -> EGO VIDEO  (448x448, 49 frames)           |
 └──────────────────────────────────────────────────┘

 Legend:  frozen = VAE, UMT5, CLIP, base DiT weights      TRAINABLE = LoRA only
          NF4 4-bit = the quantization we validated (14B -> ~8.6 GB on the 24 GB GPU)
          (W) width-concat   (C) channel-concat   |=side-by-side
```

---

## 2. Models used, and **why**

All generation-stack weights ship inside one snapshot:
`Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` (loaded as subfolders in
[sft_trainer.py:710-722](EgoX/core/finetune/models/wan_i2v/sft_trainer.py#L710)).
Only the LoRA is trained; everything else is frozen.

### 2.1 Generation stack (inference + training)

| Model | Class / repo | Role in EgoX | Why this model |
|---|---|---|---|
| **Wan2.1-I2V-14B-480P** DiT | `WanTransformer3DModel_GGA` (EgoX subclass of Wan DiT), [custom_transformer.py:445](EgoX/core/finetune/models/wan_i2v/custom_transformer.py#L445) | The denoising backbone — 40 DiT blocks, generates the ego video | **I2V** variant already exposes a **36-channel** input (16 noise + 4 mask + 16 condition). EgoX needs exactly those extra mask+condition channels for its channel-concat guidance, so no architecture surgery is required — the conditioning slot is built in. 14B gives the capacity for photorealistic, temporally coherent video; 480P keeps the latent sequence tractable. Frozen + LoRA = cheap adaptation that preserves the base model's video prior. |
| **AutoencoderKLWan** (VAE) | `AutoencoderKLWan`, [sft_trainer.py:716](EgoX/core/finetune/models/wan_i2v/sft_trainer.py#L716) | Encode exo/ego-prior video → latents; decode final latent → pixels. ÷8 spatial, ÷4 temporal, 16-dim latent | Must be the **matching** VAE the DiT was trained on (latent statistics `latents_mean/std`, `z_dim=16`, downsample factors). Operating in this VAE's latent space is what makes the 28k-token sequence and the 36-channel layout feasible. Frozen — it is a fixed codec, not part of the learned task. |
| **UMT5-XXL** text encoder | `UMT5EncoderModel`, [sft_trainer.py:712](EgoX/core/finetune/models/wan_i2v/sft_trainer.py#L712) | Encodes the dual-view text caption → cross-attention conditioning | The encoder Wan2.1 was pretrained with; swapping it would break the cross-attention distribution. Multilingual UMT5-XXL gives strong text grounding for the scene description that steers both views. Frozen. |
| **CLIP-ViT-H** image encoder | `CLIPVisionModel`, [sft_trainer.py:720](EgoX/core/finetune/models/wan_i2v/sft_trainer.py#L720) | Encodes a reference frame → image-embed for the I2V image cross-attention (`add_k/v_proj`, [custom_transformer.py:88](EgoX/core/finetune/models/wan_i2v/custom_transformer.py#L88)) | Inherited from Wan's **I2V** image-conditioning path. EgoX reuses it to inject appearance/identity cues; it decodes the shared exo latent back to a frame and CLIP-encodes it ([encode_image](EgoX/core/finetune/models/wan_i2v/sft_trainer.py#L253)). Keeping it satisfies the base I2V interface; frozen. |
| **FlowMatchEulerDiscreteScheduler** | [sft_trainer.py:718](EgoX/core/finetune/models/wan_i2v/sft_trainer.py#L718) | Flow-matching noise schedule; 50-step denoise at inference | The scheduler Wan2.1 trains/samples under (flow-matching `sigmas`, `num_train_timesteps`). The training loss (`noise − latent` target, [sft_trainer.py:918](EgoX/core/finetune/models/wan_i2v/sft_trainer.py#L918)) is defined against it, so it is not interchangeable. |
| **EgoX LoRA** | `DAVIAN-Robotics/EgoX` → `pytorch_lora_weights.safetensors`; loaded + fused [infer.py:72-75](EgoX/infer.py#L72) | Rank-256 / α-256 adapter on the DiT — the **only** trained weights | The whole method premise: adapt a frozen video model to the exo→ego task with ~0.1–1 GB of weights instead of retraining 14B. Rank 256 is large for a LoRA (the task is a substantial viewpoint transform), but still a fraction of full fine-tuning. |

### 2.2 Ego-prior pre-processing models (separate repo `EgoX-EgoPriorRenderer`)

These produce `ego_Prior.mp4` + `depth_maps/` **before** EgoX runs. Not loaded by
this repo — only their outputs are consumed (and re-used in [infer.py:107-230](EgoX/infer.py#L107)
to build the GGA ray tensors).

| Model | Role | Why |
|---|---|---|
| **MoGe-2** | Per-frame metric monocular depth | Sharp, geometrically accurate depth for lifting exo pixels to 3D — but temporally jittery on its own. |
| **Video Depth Anything (VDA)** | Temporally smooth video depth | Stabilizes depth across frames (affine-invariant) so the rendered point cloud doesn't flicker (Fig 9). |
| **ViPE** | Momentum affine alignment of the two depth sources + camera poses | Fuses MoGe-2's sharpness with VDA's temporal stability (Eq. 1); without it the ego render has "unstable, unexpected camera movements" (Fig 9). |
| **PyTorch3D** point renderer | Rasterize the aligned point cloud from the ego pose → ego-prior video | Gives a **pixel-aligned but hole-filled** guidance image of what the ego view roughly looks like (Eq. 2) — the `p0` channel-condition. |
| **Viser** (in-the-wild only) | Interactively hand-set the ego camera extrinsics | Ego-Exo4D provides GT ego poses; for in-the-wild clips with no GT, the user defines the trajectory manually (Fig 8). |

### 2.3 Captioning (training-data prep only)

| Model | Role | Why / needed? |
|---|---|---|
| **GPT-4o(-mini)** via OpenAI API | Writes the dual-view (`[Exo view] … [Ego view] …`) text prompt, [caption.py](EgoX/caption.py) | Supplies the cross-attention text conditioning. **Not needed for inference** on provided examples — captions are already in `meta.json`; only required to caption new/in-the-wild clips. |

---

## 3. The three conditioning mechanisms (what the LoRA actually learns to use)

1. **Width-wise concatenation (W)** — clean exo latent `x0` is placed *beside* the
   noisy ego latent `zt` along the width axis and **held fixed every denoise step**
   (`noise_pred[..., :-ego_latent_width] = 0`,
   [sft_trainer.py:643](EgoX/core/finetune/models/wan_i2v/sft_trainer.py#L643)).
   *Why:* lets self-attention freely warp exo content into the ego view (implicit
   cross-view correspondence) without an explicit flow network. This is the
   "IC-LoRA" shared-latent trick in code
   ([_setup_ic_lora_latent](EgoX/core/finetune/models/wan_i2v/sft_trainer.py#L219)).

2. **Channel-wise concatenation (C)** — the ego-prior latent `p0` plus a binary
   mask `(m¹ exo=1, m⁰ ego=0)` is concatenated onto the input channels →
   16 noise + 4 mask + 16 condition = **36ch**
   ([sft_trainer.py:904](EgoX/core/finetune/models/wan_i2v/sft_trainer.py#L904)).
   *Why:* injects **pixel-aligned** geometric guidance (the rendered ego prior)
   directly where the model denoises, using the I2V model's existing extra channels.

3. **Geometry-Guided Self-Attention (GGA)** — an additive log-bias on the
   self-attention logits where **ego queries attend to exo keys** weighted by the
   **cosine similarity of their 3D ray directions** (Fig 4),
   [custom_transformer.py:103-129](EgoX/core/finetune/models/wan_i2v/custom_transformer.py#L103).
   Directions are downsampled `4×16×16` to match the VAE patch grid; sharpened by
   `cos_sim_scaling_factor` (1.0 train, **3.0** infer).
   *Why:* forces cross-view attention to be **geometrically consistent** — an ego
   pixel pulls from exo pixels that look in the same real-world direction, not just
   visually similar ones.

> ⚠️ **Gotcha:** GGA is gated by the `do_kv_cache` flag
> ([custom_transformer.py:103](EgoX/core/finetune/models/wan_i2v/custom_transformer.py#L103)).
> Training relies on gradient checkpointing (`gradient_checkpointing=True`,
> [args.py:68](EgoX/core/finetune/schemas/args.py#L68)) whose code path does **not**
> forward `do_kv_cache`, so it defaults to `False` and GGA stays active. Disabling
> grad-checkpointing would silently turn GGA off during training.

---

## 4. Trainable vs. frozen summary

| Component | State | Param share |
|---|---|---|
| EgoX LoRA (rank 256 on DiT) | 🔥 **trained** | ~0.1–1 GB |
| Wan2.1-I2V-14B DiT base | ❄ frozen | ~28 GB (bf16) |
| AutoencoderKLWan (VAE) | ❄ frozen | ~1 GB |
| UMT5-XXL text encoder | ❄ frozen | ~11 GB |
| CLIP-ViT-H image encoder | ❄ frozen | ~2–4 GB |

Training: LoRA rank 256, batch 1, bf16, 49×448×1232, flow-matching MSE on the **ego
half only** ([sft_trainer.py:936-939](EgoX/core/finetune/models/wan_i2v/sft_trainer.py#L936)).
Paper trains ~1 day on 8×H200 over ~3,600 Ego-Exo4D clips.

See [models.md](models.md) for the 24 GB quantization/QLoRA reproduction analysis and
[pipeline.md](pipeline.md) for the filled-in paper pipeline.

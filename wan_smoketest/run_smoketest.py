#!/usr/bin/env python
"""
Stock Wan2.1-I2V-14B-480P smoke test on a single 24 GB GPU.

Runs the DOWNLOADED diffusers snapshot (no EgoX conditioning) to validate:
  env + Blackwell bnb + 24 GB fit + quantization quality
i.e. Stage 0 of the repro plan (models.md).

Modes (same image/seed/prompt → directly comparable):
  full  : bf16 transformer + enable_model_cpu_offload          (reference quality)
  int8  : bitsandbytes 8-bit transformer + offload             (safer-quality quant)
  nf4   : bitsandbytes 4-bit nf4 transformer + offload         (smallest; tests diffusers #11006 garbage bug)

Usage:
  PYTHONNOUSERSITE=1 python run_smoketest.py --mode full
  PYTHONNOUSERSITE=1 python run_smoketest.py --mode nf4
"""
import argparse, time, os
import numpy as np
import torch
from diffusers import WanImageToVideoPipeline, WanTransformer3DModel, AutoencoderKLWan, BitsAndBytesConfig
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel  # NOT ...WithProjection — see diffusers #11006

MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"  # resolved from HF cache (already downloaded)
HERE = os.path.dirname(os.path.abspath(__file__))

def build_pipe(mode, offload):
    # image encoder + VAE always full precision (cheap, and fp32 VAE is what Wan expects)
    image_encoder = CLIPVisionModel.from_pretrained(MODEL_ID, subfolder="image_encoder", torch_dtype=torch.float32)
    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)

    tf_kwargs = dict(subfolder="transformer", torch_dtype=torch.bfloat16)
    if mode == "int8":
        tf_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif mode == "nf4":
        tf_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    elif mode != "full":
        raise ValueError(mode)
    transformer = WanTransformer3DModel.from_pretrained(MODEL_ID, **tf_kwargs)

    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID, vae=vae, image_encoder=image_encoder, transformer=transformer, torch_dtype=torch.bfloat16)

    # offload strategy:
    #   model      : whole-component swap. Fits quantized (nf4), OOMs on the 28 GB bf16 transformer.
    #   sequential : per-submodule streaming (diffusers analog of Wan issue #241 block-offload).
    #                Lowest VRAM, fits the full bf16 transformer on 24 GB. Slow (streams every layer).
    if offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()   # cap the decode memory spike
    return pipe

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "int8", "nf4"], required=True)
    ap.add_argument("--offload", choices=["model", "sequential"], default="model")
    ap.add_argument("--image", default=os.path.join(HERE, "inputs/i2v_input.JPG"))
    ap.add_argument("--frames", type=int, default=33)   # ~2 s @16fps; small for a fast smoke test
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    prompt = ("A white cat wearing sunglasses rides a surfboard on gentle ocean waves, "
              "cinematic, sharp focus, natural daylight.")
    neg = "low quality, blurry, distorted, static, watermark"

    pipe = build_pipe(args.mode, args.offload)

    image = load_image(args.image)
    max_area = 480 * 832
    ar = image.height / image.width
    mod = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    h = round(np.sqrt(max_area * ar)) // mod * mod
    w = round(np.sqrt(max_area / ar)) // mod * mod
    image = image.resize((w, h))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = pipe(image=image, prompt=prompt, negative_prompt=neg,
               height=h, width=w, num_frames=args.frames,
               num_inference_steps=args.steps, guidance_scale=5.0,
               generator=torch.Generator("cuda").manual_seed(args.seed)).frames[0]
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9

    outpath = os.path.join(HERE, f"outputs/wan_i2v_{args.mode}_{args.offload}_{w}x{h}_{args.frames}f.mp4")
    export_to_video(out, outpath, fps=16)
    print(f"\n[{args.mode}/{args.offload}] {w}x{h} {args.frames}f/{args.steps}steps | "
          f"peak VRAM {peak:.1f} GB | {dt:.0f}s | -> {outpath}")

if __name__ == "__main__":
    main()

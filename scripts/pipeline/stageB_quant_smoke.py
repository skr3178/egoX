#!/usr/bin/env python
"""Stage B quantization smoke test: load the Wan2.1-I2V-14B transformer in 4-bit NF4
and confirm it runs on the Blackwell (sm_120) GPU within 24 GB.

This is the core de-risk for the QLoRA plan:
  - does bitsandbytes 4-bit work on sm_120?
  - does the 14B transformer fit in 24 GB once quantized to NF4?
Uses the documented config (NF4 + double-quant + bf16 compute).
"""
import os
import time
import torch
from diffusers import BitsAndBytesConfig, WanTransformer3DModel
from huggingface_hub import snapshot_download

REPO = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"


def gb(x):
    return f"{x/1e9:.2f} GB"


def main():
    import bitsandbytes as bnb
    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | bnb {bnb.__version__}")
    print(f"device {torch.cuda.get_device_name(0)} | cap {torch.cuda.get_device_capability(0)}")
    print(f"torch from: {torch.__file__}")

    qcfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # resolve the LOCAL cached snapshot (no network — avoids the sharded-index API call)
    tpath = os.path.join(snapshot_download(REPO, local_files_only=True), "transformer")
    print(f"\nloading transformer from {tpath}")
    print("(reads ~65 GB fp32 shards, quantizes to NF4 on the fly)...")
    t0 = time.time()
    m = WanTransformer3DModel.from_pretrained(
        tpath, quantization_config=qcfg, torch_dtype=torch.bfloat16,
    )
    dt = time.time() - t0
    print(f"loaded in {dt:.0f}s")

    dev = next(m.parameters()).device
    print(f"transformer device: {dev}")
    print(f"in_channels (config): {getattr(m.config, 'in_channels', '?')}")
    torch.cuda.synchronize()
    print(f"\nVRAM allocated: {gb(torch.cuda.memory_allocated())}")
    print(f"VRAM reserved : {gb(torch.cuda.memory_reserved())}")
    print(f"peak allocated: {gb(torch.cuda.max_memory_allocated())}")

    # count 4-bit (Params4bit) layers to prove quantization actually happened
    n4 = sum(1 for _, p in m.named_parameters() if p.__class__.__name__ == "Params4bit")
    print(f"Params4bit tensors: {n4}")

    print("\nRESULT: NF4 transformer loaded on", dev, "->",
          "✅ 4-bit on Blackwell works" if str(dev).startswith("cuda") and n4 > 0
          else "⚠️ check (not on GPU or not quantized)")


if __name__ == "__main__":
    main()

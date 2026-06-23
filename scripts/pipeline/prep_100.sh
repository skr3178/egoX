#!/bin/bash
# 100-clip cooking dataset prep, end-to-end, single 24 GB machine:
#   Stage-1 (2 vitS workers, 50 each) -> assemble training meta -> precompute (resize 49x256x704
#   + VAE/text/img encode + GGA cache, encoders-only). Result = dataset ready for LoRA training.
set -uo pipefail
ROOT=/media/skr/storage/paper_reproduction/egoX
EP=/media/skr/storage/conda_envs/egox-egoprior/bin/python
ACC=/media/skr/storage/conda_envs/egox/bin/accelerate
CT=/media/skr/SeagateHub1/egoexo4d/cooking_train
cd "$ROOT"

echo "=== [$(date)] STAGE-1: 2 vitS workers (lyra_svda, 50 clips each, faithful calib) ==="
PYTHONNOUSERSITE=1 HF_HUB_ENABLE_HF_TRANSFER=0 $EP local/stage1_full.py --shard 0/2 --limit 50 \
    > "$ROOT/local/prep100_s1a.log" 2>&1 &
PA=$!
PYTHONNOUSERSITE=1 HF_HUB_ENABLE_HF_TRANSFER=0 $EP local/stage1_full.py --shard 1/2 --limit 50 \
    > "$ROOT/local/prep100_s1b.log" 2>&1 &
PB=$!
wait $PA; rcA=$?
wait $PB; rcB=$?
echo "=== [$(date)] STAGE-1 done (worker A rc=$rcA, B rc=$rcB) ==="

echo "=== [$(date)] ASSEMBLE training meta ==="
PYTHONNOUSERSITE=1 $EP local/assemble_meta.py "$CT/meta_train_cooking.json"

echo "=== [$(date)] PRECOMPUTE: resize 49x256x704 + VAE/text/img encode + GGA cache (encoders-only) ==="
cd "$ROOT/EgoX"
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 EGOX_PRECOMPUTE_ONLY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MASTER_ADDR=localhost MASTER_PORT=29510
$ACC launch --config_file configs_acc/1gpu.yaml --num_processes 1 --num_machines 1 \
  finetune.py --model_path ./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers \
    --model_name wan-i2v --model_type wan-i2v --training_type lora --rank 64 --lora_alpha 64 \
    --output_dir ./results/precompute_100 --report_to tensorboard \
    --data_root "$CT" --meta_data_file "$CT/meta_train_cooking.json" \
    --train_resolution 49x256x704 --train_epochs 1 --seed 42 \
    --batch_size 1 --gradient_accumulation_steps 1 --mixed_precision bf16 \
    --num_workers 4 --pin_memory True --checkpointing_steps 10 --checkpointing_limit 1 \
    --gen_fps 30 --cos_sim_scaling_factor 1.0

echo "=== [$(date)] PREP-100 COMPLETE ==="
echo "  clips in meta:    $(PYTHONNOUSERSITE=1 $EP -c "import json;print(len(json.load(open('$CT/meta_train_cooking.json'))))")"
echo "  video-latent cache: $(ls $CT/cache/video_latent/wan-i2v/49x256x704/ 2>/dev/null | wc -l)"
echo "  attn(GGA) cache:    $(ls $CT/cache/attn_maps/wan-i2v/49x256x704/ 2>/dev/null | wc -l)"
echo "  -> dataset ready for LoRA training (data_root=$CT, meta=meta_train_cooking.json, res=49x256x704)"

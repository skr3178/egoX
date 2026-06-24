#!/bin/bash
# GPU smoke: rank-128 + val_loss path. A few real opt-steps + val firing, throwaway output.
# Confirms: NF4 rank-128 trains, val_loss computes+logs, peak VRAM < 24 GB.
set -eo pipefail
cd "$(dirname "$0")/../EgoX"
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 EGOX_NF4=1 EGOX_8BIT_OPTIM=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=localhost MASTER_PORT=29522
CT=/media/skr/SeagateHub1/egoexo4d/cooking_train

/media/skr/storage/conda_envs/egox/bin/accelerate launch --config_file configs_acc/1gpu.yaml --num_processes 1 --num_machines 1 \
  finetune.py \
    --model_path ./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers \
    --model_name wan-i2v --model_type wan-i2v \
    --training_type lora --rank 128 --lora_alpha 128 \
    --output_dir ./results/EgoX_r128_smoke \
    --report_to tensorboard \
    --data_root "$CT" \
    --meta_data_file "$CT/meta_train_clean.json" \
    --validation_meta_file "$CT/meta_val_clean.json" \
    --val_loss_steps 2 \
    --train_epochs 1 --train_steps 4 --seed 42 \
    --batch_size 1 --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --lr_scheduler constant_with_warmup --lr_warmup_steps 1 \
    --optimizer adamw --beta1 0.9 --beta2 0.95 --weight_decay 1e-4 --max_grad_norm 1.0 \
    --mixed_precision bf16 --gradient_checkpointing True \
    --num_workers 8 --pin_memory True \
    --checkpointing_steps 1000 --checkpointing_limit 1 \
    --gen_fps 30 --cos_sim_scaling_factor 1.0
echo "SMOKE END: $(date)"

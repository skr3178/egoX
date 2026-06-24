#!/bin/bash
# Re-run ckpt-100 inference with the §5a-CORRECTED GGA (pinhole+R^T+matrix, no-crop).
# New output dir so we can A/B against the old (buggy-GGA) eval_out_ckpt100/.
set -uo pipefail
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EGOX_NF4=1                 # NF4 4-bit base (matches training)
export EGOX_INFER_OFFLOAD=1       # cpu offload (all-resident OOMs at 24GB)
cd /media/skr/storage/paper_reproduction/egoX/EgoX

SNAP=$(ls -d ~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-I2V-14B-480P-Diffusers/snapshots/*/ | head -1)
LORA=results/EgoX_cooking84/checkpoint-100
META=/media/skr/SeagateHub1/egoexo4d/cooking_train/infer_meta_ckpt100.json
OUT=/media/skr/SeagateHub1/egoexo4d/cooking_train/eval_out_ckpt100_ggafix
mkdir -p "$OUT"

echo "model_path=$SNAP"
echo "lora=$LORA  meta=$META  out=$OUT"
/media/skr/storage/conda_envs/egox/bin/python infer_nf4.py \
    --model_path "$SNAP" --lora_path "$LORA" --lora_rank 256 \
    --meta_data_file "$META" \
    --use_GGA --idx 0 --seed 42 --cos_sim_scaling_factor 1.0 \
    --out "$OUT"
RC=$?
echo "------------------------------------------------------------"
echo "infer exit: $RC"
echo "outputs:"; ls -la "$OUT" 2>/dev/null
echo "=== CKPT100 GGAFIX INFER COMPLETE ==="

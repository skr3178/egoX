#!/bin/bash
export PYTHONNOUSERSITE=1            # egox env: avoid ~/.local torch leak
export HF_HUB_OFFLINE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # recover ~600MB fragmentation (OOM was only 600MB over)
cd /media/skr/storage/paper_reproduction/egoX/EgoX
SNAP=$(ls -d ~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-I2V-14B-480P-Diffusers/snapshots/*/ | head -1)
LORA=/media/skr/SeagateHub1/egox_checkpoints/EgoX
OUT=/media/skr/storage/paper_reproduction/egoX/local/infer_out
LOCAL=/media/skr/storage/paper_reproduction/egoX/local
PEAK="$LOCAL/.ipeak"; FLAG="$LOCAL/.irun"; echo 0 > "$PEAK"; touch "$FLAG"
( while [ -f "$FLAG" ]; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1|tr -d ' ')
    p=$(cat "$PEAK" 2>/dev/null||echo 0); [ -n "$u" ]&&[ "$u" -gt "$p" ] 2>/dev/null&&echo "$u">"$PEAK"
    sleep 2; done ) &
echo "model_path=$SNAP"
/media/skr/storage/conda_envs/egox/bin/python infer_nf4.py \
    --model_path "$SNAP" --lora_path "$LORA" \
    --meta_data_file example/egoexo4D/meta.json \
    --use_GGA --idx 0 --seed 42 --cos_sim_scaling_factor 3.0 --out "$OUT"
RC=$?
rm -f "$FLAG"; sleep 2
echo "------------------------------------------------------------"
echo "infer exit: $RC | peak VRAM: $(cat "$PEAK") MiB / 24467 MiB"
echo "outputs:"; ls -la "$OUT" 2>/dev/null
echo "=== INFER NF4 COMPLETE ==="

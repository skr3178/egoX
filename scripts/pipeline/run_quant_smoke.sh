#!/bin/bash
export PYTHONNOUSERSITE=1   # egox env: avoid ~/.local torch 2.11+cu130 leak (bnb built for 2.10)
export HF_HUB_OFFLINE=0
LOCAL=/media/skr/storage/paper_reproduction/egoX/local
PEAK="$LOCAL/.qpeak"; FLAG="$LOCAL/.qrun"; echo 0 > "$PEAK"; touch "$FLAG"
( while [ -f "$FLAG" ]; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1|tr -d ' ')
    p=$(cat "$PEAK" 2>/dev/null||echo 0); [ -n "$u" ]&&[ "$u" -gt "$p" ] 2>/dev/null&&echo "$u">"$PEAK"
    sleep 1; done ) &
/media/skr/storage/conda_envs/egox/bin/python "$LOCAL/stageB_quant_smoke.py"; RC=$?
rm -f "$FLAG"; sleep 2
echo "------------------------------------------------------------"
echo "python exit: $RC | sampled peak VRAM: $(cat "$PEAK") MiB / 24467 MiB"
echo "=== QUANT SMOKE COMPLETE ==="

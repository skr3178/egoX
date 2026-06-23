#!/bin/bash
LOCAL=/media/skr/storage/paper_reproduction/egoX/local
PEAK="$LOCAL/.apeak"; FLAG="$LOCAL/.arun"; echo 0 > "$PEAK"; touch "$FLAG"
( while [ -f "$FLAG" ]; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1|tr -d ' ')
    p=$(cat "$PEAK" 2>/dev/null||echo 0); [ -n "$u" ]&&[ "$u" -gt "$p" ] 2>/dev/null&&echo "$u">"$PEAK"
    sleep 2; done ) &
bash /media/skr/storage/paper_reproduction/egoX/EgoX/scripts/finetune_phaseA.sh; RC=$?
rm -f "$FLAG"; sleep 2
echo "------------------------------------------------------------"
echo "phaseA exit: $RC | peak VRAM: $(cat "$PEAK") MiB / 24467 MiB"
echo "=== PHASE A SMOKE DONE ==="

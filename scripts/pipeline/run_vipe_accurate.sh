#!/bin/bash
# Accurate vit-L ViPE Stage-1 run: let GeoCalib ESTIMATE intrinsics (no forced GT, unlike the
# earlier fit test which passed a wrong 255.6 and was inaccurate). Faithful path.
set -uo pipefail
ENV=/media/skr/storage/conda_envs/egox-egoprior
LOCAL=/media/skr/storage/paper_reproduction/egoX/local
EX=/media/skr/storage/paper_reproduction/egoX/EgoX/example/egoexo4D
CLIP=cmu_soccer06_6_877_925
EXO="$EX/videos/$CLIP/exo.mp4"
OUT="$LOCAL/vipe_accurate_out"
export PYTHONNOUSERSITE=1
export HF_HUB_ENABLE_HF_TRANSFER=0     # env lacks hf_transfer

PEAK="$LOCAL/.vpeak"; FLAG="$LOCAL/.vrun"; echo 0 > "$PEAK"; touch "$FLAG"
( while [ -f "$FLAG" ]; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1|tr -d ' ')
    p=$(cat "$PEAK" 2>/dev/null||echo 0); [ -n "$u" ]&&[ "$u" -gt "$p" ] 2>/dev/null&&echo "$u">"$PEAK"
    sleep 1; done ) &

echo "=== accurate vit-L ViPE (lyra, GeoCalib-estimated intrinsics) on $CLIP ==="
rm -rf "$OUT"; mkdir -p "$OUT"
"$ENV/bin/vipe" infer "$EXO" -o "$OUT" -p lyra --assume_fixed_camera_pose
RC=$?
rm -f "$FLAG"; sleep 2
echo "------------------------------------------------------------"
echo "infer exit: $RC | peak VRAM: $(cat "$PEAK") MiB / 24467 MiB"
echo "outputs:"; find "$OUT" -maxdepth 3 -type f | sed "s#$OUT/##" | head -20

# --- accuracy check: compare GeoCalib intrinsics vs satya vit-S reference (depth-model-independent) ---
echo "=== intrinsics accuracy vs satya vit-S GeoCalib (should be ~equal) ==="
NPZ=$(find "$OUT" -name '*.npz' -path '*intrinsics*' | head -1)
TMP=$(mktemp -d); unzip -q "$LOCAL/../egoX/vitS/vipe_intrinsics_egoexo4d_5clips.zip" -d "$TMP" 2>/dev/null
PYTHONNOUSERSITE=1 "$ENV/bin/python" - "$NPZ" "$TMP/$CLIP.npz" <<'PY'
import numpy as np, sys
try:
    a=np.load(sys.argv[1])['data']; b=np.load(sys.argv[2])['data']
    print(f"  vit-L (this run) data[0]: {a[0]}")
    print(f"  vit-S (satya ref) data[0]: {b[0]}")
    n=min(len(a),len(b)); print(f"  max abs diff: {float(np.max(np.abs(a[:n]-b[:n]))):.3f}")
except Exception as e:
    print("  compare failed:", e)
PY
rm -rf "$TMP"
echo "=== ACCURATE VIPE COMPLETE ==="

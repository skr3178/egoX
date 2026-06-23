#!/bin/bash
# Wait for ckpt downloads, then run ONE-clip ViPE (lyra / vit-L) while sampling peak VRAM,
# to determine whether the faithful Stage 1 pipeline fits the 24 GB Blackwell GPU.
set -uo pipefail

ENV=/media/skr/storage/conda_envs/egox-egoprior
LOCAL=/media/skr/storage/paper_reproduction/egoX/local
EX=/media/skr/storage/paper_reproduction/egoX/EgoX/example/egoexo4D
CLIP=cmu_soccer06_6_877_925
EXO="$EX/videos/$CLIP/exo.mp4"
OUT="$LOCAL/vipe_test_out"
export PYTHONNOUSERSITE=1          # avoid the ~/.local torch 2.11+cu130 leak
export HF_HUB_ENABLE_HF_TRANSFER=0 # egox-egoprior env lacks hf_transfer; use normal HF downloads
GPU_TOTAL=24467               # MiB (RTX PRO 4000 Blackwell)

# exo K scaled from 4K meta intrinsics to the 784x448 example video
K='[[255.6,0.0,392.0],[0.0,259.65,224.0],[0.0,0.0,1.0]]'

echo "=== [wait] for checkpoint downloads to finish ==="
for i in $(seq 1 120); do
  if grep -q 'CKPT DOWNLOAD COMPLETE' "$LOCAL/ckpt_download.log" 2>/dev/null; then echo "  downloads complete"; break; fi
  if ! pgrep -f download_stage1_ckpts.sh >/dev/null 2>&1; then echo "  downloader exited"; break; fi
  sleep 15
done
# sanity: required ckpts present
for f in ~/.cache/torch/hub/checkpoints/video_depth_anything_vitl.pth \
         ~/.cache/torch/hub/checkpoints/groundingdino_swint_ogc.pth \
         ~/.cache/torch/hub/sam/sam_vit_b_01ec64.pth \
         ~/.cache/torch/hub/geocalib/pinhole.tar \
         ~/.cache/torch/hub/aot/R50_DeAOTL_PRE_YTB_DAV.pth; do
  [ -s "$f" ] && echo "  ok  $(basename $f)" || { echo "  MISSING $f — aborting test"; exit 3; }
done
[ -d ~/.cache/huggingface/hub/models--Ruicheng--moge-2-vitl-normal ] && echo "  ok  MoGe-2 (HF)" || { echo "  MISSING MoGe — abort"; exit 3; }

echo "=== [gpu] free memory before run ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# --- peak-VRAM sampler (flag-file controlled) ---
PEAK="$LOCAL/.vram_peak"; FLAG="$LOCAL/.vram_running"; echo 0 > "$PEAK"; touch "$FLAG"
( while [ -f "$FLAG" ]; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    p=$(cat "$PEAK" 2>/dev/null || echo 0)
    [ -n "$u" ] && [ "$u" -gt "$p" ] 2>/dev/null && echo "$u" > "$PEAK"
    sleep 1
  done ) &

echo "=== [run] vipe infer (pipeline=lyra / vit-L) on $CLIP (50 frames @ 784x448) ==="
rm -rf "$OUT"; mkdir -p "$OUT"
"$ENV/bin/vipe" infer "$EXO" -o "$OUT" -p lyra \
    --assume_fixed_camera_pose --use_exo_intrinsic_gt "$K"
RC=$?
rm -f "$FLAG"; sleep 2

PK=$(cat "$PEAK" 2>/dev/null || echo 0)
echo
echo "============================================================"
echo "vipe exit code : $RC"
echo "PEAK VRAM      : ${PK} MiB / ${GPU_TOTAL} MiB  ($(awk "BEGIN{printf \"%.0f\", $PK*100/$GPU_TOTAL}")%)"
if [ "$RC" -eq 0 ] && [ "$PK" -lt "$GPU_TOTAL" ]; then
  echo "RESULT         : ✅ FITS — faithful vit-L Stage 1 runs on this 24 GB GPU"
else
  echo "RESULT         : ❌ check above (nonzero exit or OOM)"
fi
echo "=== outputs ==="; find "$OUT" -maxdepth 3 -type f | head -20
echo "=== TEST COMPLETE ==="

#!/bin/bash
# Pre-download all Stage 1 (ViPE / lyra vit-L) model checkpoints to the EXACT paths
# ViPE expects, so the first run downloads nothing. Resumable + idempotent.
set -uo pipefail

HUB="$HOME/.cache/torch/hub"
CKPT="$HUB/checkpoints"
EGOX_HF=/media/skr/storage/conda_envs/egox/bin/hf            # has hf + hf_transfer
GDOWN=/media/skr/storage/conda_envs/egox-egoprior/bin/gdown  # for AOT (gdrive)
mkdir -p "$CKPT" "$HUB/sam" "$HUB/geocalib" "$HUB/aot"

# curl with resume (-C -), retries, generous timeouts for the ~12 MB/s link
dl() { # url dest
  local url="$1" dst="$2"
  echo ">>> $(basename "$dst")"
  curl -L --fail --retry 8 --retry-delay 5 --retry-all-errors \
       --connect-timeout 30 -C - -o "$dst" "$url" \
    && echo "    done: $(du -h "$dst" | cut -f1)" \
    || echo "    FAILED: $dst"
}

echo "=== [1/6] VDA-Large (vitl) ==="
dl "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth" \
   "$CKPT/video_depth_anything_vitl.pth"

echo "=== [2/6] GroundingDINO (swint_ogc) ==="
dl "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth" \
   "$CKPT/groundingdino_swint_ogc.pth"

echo "=== [3/6] SAM (vit_b) ==="
dl "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" \
   "$HUB/sam/sam_vit_b_01ec64.pth"

echo "=== [4/6] GeoCalib (pinhole.tar) — clearing stale .partial first ==="
rm -f "$HUB/geocalib/"*.partial
dl "https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar" \
   "$HUB/geocalib/pinhole.tar"

echo "=== [5/6] AOT (R50_DeAOTL, gdrive via gdown) ==="
AOT="$HUB/aot/R50_DeAOTL_PRE_YTB_DAV.pth"
if [ -s "$AOT" ]; then echo "    exists, skip"; else
  "$GDOWN" "https://drive.google.com/file/d/1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ/view" -O "$AOT" --fuzzy \
    && echo "    done: $(du -h "$AOT" | cut -f1)" || echo "    FAILED: AOT"
fi

echo "=== [6/6] MoGe-2-vitl (HF cache) ==="
HF_HUB_ENABLE_HF_TRANSFER=1 "$EGOX_HF" download Ruicheng/moge-2-vitl-normal >/dev/null \
  && echo "    done (HF cache)" || echo "    FAILED: MoGe"

echo
echo "=== SUMMARY (sizes on disk) ==="
for f in "$CKPT/video_depth_anything_vitl.pth" "$CKPT/groundingdino_swint_ogc.pth" \
         "$HUB/sam/sam_vit_b_01ec64.pth" "$HUB/geocalib/pinhole.tar" \
         "$HUB/aot/R50_DeAOTL_PRE_YTB_DAV.pth"; do
  [ -s "$f" ] && printf "  %6s  %s\n" "$(du -h "$f" | cut -f1)" "$f" || printf "  MISSING %s\n" "$f"
done
du -sh ~/.cache/huggingface/hub/models--Ruicheng--moge-2-vitl-normal 2>/dev/null | sed 's/^/  /'
echo "=== CKPT DOWNLOAD COMPLETE ==="

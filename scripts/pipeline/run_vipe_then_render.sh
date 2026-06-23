#!/bin/bash
set -uo pipefail
export PYTHONNOUSERSITE=1
export HF_HUB_ENABLE_HF_TRANSFER=0
ENV=/media/skr/storage/conda_envs/egox-egoprior
EX=/media/skr/storage/paper_reproduction/egoX/EgoX/example/egoexo4D
CLIP=cmu_soccer06_6_877_925
EXO="$EX/videos/$CLIP/exo.mp4"
VOUT=/media/skr/storage/paper_reproduction/egoX/local/vipe_accurate_out49
ROUT=/media/skr/storage/paper_reproduction/egoX/local/egoprior_render_out
RENDERER=/media/skr/storage/paper_reproduction/egoX/EgoX/EgoX-EgoPriorRenderer

echo "=== [1/2] vipe infer on 49 frames (0-48) ==="
rm -rf "$VOUT"; mkdir -p "$VOUT"
"$ENV/bin/vipe" infer "$EXO" -o "$VOUT" -p lyra --assume_fixed_camera_pose --end_frame 48
echo "vipe exit: $?"

echo "=== [2/2] render ego-prior (end_frame 48) ==="
rm -rf "$ROUT"; mkdir -p "$ROUT"
cd "$RENDERER"
"$ENV/bin/python" scripts/render_vipe_pointcloud.py \
  --input_dir "$VOUT/$CLIP" --meta_json_path "$EX/meta.json" --out_dir "$ROUT" \
  --start_frame 0 --end_frame 48 --point_size 5.0
echo "render exit: $?"
echo "=== outputs ==="; find "$ROUT" -name '*.mp4' -o -name '*.png' | head
echo "=== VIPE+RENDER DONE ==="

#!/bin/bash
export PYTHONNOUSERSITE=1
export HF_HUB_ENABLE_HF_TRANSFER=0
ENV=/media/skr/storage/conda_envs/egox-egoprior
EX=/media/skr/storage/paper_reproduction/egoX/EgoX/example/egoexo4D
CLIP=cmu_soccer06_6_877_925
VOUT=/media/skr/storage/paper_reproduction/egoX/local/vipe_accurate_out49
ROUT=/media/skr/storage/paper_reproduction/egoX/local/egoprior_render_fisheye
cd /media/skr/storage/paper_reproduction/egoX/EgoX/EgoX-EgoPriorRenderer
rm -rf "$ROUT"; mkdir -p "$ROUT"
"$ENV/bin/python" scripts/render_vipe_pointcloud.py \
  --input_dir "$VOUT/$CLIP" --meta_json_path "$EX/meta.json" --out_dir "$ROUT" \
  --start_frame 0 --end_frame 48 --point_size 5.0 --fish_eye_rendering
echo "render exit: $?"
find "$ROUT" -name '*.mp4'
echo "=== FISHEYE RENDER DONE ==="

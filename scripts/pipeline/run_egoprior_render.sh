#!/bin/bash
export PYTHONNOUSERSITE=1
export HF_HUB_ENABLE_HF_TRANSFER=0
cd /media/skr/storage/paper_reproduction/egoX/EgoX/EgoX-EgoPriorRenderer
"/media/skr/storage/conda_envs/egox-egoprior/bin/python" scripts/render_vipe_pointcloud.py \
  --input_dir "/media/skr/storage/paper_reproduction/egoX/local/vipe_accurate_out/cmu_soccer06_6_877_925" --meta_json_path "/media/skr/storage/paper_reproduction/egoX/EgoX/example/egoexo4D/meta.json" --out_dir "/media/skr/storage/paper_reproduction/egoX/local/egoprior_render_out" \
  --start_frame 0 --end_frame 49 --point_size 5.0
echo "=== EGOPRIOR RENDER DONE (exit $?) ==="
find "/media/skr/storage/paper_reproduction/egoX/local/egoprior_render_out" -name '*.mp4' -o -name '*.png' | head

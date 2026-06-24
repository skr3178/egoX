#!/bin/bash
# Generate the ego_Prior with vit-L (lyra) for ONE clip, into an isolated dir, to compare vs the
# existing vit-S (lyra_svda) prior. Faithful to stage1_cooking_clip.py but manual (clip not in meta_train).
set -uo pipefail
ENV=/media/skr/storage/conda_envs/egox-egoprior
RENDERER=/media/skr/storage/paper_reproduction/egoX/EgoX/EgoX-EgoPriorRenderer
CT=/media/skr/SeagateHub1/egoexo4d/cooking_train
CLIP=fair_cooking_06_6_1000_1048
TAKE=fair_cooking_06_6
CALIB=/media/skr/SeagateHub1/egoexo4d/takes/$TAKE/trajectory/online_calibration.jsonl
OUT=/media/skr/SeagateHub1/egoexo4d/vitL_prior_cmp
export PYTHONNOUSERSITE=1 HF_HUB_ENABLE_HF_TRANSFER=0

rm -rf "$OUT"; mkdir -p "$OUT/videos/$CLIP"
cp "$CT/videos/$CLIP/exo.mp4" "$OUT/videos/$CLIP/exo.mp4"

# Kgt (meta intrinsics scaled to exo res) + render_meta, from the val meta entry
"$ENV/bin/python" - "$CT" "$CLIP" "$OUT" <<'PY'
import json, cv2, sys
CT, CLIP, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
e = json.load(open(f"{CT}/meta_val_20.json"))[CLIP]
cap = cv2.VideoCapture(f"{OUT}/videos/{CLIP}/exo.mp4"); W=int(cap.get(3)); H=int(cap.get(4)); cap.release()
K = [[float(x) for x in r] for r in e["camera_intrinsics"]]
sx, sy = W/(2*K[0][2]), H/(2*K[1][2])
Kgt = [[K[0][0]*sx,0.0,W/2.0],[0.0,K[1][1]*sy,H/2.0],[0.0,0.0,1.0]]
open(f"{OUT}/Kgt.json","w").write(json.dumps(Kgt, separators=(',',':')))
rmeta = {"test_datasets":[{"exo_path":f"./videos/{CLIP}/exo.mp4","ego_prior_path":f"./videos/{CLIP}/ego_Prior.mp4",
    "prompt":e["prompt"],"camera_intrinsics":e["camera_intrinsics"],"camera_extrinsics":e["camera_extrinsics"],
    "ego_intrinsics":e["ego_intrinsics"],"ego_extrinsics":e["ego_extrinsics"]}]}
json.dump(rmeta, open(f"{OUT}/render_meta_{CLIP}.json","w"))
print(f"exo {W}x{H}  Kgt={Kgt}")
PY
KGT=$(cat "$OUT/Kgt.json")
echo "KGT=$KGT"

# 1) ViPE vit-L (lyra)
cp "$OUT/videos/$CLIP/exo.mp4" "$OUT/$CLIP.mp4"
echo "=== [1/3] ViPE vit-L (lyra) ==="
"$ENV/bin/vipe" infer "$OUT/$CLIP.mp4" -o "$OUT/vipe_results" -p lyra \
    --assume_fixed_camera_pose --end_frame 48 --use_exo_intrinsic_gt "$KGT"
rm -f "$OUT/$CLIP.mp4"
VRES="$OUT/vipe_results/$CLIP"

# 2) depth zip -> npy
echo "=== [2/3] depth -> npy ==="
"$ENV/bin/python" "$RENDERER/scripts/convert_depth_zip_to_npy.py" \
    --depth_path "$VRES/depth" --egox_depthmaps_path "$OUT/depth_maps"

# 3) render fisheye ego_prior (same flags as Stage-1)
echo "=== [3/3] render fisheye ego_prior (vit-L) ==="
cd "$RENDERER"
"$ENV/bin/python" scripts/render_vipe_pointcloud.py \
    --input_dir "$VRES" --meta_json_path "$OUT/render_meta_$CLIP.json" --out_dir "$OUT/ego_prior_render" \
    --start_frame 0 --end_frame 48 --point_size 5.0 \
    --fish_eye_rendering --use_mean_bg --only_bg --online_calibration_path "$CALIB"
cp "$OUT/ego_prior_render/$CLIP/ego_Prior.mp4" "$OUT/videos/$CLIP/ego_Prior.mp4" 2>/dev/null || true
echo "=== VIT-L PRIOR DONE ==="
ls -la "$OUT/videos/$CLIP/" "$OUT/depth_maps/$CLIP/" 2>/dev/null | head

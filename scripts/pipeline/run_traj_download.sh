#!/bin/bash
EGOEXO=/media/skr/storage/paper_reproduction/hands/.venv-egoexo/bin/egoexo
OUT=/media/skr/SeagateHub1/egoexo4d
UIDS=$(cat /media/skr/storage/paper_reproduction/egoX/local/cooking_uids.txt)
echo "=== downloading take_trajectory for $(echo "$UIDS"|wc -w) cooking takes -> $OUT ==="
"$EGOEXO" -o "$OUT" --parts take_trajectory --uids $UIDS --s3_profile default --num_workers 8 -y
echo "=== TRAJ DOWNLOAD DONE (exit $?) ==="
echo "online_calibration.jsonl count now: $(find "$OUT/takes" -iname 'online_calibration.jsonl' 2>/dev/null | wc -l)"

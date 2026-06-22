# EgoX Preprocessing Pipeline (reproduction guide)

**No — you can't use Ego-Exo4D as-is. It needs a heavy preprocessing pipeline.**

What you downloaded (the `meta_*.json` + CSVs) is the **output** of EgoX's preprocessing,
not the raw data. The metadata tells you *which* clips and *what* camera parameters/prompts
to use — but the actual training inputs are derived files that don't exist yet.

## What each training sample requires

From [core/finetune/datasets/wan_dataset.py:84-96](EgoX/core/finetune/datasets/wan_dataset.py#L84-L96):

```python
self.exo_videos.append(... meta['exo_video_path'])        # trimmed exo clip
self.ego_gt_videos.append(... meta['ego_video_path'])     # trimmed ego GT clip (width-concat)
self.ego_prior_videos.append(... meta['ego_prior_path'])  # RENDERED ego prior  ← the hard one
self.prompts.append(meta['prompt'])                       # VLM-generated caption
... os.path.join(data_root,'depth_maps', meta['take_name'])     # depth maps
... meta['vipe_results_path'] ... meta['best_camera']           # ViPE pose results
```

So per clip you need **six derived artifacts**, none of which ship with Ego-Exo4D:

| Artifact | How it's produced | Status |
|---|---|---|
| Trimmed `exo.mp4` clips | Cut raw takes by `frame_range` (from the CSV) | You must do it |
| Trimmed ego GT clips | Same, ego camera | You must do it |
| `ego_Prior.mp4` | Reproject exo point cloud into ego camera | ⚠️ original repo never released this; shi3z reverse-engineered it |
| Depth maps | Monocular depth per take | You must run it |
| ViPE results (poses + intrinsics `.npz`) | Run ViPE (Video Pose Engine) | You must run it |
| `prompt` captions | VLM via [caption.py](EgoX/caption.py) | Already in the JSON for these clips |

## The critical piece: the Ego Prior

EgoX's whole method conditions on a **rendered ego-prior video** — the exo scene
geometrically reprojected into the egocentric viewpoint. That's what
[infer.py](EgoX/infer.py) spends 150+ lines doing (depth → point cloud → reproject into
ego intrinsics/extrinsics). The metadata you have provides the camera params for this, but
generating the prior videos requires running the renderer.

This is exactly why the **shi3z fork** exists — from its README:

> "The original repository has not yet released the data preprocessing code for Ego Prior
> generation. We reverse-engineered the Ego Prior pipeline based on the paper."

→ [generate_ego_prior.py](EgoX-shi3z/generate_ego_prior.py) + `EgoX-EgoPriorRenderer/`
(uses ViPE).

## What you'd actually have to do

1. **Download raw Ego-Exo4D takes** (large — the actual videos; you only have metadata
   now) via their official downloader.
2. **Trim** exo + ego into clips per the `frame_range` in the CSVs.
3. **Run ViPE** on each take → camera poses + intrinsics.
4. **Compute depth maps** per take.
5. **Render ego-prior videos** (the shi3z renderer).
6. **Captions** are already provided in `meta_train.json`.

The metadata you downloaded covers **3,510 train / seen-val / unseen-test clips** — it's
the recipe and labels, but you still have to build the ingredients from raw Ego-Exo4D using
the (forked) preprocessing code.

---

See also [dataset.md](dataset.md) for the take inventory, storage estimates, and access details.

# EgoX Dataset Notes (Ego-Exo4D)

Status of the Ego-Exo4D data needed to train/eval the EgoX reproduction.

## What EgoX consumes

EgoX is **not** trained on raw Ego-Exo4D. Each training sample is built from derived
artifacts ([core/finetune/datasets/wan_dataset.py:84-96](EgoX/core/finetune/datasets/wan_dataset.py#L84-L96)):

| Artifact | Source | Status |
|---|---|---|
| Trimmed `exo.mp4` clip | cut raw take by `frame_range` (CSV) | must build |
| Trimmed ego GT clip (width-concat) | cut raw ego by `frame_range` | must build |
| **`ego_Prior.mp4`** (rendered ego prior) | reproject exo point cloud into ego cam | ⚠️ original repo never released; reverse-engineered in `EgoX-shi3z` (`generate_ego_prior.py` + `EgoX-EgoPriorRenderer`, uses ViPE) |
| Depth maps | monocular depth per take | must run |
| ViPE results (poses + intrinsics `.npz`) | run ViPE (Video Pose Engine) | must run |
| `prompt` captions | VLM ([caption.py](EgoX/caption.py)) | already provided in `meta_train.json` |

The downloaded `egoexo4D/meta_*.json` + CSVs are the **recipe/labels** (which clips,
camera params, prompts) — the actual video files still have to be produced from raw
Ego-Exo4D via the (forked) preprocessing code.

Note: EgoX uses **one exo view** per clip (`best_camera`) + the ego RGB — not all 4 exo cams.

## Downloaded metadata (in `egoexo4D/`)

| File | Contents |
|---|---|
| `meta_train.json` | 3,510 train clip entries (exo_path, ego_prior_path, prompt, camera intr/extr, ego intr/extr) |
| `meta_seen.json` | val (`val_datasets`) |
| `meta_unseen.json` | test (`test_datasets`) |
| `dataset_info_train_seen.csv` | `task_name\|frame_range\|split` — 3,979 rows, train+val |
| `dataset_info_unseen.csv` | test rows |

## Take requirements vs. what's on disk

| | Count |
|---|---|
| Distinct takes EgoX needs (train+val) | **1,412** (→ 3,979 clip rows / 3,510 train entries) |
| + unseen/test takes | 15 |
| **Total takes needed** | **1,427** |
| Valid in official Ego-Exo4D catalog (`takes.json`) | 1,427 / 1,427 ✅ (exact name match) |
| **Physically on disk** | **7 of 1,427** ❌ |
| Train/val clips currently buildable | **14 of 3,979** |

The 7 present takes (downloaded for the sibling **TriHands** project) live at
`/media/skr/storage/paper_reproduction/hands/trihands/egoexo_data/takes/`:
`iiith_cooking_109_4`, `iiith_cooking_111_4`, `iiith_cooking_112_3`,
`sfu_cooking023_4`, `sfu_cooking025_2`, `sfu_cooking026_6`, `sfu_cooking031_10`.
(10 takes total there; 3 are not needed by EgoX.)

## Access

- Ego-Exo4D **license granted**, expires **2026-07-02**.
- AWS creds in `~/.aws` default profile (not in repo).
- Downloader: `egoexo` CLI in `.venv-egoexo` (uv, py3.10) under the hands project.
  - `egoexo -o <out> --parts ... --s3_profile default -y`

## Storage estimate (remaining 1,420 takes)

Measured from the 10 takes on disk: avg **488 MB/take** (video parts only).
Per take the 658 MB of video is mostly the 4× 4K exo cams (~613 MB); ego RGB is ~39 MB.

| Scenario | Estimate |
|---|---|
| Full take bundles (4 cams + everything) | **~700 GB** |
| Only parts EgoX uses (1 exo + ego RGB) | **~210–280 GB** |
| + derived artifacts (depth, ViPE, ego-prior, GT clips) | +50–100 GB |

Caveats:
- All 10 measured takes are short cooking takes (~25–52 s); EgoX's 1,427 span bike,
  basketball, cooking, etc. — duration/stream counts vary, so ±~30%.
- `egoexo` downloads `frame_aligned_videos` as a group (all cams), so plan for the
  ~700 GB transfer even if you only keep ~250 GB after pruning unused exo cams.

## Available disks (as of 2026-06-21)

| Disk | Mount | Total | Free | Type |
|---|---|---|---|---|
| `nvme1n1p1` | `/` | 916 GB | 322 GB | NVMe |
| `nvme0n1p2` (Storage_Drive) | `/media/skr/storage` | 910 GB | 197 GB | NVMe |
| **`sda2` (SeagateHub1)** | **`/media/skr/SeagateHub1`** | **3.6 TB** | **3.5 TB** | **USB HDD** |

→ The **4 TB SeagateHub1 (3.5 TB free)** is the target for the dataset — it absorbs the
full ~700 GB download plus all derived artifacts with ~2.6 TB to spare. The NVMe
`/media/skr/storage` (197 GB free) is too small for the bulk pull.

### Storage plan

- **Bulk** (raw Ego-Exo4D takes + ViPE results + depth maps) → `/media/skr/SeagateHub1/egoexo4d/`
  — sequential, bandwidth-bound; HDD is fine (the `egoexo` downloader does only ~6–9 MB/s
  per connection, so the disk is never the bottleneck).
- **Active training clips** (trimmed exo + ego-prior + width-concat GT) → optionally stage
  onto the NVMe `/media/skr/storage` for faster dataloader random reads. Reading video clips
  from the spinning USB drive during training works but is slower than NVMe; if I/O stalls
  appear, stage the active subset onto NVMe or pre-decode to a compact format.

Worst-case footprint (full bundles + V2/ViPE install + all artifacts) ≈ **~850 GB** → leaves
~2.6 TB free on SeagateHub1.

## Recommended order

1. **Pilot** — build the 14 clips from the 7 already-local takes end-to-end through the
   ego-prior renderer to validate the pipeline before any large download.
2. Generate the take-UID list (map `task_name` → uid via `takes.json`) and fetch the
   needed video parts into `/media/skr/SeagateHub1/egoexo4d/`; prune unused exo cams if
   you want to stay near the lean footprint.

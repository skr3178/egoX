# Stage 1 (ViPE / EgoPriorRenderer) — Blackwell findings

Local-only notes (kept outside the git repo on purpose). Dated 2026-06-22.

## Env

- `egox-egoprior` at `/media/skr/storage/conda_envs/egox-egoprior` (py3.10, torch 2.7.0+cu128,
  CUDA 12.8, sm_120 verified; pytorch3d 0.7.9 compiled `_C` loads; moge, vipe import).
- Built via `egox_blackwell_local/build_stage1_env_blackwell.sh` (Blackwell port of the
  satya/RTX3060 `build_stage1_env.sh`). Three gotchas baked into the script:
  `UV_HTTP_TIMEOUT=600`, `PYTHONNOUSERSITE=1` (user-site torch 2.11+cu130 leak),
  `CPATH/LIBRARY_PATH` → `$ENV/targets/x86_64-linux/{include,lib}` (conda CUDA header layout).
- **Runtime gotcha:** must run with `HF_HUB_ENABLE_HF_TRANSFER=0` — the env lacks `hf_transfer`,
  and ViPE does runtime HF downloads (e.g. GroundingDINO BERT tokenizer / DINOv2). With the var
  inherited as 1 from the shell, ViPE crashes before loading any model. Also `PYTHONNOUSERSITE=1`.

## vit-L FIT TEST — PASSED (2026-06-22)

Faithful `lyra` pipeline (MoGe-2 + VDA **vit-L**) on `cmu_soccer06_6_877_925`, 50 frames @ 784×448:

| metric | value |
|---|---|
| vipe exit | 0 |
| **peak VRAM** | **21013 MiB / 24467 MiB (86%)** |
| verdict | ✅ FITS on the 24 GB Blackwell GPU |

→ The faithful vit-L Stage 1 pipeline runs end-to-end on this GPU. vit-S downgrade
(`lyra_svda.yaml`) is NOT needed here — that was only for the 12 GB RTX 3060.
86% is comfortable but not huge headroom; longer clips/higher res could tighten it.
Test harness: `egox_blackwell_local/test_vipe_fit.sh` (waits for ckpts, samples peak VRAM).

## Stage 1 model checkpoints (sources, sizes, cache paths)

Pre-downloaded by `egox_blackwell_local/download_stage1_ckpts.sh` to the EXACT paths ViPE checks:

| Model | Source | Size | Cache path |
|---|---|---|---|
| VDA-Large vit-L | HF `depth-anything/Video-Depth-Anything-Large` → `video_depth_anything_vitl.pth` | 1467 MB | `~/.cache/torch/hub/checkpoints/` |
| VDA-Small vit-S (downgrade) | HF `…-Small` → `video_depth_anything_vits.pth` | 111 MB | same |
| MoGe-2 vit-L | HF `Ruicheng/moge-2-vitl-normal` → `model.pt` | 1324 MB | HF cache |
| GroundingDINO | HF `ShilongLiu/GroundingDINO` → `groundingdino_swint_ogc.pth` | 662 MB | `~/.cache/torch/hub/checkpoints/` |
| SAM vit_b | `dl.fbaipublicfiles.com/.../sam_vit_b_01ec64.pth` | 358 MB | `~/.cache/torch/hub/sam/` |
| GeoCalib pinhole | `github.com/cvg/GeoCalib …/geocalib-pinhole.tar` → saved as `pinhole.tar` | 111 MB | `~/.cache/torch/hub/geocalib/` |
| AOT R50_DeAOTL | gdrive (gdown, fuzzy) → `R50_DeAOTL_PRE_YTB_DAV.pth` | ~200 MB | `~/.cache/torch/hub/aot/` |

VDA/MoGe `.pth`/`.pt` are plain PyTorch `state_dict` pickles, loaded via
`torch.hub.load_state_dict_from_url(..., map_location="cpu")` / `from_pretrained`.
Note: ViPE also does small runtime HF pulls beyond these 7 (hence the hf_transfer=0 rule).

## ViPE intrinsics `.npz` — format + the synthesis trap

- **Format (confirmed via `core/finetune/datasets/utils.py:99` `iproj_disp`):** npz keys
  `['data','inds']`; `data` shape `(N_frames, 4)` = `[fx, fy, cx, cy]` per frame.
  The loader reads `data[0:1,:]` → `(1,4)` → `iproj_disp` does
  `fx,fy,cx,cy = intrinsics.unbind(dim=-1)`. **The `#! (3,3)` comment in
  `wan_dataset.py:339` is WRONG** — it's (N,4), not 3×3. ViPE's native output already
  matches; NO conversion needed.
- **Depth-model-independent:** intrinsics come from GeoCalib (calibration), unaffected by the
  VDA vit-S↔vit-L swap. So satya's vit-S-run `.npz` are valid for the faithful pipeline.

### USE the satya saved files — do NOT synthesize

- Saved at `/media/skr/storage/paper_reproduction/egoX/vitS/`:
  `vipe_intrinsics_egoexo4d_5clips.zip` (5 example clips) + `vipe_intrinsics_joker.zip`.
  Real GeoCalib output, correct `(50,4)` format, processed at 784×448 (cx=392, cy=224).
- **Synthesis is wrong:** naive scaling of the Ego-Exo4D 4K intrinsics (fx 1251.9 × 784/3840
  ≈ **256**) disagrees with real GeoCalib (cmu_soccer **fx≈349**) by ~37%. So
  `gen_stageB_inputs.py` should COPY the satya `.npz`, not synthesize them.
- Minor caveat: shipped depth maps are 448×**796** but intrinsics imply 784 wide (~12 px width
  mismatch — authors' depth vs satya's ViPE run). Negligible for a smoke test; for full
  consistency depth+intrinsics should come from one run.

### Caveat on the "validation" comparison

The fit-test run was launched WITH `--use_exo_intrinsic_gt`, so its saved `intrinsics/exo.npz`
just echoes the GT I passed (256), NOT a GeoCalib estimate — it does NOT prove vit-L↔vit-S
intrinsic equality. To actually confirm, re-run one clip WITHOUT `--use_exo_intrinsic_gt`
(let GeoCalib estimate) and compare to satya's `349.14`.

## Next steps

1. Wire satya `.npz` into `vipe_results/<take>/intrinsics/<best_camera>.npz` + write training
   meta (update `gen_stageB_inputs.py` to copy, not synthesize).
2. (Optional) GeoCalib re-run for the model-independence confirmation.
3. Stage 2 QLoRA code (bnb 4-bit + peft single-GPU) — still the real gating work.

#!/bin/bash
# Stage 1 (EgoX-EgoPriorRenderer / ViPE) env build — BLACKWELL (sm_120) variant.
# Mirrors build_stage1_env.sh (satya/RTX3060) but targets this machine:
#   - path auto-derived from this script's location
#   - TORCH_CUDA_ARCH_LIST=12.0+PTX (Blackwell) instead of 8.6 (Ampere)
# conda supplies ONLY the CUDA 12.8 toolchain (nvcc + eigen + dev libs);
# uv does all Python installs. Faithful to envs/base.yml + envs/requirements.txt
# (torch 2.7.0+cu128 — first torch with sm_120 support, so paper-pin == Blackwell-capable).
set -eo pipefail

# CRITICAL on this machine: ~/.local has a user-site torch 2.11.0+cu130 that shadows the
# env's pinned torch 2.7.0+cu128 — breaks pytorch3d build (CUDA 12.8 vs 13.0 mismatch).
# Disable user-site for the whole build so the env's own torch is used.
export PYTHONNOUSERSITE=1

# ~12 MB/s link: large CUDA wheels exceed uv's default 30s HTTP timeout. Give it room.
export UV_HTTP_TIMEOUT=600
export UV_CONCURRENT_DOWNLOADS=4

CONDA=~/miniconda3/bin/conda
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RENDER="$HERE/EgoX/EgoX-EgoPriorRenderer"
cd "$RENDER"
echo "renderer: $RENDER"

echo "=== [1/7] conda env create (toolchain: nvcc12.8 + eigen + dev libs) ==="
if "$CONDA" env list | grep -q egox-egoprior; then
  echo "env egox-egoprior already exists, skipping create"
else
  "$CONDA" env create -f envs/base.yml
fi
ENVDIR=$("$CONDA" env list | awk '/egox-egoprior/{print $NF}')
PY="$ENVDIR/bin/python"
export CUDA_HOME="$ENVDIR"
export CONDA_PREFIX="$ENVDIR"
export PATH="$ENVDIR/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="12.0+PTX"   # RTX PRO 4000 Blackwell (sm_120)
# conda CUDA 12.8 puts headers/libs under targets/, NOT $CUDA_HOME/include — expose them
# so nvcc/gcc find cuda_runtime_api.h during the pytorch3d / vipe_ext source builds.
export CPATH="$ENVDIR/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$ENVDIR/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
echo "env:    $ENVDIR"
echo "python: $PY"
echo "nvcc:   $("$ENVDIR/bin/nvcc" --version | tail -1)"
echo "arch:   $TORCH_CUDA_ARCH_LIST"

echo "=== [2/7] uv pip: pinned requirements (torch 2.7.0+cu128 etc.) ==="
uv pip install --python "$PY" -r envs/requirements.txt

echo "=== [3/7] FAIL-FAST: confirm torch sees Blackwell sm_120 BEFORE source builds ==="
"$PY" - <<'PYEOF'
import torch, sys
al = torch.cuda.get_arch_list()
print("torch", torch.__version__, "| arch list:", al)
if not any("120" in a or "sm_120" in a for a in al):
    print("FATAL: this torch build has NO sm_120 kernels — pytorch3d/vipe_ext will compile but fail at runtime.")
    print("       Bump torch to a cu128 build that lists sm_120, then re-run.")
    sys.exit(2)
print("OK: sm_120 present in torch arch list")
PYEOF

echo "=== [4/7] uv pip: pytorch3d v0.7.9 (FRAGILE source build, sm_120) ==="
uv pip install --python "$PY" --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"

echo "=== [5/7] uv pip: MoGe (source) ==="
uv pip install --python "$PY" --no-build-isolation \
    "git+https://github.com/microsoft/MoGe.git"

echo "=== [6/7] uv pip: build vipe_ext CUDA extension + install vipe (-e .) ==="
uv pip install --python "$PY" --no-build-isolation -e .

echo "=== [7/7] verify ==="
"$PY" -c "import torch, vipe, pytorch3d, moge; print('stage-1 env OK | torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
echo "=== BUILD COMPLETE ==="

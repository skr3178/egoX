#!/bin/bash
# Stage 1 (EgoX-EgoPriorRenderer / ViPE) env build.
# conda supplies ONLY the CUDA 12.8 toolchain (nvcc + eigen + cusparse/cublas/cusolver);
# uv does all Python package installs into that env. Faithful to envs/base.yml + envs/requirements.txt.
set -eo pipefail

CONDA=~/miniconda3/bin/conda
RENDER=/home/satya/skr/egoX/EgoX/EgoX-EgoPriorRenderer
cd "$RENDER"

echo "=== [1/6] conda env create (toolchain: nvcc12.8 + eigen + dev libs) ==="
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
export TORCH_CUDA_ARCH_LIST="8.6"   # RTX 3060 (Ampere, sm_86)
echo "env:    $ENVDIR"
echo "python: $PY"
echo "nvcc:   $("$ENVDIR/bin/nvcc" --version | tail -1)"

echo "=== [2/6] uv pip: pinned requirements (torch 2.7.0+cu128 etc.) ==="
uv pip install --python "$PY" -r envs/requirements.txt

echo "=== [3/6] uv pip: pytorch3d v0.7.9 (FRAGILE source build) ==="
uv pip install --python "$PY" --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"

echo "=== [4/6] uv pip: MoGe (source) ==="
uv pip install --python "$PY" --no-build-isolation \
    "git+https://github.com/microsoft/MoGe.git"

echo "=== [5/6] uv pip: build vipe_ext CUDA extension + install vipe (-e .) ==="
uv pip install --python "$PY" --no-build-isolation -e .

echo "=== [6/6] verify ==="
"$PY" -c "import vipe, pytorch3d, moge; print('stage-1 env OK')"
echo "=== BUILD COMPLETE ==="

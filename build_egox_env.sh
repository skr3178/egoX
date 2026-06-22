set -e
EGO=/media/skr/storage/conda_envs/egox
export PYTHONNOUSERSITE=1
export PIP_CONFIG_FILE=/dev/null
echo "=== [1/5] create conda env py3.10 ==="
~/miniconda3/bin/conda create -p $EGO python=3.10 -y
mkdir -p $EGO/etc/conda/activate.d
echo 'export PYTHONNOUSERSITE=1' > $EGO/etc/conda/activate.d/zz_nousersite.sh
PIP="$EGO/bin/pip --no-cache-dir"
echo "=== [2/5] torch 2.10.0 + torchvision 0.25.0 (cu128) ==="
$PIP install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
echo "=== [3/5] bitsandbytes 0.49.2 ==="
$PIP install bitsandbytes==0.49.2
echo "=== [4/5] EgoX deps (pinned) ==="
$PIP install diffusers==0.34.0 transformers==4.49.0 accelerate==1.5.2 peft==0.17.1 \
    sentencepiece ftfy imageio imageio-ffmpeg opencv-python-headless tyro decord huggingface_hub
echo "=== [5/5] VERIFY (user-site OFF) ==="
PYTHONNOUSERSITE=1 $EGO/bin/python - <<'PYEOF'
import torch, diffusers, transformers, peft, bitsandbytes, accelerate
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
print("diffusers", diffusers.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__, "| bnb", bitsandbytes.__version__, "| accelerate", accelerate.__version__)
print("torch from:", torch.__file__)
from diffusers import BitsAndBytesConfig, WanTransformer3DModel, AutoencoderKLWan
from transformers import UMT5EncoderModel, CLIPVisionModel
print("OK: all EgoX-required classes importable")
PYEOF
echo "=== BUILD COMPLETE ==="

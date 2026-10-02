#!/usr/bin/env bash
#SBATCH --job-name=absent-wan-setup
#SBATCH --account=rewetz
#SBATCH --qos=rewetz
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=12:00:00
#SBATCH --output=/blue/rewetz/will.english/absent-wan/logs/setup-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
base=/blue/rewetz/will.english/absent-wan
export CONDA_PKGS_DIRS=$base/conda-pkgs
export PIP_CACHE_DIR=$base/pip-cache
export HF_HOME=$base/hf-cache
export XDG_CACHE_HOME=$base/cache
export TMPDIR=$base/tmp
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR"

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
if [ ! -x "$base/env/bin/python" ]; then
  conda create -p "$base/env" --clone /blue/rewetz/will.english/absent-joyai/env -y
fi
conda activate "$base/env"
if [ ! -d "$base/VideoX-Fun/.git" ]; then
  git clone --depth 1 https://github.com/aigc-apps/VideoX-Fun.git "$base/VideoX-Fun"
fi
git -C "$base/VideoX-Fun" rev-parse HEAD > "$base/videox-fun-revision.txt"
python -m pip install -r "$base/VideoX-Fun/requirements.txt" 'huggingface_hub[cli]'
hf download alibaba-pai/Wan2.2-VACE-Fun-A14B --repo-type model \
  --local-dir "$base/VideoX-Fun/models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B"
python -c 'import torch, diffusers, transformers; print("torch", torch.__version__, "diffusers", diffusers.__version__, "transformers", transformers.__version__)'
echo WAN_SETUP_READY

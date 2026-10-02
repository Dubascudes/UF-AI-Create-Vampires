#!/usr/bin/env bash
#SBATCH --job-name=absent-joyai-setup
#SBATCH --account=rewetz
#SBATCH --qos=rewetz
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --output=/blue/rewetz/will.english/absent-joyai/logs/setup-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"

base=/blue/rewetz/will.english/absent-joyai
src=$base/src
envdir=$base/env
export CONDA_PKGS_DIRS=$base/conda-pkgs
export PIP_CACHE_DIR=$base/pip-cache
export HF_HOME=$base/hf-cache
export XDG_CACHE_HOME=$base/cache
export TMPDIR=$base/tmp
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR"
tar -xzf "$base/src.tar.gz" -C "$src"

module load conda
if [ ! -x "$envdir/bin/python" ]; then
  conda create -p "$envdir" python=3.10 -y
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$envdir"
python -m pip install -r "$src/deploy/requirements.txt" 'huggingface_hub[cli]'
conda install -c conda-forge cuda-nvcc=12.8 cuda-cudart-dev=12.8 \
  libcublas-dev libcusparse-dev libcusolver-dev -y

python -m pip install flash-attn-4==4.0.0b13 \
  nvidia-cutlass-dsl==4.5.1 quack-kernels==0.4.1 apache-tvm-ffi==0.1.12

if [ ! -d "$base/cutlass/.git" ]; then
  git clone https://github.com/NVIDIA/cutlass.git "$base/cutlass"
fi
git -C "$base/cutlass" checkout dcf215af
export CUDA_HOME=$envdir
export JOYOMNI_OPS_CUDA_ARCHS='100a;120a'
export JOYOMNI_OPS_CUTLASS_DIR=$base/cutlass
export MAX_JOBS=8
python -m pip install --no-build-isolation "$src/deploy/joyomni_ops"

ckpt=$src/deploy/deps/checkpoints
mkdir -p "$ckpt"
hf download jdopensource/JoyAI-Video-Edit --repo-type model \
  --local-dir "$ckpt/JoyAI-Video-Edit" \
  --include 'dit/joyai_video_edit_dit_0811.pth' 'vae/*'
hf download XiaomiMiMo/MiMo-VL-7B-RL-2508 --repo-type model \
  --local-dir "$ckpt/MiMo-VL-7B-RL-2508"

python -c 'import torch, joyomni_ops, flash_attn.cute; print("cuda", torch.version.cuda, "joyomni_ops", joyomni_ops.has_fp8())'
echo 'JOYAI_SETUP_READY'

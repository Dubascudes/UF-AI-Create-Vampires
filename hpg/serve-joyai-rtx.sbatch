#!/usr/bin/env bash
#SBATCH --job-name=absent-joyai-rtx
#SBATCH --account=rewetz
#SBATCH --qos=rewetz
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=12:00:00
#SBATCH --output=/blue/rewetz/will.english/absent-joyai/logs/serve-rtx-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"

base=/blue/rewetz/will.english/absent-joyai
export HF_HOME=$base/hf-cache
export XDG_CACHE_HOME=$base/cache
export TMPDIR=$base/tmp
export JOYOMNI_CACHE_ROOT=$base/cache-rtx6000
export JOYOMNI_WIDTH=840
export JOYOMNI_HEIGHT=480
export JOYOMNI_FPS=24
export JOYOMNI_HOST=0.0.0.0
export JOYOMNI_PORT=8080
export JOYOMNI_SKIP_LOAD_WARMUP=1
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR" "$JOYOMNI_CACHE_ROOT"

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$base/env"
cd "$base/src"
python -c 'import torch; print("GPU:", torch.cuda.get_device_name(0))'
export JOYOMNI_DEPLOY_DIR=$base/src/deploy
bash "$base/run-joyai-no-record.sh" --no-online-gate

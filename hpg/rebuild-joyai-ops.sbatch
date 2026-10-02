#!/usr/bin/env bash
#SBATCH --job-name=absent-joyai-ops
#SBATCH --account=rewetz
#SBATCH --qos=rewetz
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=02:00:00
#SBATCH --output=/blue/rewetz/will.english/absent-joyai/logs/ops-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"

base=/blue/rewetz/will.english/absent-joyai
export CONDA_PKGS_DIRS=$base/conda-pkgs
export PIP_CACHE_DIR=$base/pip-cache
export XDG_CACHE_HOME=$base/cache
export TMPDIR=$base/tmp
export CUDA_HOME=$base/env
export JOYOMNI_OPS_CUDA_ARCHS='100a;120a'
export JOYOMNI_OPS_CUTLASS_DIR=$base/cutlass
export MAX_JOBS=8
mkdir -p "$TMPDIR"

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$base/env"
python -m pip install --no-build-isolation --force-reinstall --no-deps "$base/src/deploy/joyomni_ops"
python -c 'import joyomni_ops; print("DUAL_ARCH_OPS_READY", joyomni_ops.has_fp8())'

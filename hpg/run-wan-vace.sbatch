#!/usr/bin/env bash
#SBATCH --job-name=absent-wan-vace
#SBATCH --account=rewetz
#SBATCH --qos=rewetz
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=04:00:00
#SBATCH --output=/blue/rewetz/will.english/absent-wan/logs/run-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
base=/blue/rewetz/will.english/absent-wan
export HF_HOME=$base/hf-cache
export XDG_CACHE_HOME=$base/cache
export TMPDIR=$base/tmp
export TOKENIZERS_PARALLELISM=false
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$base/env"
cd "$base/VideoX-Fun"
python "$base/configure-wan-vace-trial.py" \
  examples/wan2.2_vace_fun/predict_v2v_mask.py \
  examples/wan2.2_vace_fun/predict_absent_trial.py "$base/trial"
python -c 'import torch; print("GPU", torch.cuda.get_device_name(0))'
start=$(date +%s)
python examples/wan2.2_vace_fun/predict_absent_trial.py
end=$(date +%s)
echo "WAN_GENERATION_SECONDS=$((end-start))"
echo WAN_TRIAL_COMPLETE

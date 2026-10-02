#!/bin/bash
#SBATCH --job-name=absent-network-probe
#SBATCH --account=rewetz
#SBATCH --qos=rewetz
#SBATCH --partition=hpg-dev
#SBATCH --cpus-per-task=1
#SBATCH --mem=1gb
#SBATCH --time=00:20:00
#SBATCH --chdir=/blue/rewetz/will.english/absent-network-probe
#SBATCH --output=/blue/rewetz/will.english/absent-network-probe/probe-%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
export TMPDIR=/blue/rewetz/will.english/absent-network-probe/tmp
mkdir -p "$TMPDIR"
exec /usr/bin/python3 /blue/rewetz/will.english/absent-network-probe/echo_server.py

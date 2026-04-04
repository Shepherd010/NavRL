#!/usr/bin/env bash
# =============================================================================
# NavRL Curriculum Training Launcher
#
# Usage:
#   bash train_curriculum.sh                       # default: stage 1, auto-advance
#   bash train_curriculum.sh stage=2               # resume from stage 2
#   bash train_curriculum.sh headless=false         # with GUI
#   bash train_curriculum.sh curriculum.auto_advance=false  # manual stage switch
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default overrides for curriculum training
# mode=ppo  — CNN-PPO, no topo-transformer
# curriculum.enabled=true — activates the curriculum plugin
# use_topo=false — skip topology extraction
python "${SCRIPT_DIR}/train.py" \
    mode=ppo \
    use_topo=false \
    curriculum.enabled=true \
    wandb.name=curriculum \
    "$@"

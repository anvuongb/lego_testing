#!/bin/bash
# Train the v4 Lego generation model with curriculum learning
# Usage: ./train_v4.sh [name] [stages]

NAME=${1:-lego_v4_$(date +%s)}
STAGES=${2:-"3,4,5,6,8,10"}

cd "$(dirname "$0")"
.venv/bin/python simple_agent_masked_ppo_v4.py \
    --name "$NAME" \
    --stages "$STAGES"

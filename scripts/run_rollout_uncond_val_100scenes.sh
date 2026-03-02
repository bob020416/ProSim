#!/bin/bash
# Runs unconditional validation rollout on 100 scenarios.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR/.."

bash scripts/run_prosim.sh prosim/rollout/run_distributed_rollout.py \
  --config prosim_demo/cfg/no_text.yaml,prosim_demo/cfg/uncond_val_100scenes.yaml \
  --ckpt checkpoints/prosim_demo_model.ckpt \
  --rollout_name wosac_val_uncond_100scenes_debug2 \
  --save_vis false \
  --save_metric true \
  --cluster local

#!/bin/bash
# Runs unconditional validation rollout using the prosim demo model.
# Equivalent to the "rollout: uncond_val" VSCode launch configuration.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR/.."

bash scripts/run_prosim.sh prosim/rollout/run_distributed_rollout.py \
  --config prosim_demo/cfg/no_text.yaml,prosim_demo/cfg/uncond_val.yaml \
  --ckpt checkpoints/prosim_demo_model.ckpt \
  --rollout_name 0301wosac_val_random10_uncond_no_motion_tags_new_skipped \
  --save_vis false \
  --save_metric true \
  --cluster local

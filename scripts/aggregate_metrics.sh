#!/bin/bash

# Aggregate metrics from rollout directory
# Usage: ./aggregate_metrics.sh <rollout_root_dir>
# Example: ./aggregate_metrics.sh results/paper/.../wosac_val_random10_uncond_32_top_3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <rollout_root_directory>"
    echo "Example: $0 results/paper/prompt_condition/waymo/mixture_control_no_priority/goal_tag_dragpoint_text_fix_1.0_uncond_val/rollout/wosac_val_random10_uncond_32_top_3"
    exit 1
fi

ROLLOUT_ROOT="$1"

if [ ! -d "$ROLLOUT_ROOT/metrics" ]; then
    echo "Error: metrics directory not found at $ROLLOUT_ROOT/metrics"
    exit 1
fi

echo "Aggregating metrics from: $ROLLOUT_ROOT"
echo ""

python prosim/rollout/report_metrics.py --root "$ROLLOUT_ROOT"

echo ""
echo "Done! Check these files:"
echo "  - $ROLLOUT_ROOT/metrics_report.json"
echo "  - $ROLLOUT_ROOT/metrics_summary.txt"

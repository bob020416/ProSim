"""
Build the trajdata cache for waymo_val from raw tfrecords.

Raw tfrecords expected at:
  /media/msc-auto/HDD/dataset/waymo_tfrecord_v1_3/validation/

Cache will be written to:
  /home/msc-auto/wjchang/ProSim/trajdata_cache2/waymo_val/

Usage:
  python build_cache.py
  python build_cache.py --num_workers 4   # parallel scene processing
"""

import sys
import os
import argparse
import multiprocessing

# Must be set before any CUDA/TF import - forked workers can't init CUDA
multiprocessing.set_start_method('spawn', force=True)

sys.path.append(os.getcwd())

argparser = argparse.ArgumentParser()
argparser.add_argument("--num_workers", type=int, default=0, help="Number of parallel workers for scene processing (0 = serial)")
args = argparser.parse_args()

from prosim.config.default import Config, get_config
from prosim.core.registry import registry

# Load base config with waymo_val as the rollout source
config = get_config('prosim_demo/cfg/no_text.yaml', cluster='local')

config.defrost()
config.DATASET.SOURCE.ROLLOUT = ['waymo_train']
config.DATASET.SKIP_CACHE_CHECK = False
config.DATASET.CACHE_MAP = True
# Force single-process: TF cannot initialize CUDA in forked worker processes
config.ROLLOUT.NUM_WORKERS = 0
config.freeze()

print(f"Cache dir : {config.DATASET.CACHE_PATH}")
print(f"Waymo val : {config.DATASET.DATA_PATHS.WAYMO_VAL}")
print(f"Building waymo_val cache (num_workers={args.num_workers})...")

dataset = registry.get_dataset('prosim_imitation')(config, 'rollout', centric='scene')

print(f"Done. {dataset.num_scenes()} scenes cached.")

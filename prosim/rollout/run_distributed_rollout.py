import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

argparser = argparse.ArgumentParser()
argparser.add_argument("--config", type=str, required=True)
argparser.add_argument("--ckpt", type=str, required=True)
argparser.add_argument('--rollout_name', type=str, required=True)
argparser.add_argument('--save_metric', type=str2bool, default=True)
argparser.add_argument('--save_rollout', type=str2bool, default=True)
argparser.add_argument('--cluster', type=str, default='local')
argparser.add_argument("--M", type=int, default=32)
argparser.add_argument("--action_noise_std", type=float, default=0.0)
argparser.add_argument("--traj_noise_std", type=float, default=0.0)
argparser.add_argument("--top_k", type=int, default=3)
argparser.add_argument("--smooth_dist", type=float, default=5.0)
argparser.add_argument("--sampler_cfg", type=str, default=None)
argparser.add_argument('--save_vis', type=str2bool, default=False)
argparser.add_argument("--vis_interval", type=int, default=0)
argparser.add_argument("--vis_max_scenes", type=int, default=0)
argparser.add_argument("--vis_fps", type=int, default=10)

args = argparser.parse_args()

from prosim.core.registry import registry
from prosim.config.default import Config, get_config
from prosim.rollout.distributed_utils import rollout_scene_distributed

print(args.cluster)

print('save_metric: ', args.save_metric)
print('save_rollout: ', args.save_rollout)
print('save_vis: ', args.save_vis)

config = get_config(args.config, cluster=args.cluster)
rollout_scene_distributed(
    config,
    args.M,
    args.ckpt,
    args.rollout_name,
    args.save_metric,
    args.save_rollout,
    args.top_k,
    args.traj_noise_std,
    args.action_noise_std,
    args.sampler_cfg,
    args.smooth_dist,
    args.save_vis,
    args.vis_interval,
    args.vis_max_scenes,
    args.vis_fps,
)

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset
from waymo_open_dataset.protos import sim_agents_submission_pb2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from prosim.config.default import get_config
from prosim.core.registry import registry
from prosim.rollout.waymo_utils import _add_waymo_map, get_waymo_file_template, get_waymo_scene_object


def load_waymo_scene(config, scene_idx):
  dataset = registry.get_dataset("prosim_imitation")(config, "rollout", centric="scene")
  data_loader = DataLoader(
    Subset(dataset, [scene_idx]),
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=dataset.get_collate_fn(),
  )
  batch = next(iter(data_loader))
  scene_name = batch.scene_ids[0]
  scene_template = get_waymo_file_template(config)
  waymo_scene = get_waymo_scene_object(scene_name, scene_template)
  return waymo_scene


def load_rollout_pb(pb_path):
  rollout = sim_agents_submission_pb2.ScenarioRollouts()
  rollout.ParseFromString(Path(pb_path).read_bytes())
  return rollout


def plot_rollout_bundle(waymo_scene, scenario_rollouts, output_path, plot_gt=False):
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=150)
  _add_waymo_map(ax, waymo_scene)

  colors = plt.cm.get_cmap("turbo", max(len(scenario_rollouts.joint_scenes), 1))

  for rollout_idx, joint_scene in enumerate(scenario_rollouts.joint_scenes):
    color = colors(rollout_idx)
    for traj in joint_scene.simulated_trajectories:
      x = np.asarray(traj.center_x, dtype=np.float32)
      y = np.asarray(traj.center_y, dtype=np.float32)
      if not np.isfinite(x).all() or not np.isfinite(y).all():
        continue
      ax.plot(x, y, color=color, linewidth=0.9, alpha=0.18, zorder=2)
      ax.scatter(x[-1], y[-1], color=color, s=4, alpha=0.35, zorder=3)

  if plot_gt:
    sim_agent_ids = set(traj.object_id for traj in scenario_rollouts.joint_scenes[0].simulated_trajectories)
    for track in waymo_scene.tracks:
      if track.id not in sim_agent_ids:
        continue
      valid = np.array([state.valid for state in track.states])
      if not valid.any():
        continue
      x = np.array([state.center_x for state in track.states])[valid]
      y = np.array([state.center_y for state in track.states])[valid]
      ax.plot(x, y, color="black", linewidth=1.2, alpha=0.4, linestyle="--", zorder=4)

  ax.set_title(
    f"Scenario {scenario_rollouts.scenario_id} | {len(scenario_rollouts.joint_scenes)} rollouts overlay"
  )
  fig.tight_layout()
  fig.savefig(output_path, bbox_inches="tight")
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", required=True, help="Comma-separated config path(s).")
  parser.add_argument("--rollout_root", required=True, help="Rollout root containing rollout/scene_<idx>.pb")
  parser.add_argument("--scene_idx", required=True, type=int, help="Dataset scene index, e.g. 0 for rollout/scene_0.pb")
  parser.add_argument("--output", default=None, help="Optional output PNG path.")
  parser.add_argument("--plot_gt", action="store_true", help="Overlay GT sim-agent trajectories as dashed black lines.")
  args = parser.parse_args()

  config = get_config(args.config, cluster="local")
  rollout_root = Path(args.rollout_root)
  pb_path = rollout_root / "rollout" / f"scene_{args.scene_idx}.pb"

  if args.output is None:
    output_path = rollout_root / "debug_vis" / f"scene_{args.scene_idx}_bundle.png"
  else:
    output_path = Path(args.output)

  scenario_rollouts = load_rollout_pb(pb_path)
  waymo_scene = load_waymo_scene(config, args.scene_idx)
  plot_rollout_bundle(waymo_scene, scenario_rollouts, output_path, plot_gt=args.plot_gt)

  print(f"Saved overlay plot to {output_path}")


if __name__ == "__main__":
  main()

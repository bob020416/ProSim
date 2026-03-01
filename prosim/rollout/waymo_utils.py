# Imports
import os
import tarfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from pathlib import Path
import tqdm

from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2
from waymo_open_dataset.utils.sim_agents import submission_specs


def _polyline_xy(points):
  if len(points) == 0:
    return None
  xy = np.array([[p.x, p.y] for p in points], dtype=np.float32)
  if xy.size == 0:
    return None
  return xy


def _polygon_xy(points):
  xy = _polyline_xy(points)
  if xy is None or len(xy) == 0:
    return None
  if not np.allclose(xy[0], xy[-1]):
    xy = np.concatenate([xy, xy[:1]], axis=0)
  return xy


def _add_waymo_map(ax, scenario):
  for feature in scenario.map_features:
    feature_type = feature.WhichOneof('feature_data')

    if feature_type == 'lane':
      xy = _polyline_xy(feature.lane.polyline)
      if xy is not None:
        ax.plot(xy[:, 0], xy[:, 1], color='#bdbdbd', linewidth=0.8, alpha=0.7, zorder=1)
    elif feature_type == 'road_line':
      xy = _polyline_xy(feature.road_line.polyline)
      if xy is not None:
        ax.plot(xy[:, 0], xy[:, 1], color='#9e9e9e', linewidth=0.8, alpha=0.7, zorder=1)
    elif feature_type == 'road_edge':
      xy = _polyline_xy(feature.road_edge.polyline)
      if xy is not None:
        ax.plot(xy[:, 0], xy[:, 1], color='#616161', linewidth=1.0, alpha=0.8, zorder=1)
    elif feature_type == 'crosswalk':
      xy = _polygon_xy(feature.crosswalk.polygon)
      if xy is not None:
        ax.fill(xy[:, 0], xy[:, 1], color='#eeeeee', alpha=0.5, zorder=0)
    elif feature_type == 'speed_bump':
      xy = _polygon_xy(feature.speed_bump.polygon)
      if xy is not None:
        ax.fill(xy[:, 0], xy[:, 1], color='#d7ccc8', alpha=0.6, zorder=0)
    elif feature_type == 'driveway':
      xy = _polygon_xy(feature.driveway.polygon)
      if xy is not None:
        ax.fill(xy[:, 0], xy[:, 1], color='#f5f5f5', alpha=0.35, zorder=0)

  ax.set_aspect("equal", adjustable="box")
  ax.axis("off")


def get_waymo_file_template(config):
  source_data = config.DATASET.SOURCE.ROLLOUT[0]

  data_path = config.DATASET.DATA_PATHS[source_data.upper()]
  split = source_data.split('_')[-1]
  if split == 'val':
    waymo_split = 'validation_splitted'
  elif split == 'test':
    waymo_split = 'testing_splitted'
  elif split == 'train':
    waymo_split = 'training_splitted'

  scene_template = os.path.join(data_path, waymo_split, waymo_split + '_{}.tfrecords')

  return scene_template

def get_waymo_scene_object(scene_name, scene_template):
  scene_id = scene_name.split('/')[-1].split('_')[-1]
  file_path = scene_template.format(scene_id)

  file_path = Path(file_path)
  file_path = os.path.expanduser(file_path)
  
  print('loading scene: ', file_path)
  print(os.path.exists(file_path))

  # filenames = tf.io.matching_files(VALIDATION_FILES)
  # Define the dataset from the TFRecords.
  filenames = [file_path]
  waymo_dataset = tf.data.TFRecordDataset(filenames)
  # Since these are raw Scenario protos, we need to parse them in eager mode.
  dataset_iterator = waymo_dataset.as_numpy_iterator()
  bytes_example = next(dataset_iterator)
  scenario = scenario_pb2.Scenario.FromString(bytes_example)

  return scenario

def joint_scene_from_states(
    states, object_ids
    ):
  # States shape: (num_objects, num_steps, 4).
  # Objects IDs shape: (num_objects,).
  simulated_trajectories = []
  for i_object in range(len(object_ids)):
    simulated_trajectories.append(sim_agents_submission_pb2.SimulatedTrajectory(
        center_x=states[i_object, :, 0], center_y=states[i_object, :, 1],
        center_z=states[i_object, :, 2], heading=states[i_object, :, 3],
        object_id=object_ids[i_object]
    ))
  return sim_agents_submission_pb2.JointScene(
      simulated_trajectories=simulated_trajectories)

def rollout_states_to_joint_scene(waymo_scene, rollout_global_states, control_agents, ego_agent_id, start_frame):
  joint_trajs = []

  for agent in control_agents:
    full_trajs = [state[agent].as_format('x,y,h')[None, :] for state in rollout_global_states]
    full_trajs = np.concatenate(full_trajs, axis=0)
    joint_trajs.append(full_trajs)

  joint_trajs = np.stack(joint_trajs, axis=0)

  # z = 0.0
  # z_trajs = np.ones_like(joint_trajs[..., :1]) * z

  control_agents[0] = str(ego_agent_id)
  control_ids = [int(agent) for agent in control_agents]
  
  # use the z from the first frame of the waymo scene
  scene_track_ids = [track.id for track in waymo_scene.tracks]
  track_indices = [scene_track_ids.index(int(agent)) for agent in control_agents]
  z_start = np.array([waymo_scene.tracks[idx].states[start_frame].center_z for idx in track_indices])
  z_trajs = np.ones_like(joint_trajs[..., :1]) * z_start[:, None, None]

  joint_trajs = np.concatenate([joint_trajs[..., :2], z_trajs, joint_trajs[..., 2:]], axis=-1)


  simulated_states = tf.convert_to_tensor(joint_trajs)
  # control_ids = tf.convert_to_tensor(control_ids)
  joint_scene = joint_scene_from_states(simulated_states, control_ids)

  return joint_scene

def plot_waymo_gt_trajectory(scenario, start_step: int, end_step: int) -> None:
  # Plot their tracks.
  fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
  _add_waymo_map(ax, scenario)

  def plot_track_trajectory(track: scenario_pb2.Track, start_step: int, end_step: int) -> None:
    valid_mask = np.array([state.valid for state in track.states])
    time_steps = np.arange(len(valid_mask))
    time_mask = (start_step <= time_steps) & (time_steps <= end_step)
    mask = valid_mask & time_mask

    if np.any(mask):
      x = np.array([state.center_x for state in track.states])
      y = np.array([state.center_y for state in track.states])
      ax.plot(x[mask], y[mask], linewidth=5)

  for track in scenario.tracks:
    if track.id in submission_specs.get_sim_agent_ids(scenario):
      plot_track_trajectory(track, start_step, end_step)

  fig.canvas.draw()
  image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

  return image

def plot_waymo_rollout_trajectory(scenario, joint_scene) -> None:
  # Plot their tracks.
  fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
  _add_waymo_map(ax, scenario)

  def plot_sim_track(ax, track) -> None:
    x = track.center_x
    y = track.center_y
    ax.plot(x, y, linewidth=5)


  for track in joint_scene.simulated_trajectories:
    plot_sim_track(ax, track)

  fig.canvas.draw()
  image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

  return image


def save_waymo_rollout_gif(scenario, joint_scene, save_path, fps=10):
  save_path = Path(save_path)
  save_path.parent.mkdir(parents=True, exist_ok=True)

  trajs = joint_scene.simulated_trajectories
  if len(trajs) == 0:
    raise ValueError("joint_scene has no simulated trajectories")

  num_steps = len(trajs[0].center_x)
  colors = plt.cm.get_cmap("tab20", max(len(trajs), 1))
  frames = []

  for step in range(num_steps):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    _add_waymo_map(ax, scenario)

    for idx, track in enumerate(trajs):
      color = colors(idx % colors.N)
      x = np.asarray(track.center_x[: step + 1])
      y = np.asarray(track.center_y[: step + 1])
      ax.plot(x, y, linewidth=2, color=color, alpha=0.9)
      ax.scatter(x[-1], y[-1], s=20, color=color)

    ax.set_title(f"scenario {scenario.scenario_id} step {step + 1}/{num_steps}")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(image)
    plt.close(fig)

  duration = 1.0 / max(fps, 1)
  imageio.mimsave(save_path, frames, duration=duration, loop=0)

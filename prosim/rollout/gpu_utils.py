import torch
import numpy as np
import re

from prosim.dataset.format_utils import BatchDataDict, InputMaskData
from prosim.rollout.waymo_utils import get_waymo_file_template, get_waymo_scene_object, joint_scene_from_states, plot_waymo_gt_trajectory, rollout_states_to_joint_scene, plot_waymo_rollout_trajectory, save_waymo_rollout_gif
from prosim.models.utils.geometry import wrap_angle, batch_rotate_2D
from prosim.rollout.utils import batch_nd_transform_points_pt, batch_nd_transform_angles_pt
from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.protos import sim_agents_submission_pb2

def _remap_condition_to_valid_agents(condition_data, valid_idx):
  idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_idx)}

  for cond_type, cond in condition_data.all_cond.items():
    if 'prompt_mask' in cond and isinstance(cond['prompt_mask'], torch.Tensor):
      cond['prompt_mask'] = cond['prompt_mask'][:, valid_idx]

    if 'prompt_idx' in cond and isinstance(cond['prompt_idx'], torch.Tensor):
      old_prompt_idx = cond['prompt_idx']
      new_prompt_idx = torch.full_like(old_prompt_idx, -1)

      for old_idx, new_idx in idx_map.items():
        new_prompt_idx[old_prompt_idx == old_idx] = new_idx

      valid_prompt_refs = old_prompt_idx >= 0
      remap_ok = (~valid_prompt_refs) | (new_prompt_idx >= 0)
      cond['mask'] = cond['mask'] & remap_ok.all(dim=-1)
      cond['prompt_idx'] = new_prompt_idx

    if cond_type == 'llm_text_OneText' and isinstance(cond.get('input'), list):
      remapped_inputs = []
      for text in cond['input']:
        remapped_lines = []
        for line in text.split('\n'):
          if not line:
            continue

          placeholder_ids = [int(match) for match in re.findall(r'<A(\d+)>', line)]
          if any(old_idx not in idx_map for old_idx in placeholder_ids):
            continue

          for old_idx, new_idx in idx_map.items():
            line = line.replace(f'<A{old_idx}>', f'<A{new_idx}>')

          remapped_lines.append(line)

        remapped_inputs.append('\n'.join(remapped_lines))

      cond['input'] = remapped_inputs
      if 'mask' in cond and isinstance(cond['mask'], torch.Tensor):
        text_valid = torch.tensor([len(text) > 0 for text in remapped_inputs], device=cond['mask'].device, dtype=torch.bool)
        cond['mask'] = cond['mask'] & text_valid

def get_waymo_specification(batch, config):
  scene_name = batch.scene_ids[0]

  scene_template = get_waymo_file_template(config)
  waymo_scene = get_waymo_scene_object(scene_name, scene_template)
  sim_agent_ids = submission_specs.get_sim_agent_ids(waymo_scene)

  batch_prompt_info = batch.extras['prompt']['motion_pred']
  batch_agent_ids = batch_prompt_info['agent_ids'][0]

  valid_idx = []
  for i, agent_id in enumerate(batch_agent_ids):
    if 'ego' in agent_id or int(agent_id) in sim_agent_ids:
      valid_idx.append(i)
  
  print('batch_agent_ids: ', batch_agent_ids)
  print('valid_idx: ', valid_idx)
  print('sim_agent_ids: ', sim_agent_ids)
  print(len(valid_idx), len(sim_agent_ids))

  assert len(valid_idx) == len(sim_agent_ids)

  batch_agent_ids = [batch_agent_ids[i] for i in valid_idx]
  missed_agent_ids = [sim_agent_id for sim_agent_id in sim_agent_ids if str(sim_agent_id) not in batch_agent_ids]
  
  assert len(missed_agent_ids) == 1

  ego_sim_agent_id = missed_agent_ids[0]

  # subsample the batch_prompt info to only include the sim_agent_ids
  for key in batch_prompt_info:
    if type(batch_prompt_info[key]) == list:
      batch_prompt_info[key][0] = [batch_prompt_info[key][0][i] for i in valid_idx]
    elif type(batch_prompt_info[key]) == torch.Tensor:
      batch_prompt_info[key] = batch_prompt_info[key][:, valid_idx]

  if 'condition' in batch.extras and len(batch.extras['condition']) > 0:
    _remap_condition_to_valid_agents(batch.extras['condition'], valid_idx)
  
  # if len(missed_agent_ids) > 0:
  #   print('missed agent ids: ', missed_agent_ids)

  #   invalid_tracks = [track for track in waymo_scene.tracks if track.id in missed_agent_ids]
  #   for track in invalid_tracks:
  #     print(track.id)
  #     valid_state = [state for state in track.states if state.valid]
  #     print('num valid states: ', len(valid_state))
  
  return batch, waymo_scene, ego_sim_agent_id

def replica_batch_for_parallel_rollout(scene_embs, policy_emds, prompt_encs, policy_agent_ids, agent_trajs, batch, M):
  # replicate scene embs
  scene_embs_M = {}
  for name in ['obs_mask', 'map_mask', 'scene_pos', 'scene_ori', 'scene_tokens']:
    scene_embs_M[name] = scene_embs[name].repeat(M, 1)
  scene_embs_M['max_map_num'] = scene_embs['max_map_num']
  scene_embs_M['max_agent_num'] = scene_embs['max_agent_num']
  scene_embs_M['scene_type'] = scene_embs['scene_type'].repeat(M)
  N = scene_embs['scene_batch_idx'].shape[0]
  device = scene_embs['scene_batch_idx'].device
  scene_embs_M['scene_batch_idx'] = torch.arange(M, device=device)[:, None].repeat(1, N).reshape(-1)

  if policy_emds is None:
    policy_emds_M = None
  else:
    policy_emds_M = {'motion_pred': {}}

    for name in policy_emds['motion_pred'].keys():
      value = policy_emds['motion_pred'][name]
      if isinstance(value, torch.Tensor):
        dim_num = value.ndim
        reshape_dims = [M] + [1] * (dim_num - 1)
        policy_emds_M['motion_pred'][name] = value.repeat(*reshape_dims)
      else:
        policy_emds_M['motion_pred'][name] = value
  

  # replicate policy_agent_ids
  policy_agent_ids_M = {}
  policy_agent_ids_M['motion_pred'] = [policy_agent_ids['motion_pred'][0]] * M

  # replicate agent_trajs
  agent_trajs_M = {'motion_pred': {}}
  agent_trajs_M['motion_pred']['traj'] = agent_trajs['motion_pred']['traj'].repeat(M, 1, 1, 1)
  agent_trajs_M['motion_pred']['init_pos'] = agent_trajs['motion_pred']['init_pos'].repeat(M, 1, 1)
  agent_trajs_M['motion_pred']['init_heading'] = agent_trajs['motion_pred']['init_heading'].repeat(M, 1, 1)
  agent_trajs_M['motion_pred']['last_step'] = agent_trajs['motion_pred']['last_step']

  if 'vel' in agent_trajs['motion_pred']:
    agent_trajs_M['motion_pred']['vel'] = agent_trajs['motion_pred']['vel'].repeat(M, 1, 1, 1)

  # replicate prompt_encs
  prompt_encs_M = {'motion_pred': {}}
  prompt_encs_M['motion_pred']['prompt'] = prompt_encs['motion_pred']['prompt'].repeat(M, 1, 1)
  prompt_encs_M['motion_pred']['prompt_mask'] = prompt_encs['motion_pred']['prompt_mask'].repeat(M, 1)
  prompt_encs_M['motion_pred']['position'] = prompt_encs['motion_pred']['position'].repeat(M, 1, 1)
  prompt_encs_M['motion_pred']['heading'] = prompt_encs['motion_pred']['heading'].repeat(M, 1, 1)
  prompt_encs_M['motion_pred']['agent_type'] = prompt_encs['motion_pred']['agent_type'].repeat(M, 1)
  prompt_encs_M['motion_pred']['prompt_emd'] = prompt_encs['motion_pred']['prompt_emd'].repeat(M, 1, 1)
  prompt_encs_M['motion_pred']['agent_ids'] = prompt_encs['motion_pred']['agent_ids'][0] * M

  # replicate batch['extras']['fut_obs']
  batch_fut_obs_M = {}
  for step in batch.extras['fut_obs'].keys():
    fut_obs = batch.extras['fut_obs'][step]
    fut_obs_M = {}
    for key in fut_obs.keys():
      if type(fut_obs[key]) == list:
        fut_obs_M[key] = fut_obs[key] * M
      elif type(fut_obs[key]) == torch.Tensor:
        ndim = fut_obs[key].ndim
        repeat_dims = [M] + [1] * (ndim -1)
        fut_obs_M[key] = fut_obs[key].repeat(*repeat_dims)
    
    batch_fut_obs_M[step] = InputMaskData.from_dict(fut_obs_M)

  batch.extras['fut_obs'] = BatchDataDict(batch_fut_obs_M)

  return scene_embs_M, policy_emds_M, prompt_encs_M, policy_agent_ids_M, agent_trajs_M, batch

def sample_M_goal_cond_to_batch(batch, sample_result, top_K, M, stop_smooth_num=5.0):
  # assume sample_result is with 1 scene (bz=1)
  device = batch.extras['prompt']['motion_pred']['prompt'].device
  
  goal_inputs_M = []
  prompt_idxs_M = []

  print('smoothing stopping action with distance: ', stop_smooth_num)
  
  for b in range(M):
      goal_inputs_b = []
      prompt_idxs_b = []
      
      for pidx, aname in enumerate(batch.extras['prompt']['motion_pred']['agent_ids'][0]):
          pair_name = f'0-{aname}-0'
          if pair_name in sample_result['motion_pred']['pair_names']:
              pred_idx = sample_result['motion_pred']['pair_names'].index(pair_name)
              pred_goal_K = sample_result['motion_pred']['goal_point'][pred_idx]
              pred_goal_K_prob = sample_result['motion_pred']['goal_prob'][pred_idx]
      
              top_k_idx = torch.argsort(-pred_goal_K_prob)[:top_K]
              
              select_idx = top_k_idx[torch.randperm(top_K)[0]]
              select_goal = pred_goal_K[select_idx]

              if torch.abs(select_goal[0]) < stop_smooth_num and torch.abs(select_goal[1]) < stop_smooth_num:
                select_goal[0] = 0.0
                select_goal[1] = 0.0
      
              goal_inputs_b.append(torch.tensor([select_goal[0], select_goal[1], 80.0]))
              prompt_idxs_b.append(torch.tensor([pidx]))
      
      goal_inputs_b = torch.stack(goal_inputs_b)
      prompt_idxs_b = torch.stack(prompt_idxs_b)
  
      goal_inputs_M.append(goal_inputs_b)
      prompt_idxs_M.append(prompt_idxs_b)
  
  goal_inputs_M = torch.stack(goal_inputs_M).to(device)
  prompt_idxs_M = torch.stack(prompt_idxs_M).to(device)
  
  N = goal_inputs_M.shape[1]
  
  mask_M = torch.ones(M, N, dtype=torch.bool).to(device)
  prompt_mask_M = torch.ones(M, N, dtype=torch.bool).to(device)
  
  caption_str = 'show as green cross'
  
  goal_cond_M = {'input': goal_inputs_M, 'mask': mask_M, 'prompt_idx': prompt_idxs_M, 'prompt_mask': prompt_mask_M, 'caption_str': caption_str}
  
  batch.extras['condition'].all_cond = {'goal': goal_cond_M}

  return batch

def parallel_rollout_batch(batch, M, model, top_K=3, sampler_model=None, smooth_dist=5.0):
  """Run M independent forward passes and stack results.

  Aligned with reference: each rollout is a full independent model.forward()
  call so that any stochastic sampling produces diverse rollouts.
  """
  with torch.no_grad():
    # step_env writes rollout predictions back into batch.extras['fut_obs'] in-place
    # (fut_obs['input'], ['position'], ['heading'], ['mask']). Snapshot the tensors
    # before the loop so every rollout starts from the original observations.
    fut_obs_snapshot = {
      t: {k: v.clone() if isinstance(v, torch.Tensor) else v
          for k, v in obs.items()}
      for t, obs in batch.extras['fut_obs'].items()
    }

    list_result_M = []
    for i in range(M):
      # Restore fut_obs to its original state before each independent rollout.
      for t, obs_snapshot in fut_obs_snapshot.items():
        for k, v in obs_snapshot.items():
          if isinstance(v, torch.Tensor):
            batch.extras['fut_obs'][t][k].copy_(v)

      result_M = model.forward(batch, 'val')['motion_pred']
      list_result_M.append(result_M)

    # Stack trajectories across M rollouts
    combined_result = {'motion_pred': {'rollout_trajs': {}}}

    first_result = list_result_M[0]
    for key in first_result['rollout_trajs'].keys():
      trajs = [r['rollout_trajs'][key]['traj'] for r in list_result_M]
      stacked_trajs = torch.stack(trajs, dim=0)  # (M, T, 4)
      init_pos = [r['rollout_trajs'][key]['init_pos'] for r in list_result_M]
      init_heading = [r['rollout_trajs'][key]['init_heading'] for r in list_result_M]
      stacked_init_pos = torch.stack(init_pos, dim=0)
      stacked_init_heading = torch.stack(init_heading, dim=0)
      combined_result['motion_pred']['rollout_trajs'][key] = {
        'traj': stacked_trajs,
        'init_pos': stacked_init_pos,
        'init_heading': stacked_init_heading,
      }

    return combined_result

def obtain_rollout_trajs_in_world(batch, result_M, noise_std=0.0):
  """Convert rollout results to world coordinates.

  Aligned with reference: handles stacked (A, M, T, 4) trajectory format
  and returns (M, A, T, 3) numpy array with a single object_ids list.
  """
  batch_ids = []
  object_ids = []

  trajs = []
  init_pos = []
  init_heads = []

  for agent_name, results in result_M['motion_pred']['rollout_trajs'].items():
    batch_id = int(agent_name.split('-')[0])
    object_id = agent_name.split('-')[1]

    batch_ids.append(batch_id)
    object_ids.append(object_id)
    trajs.append(results['traj'])
    init_pos.append(results['init_pos'])
    init_heads.append(results['init_heading'])

  batch_ids = torch.tensor(batch_ids)
  trajs = torch.stack(trajs, axis=0)      # (A, M, T, 4)
  init_pos = torch.stack(init_pos, axis=0)      # (A, M, 2)
  init_heads = torch.stack(init_heads, axis=0)  # (A, M, 1)

  print('trajs shape: ', trajs.shape)
  A, M, T, _ = trajs.shape

  if noise_std > 0.0:
    print('WARNING: adding noise to the rollout trajectories with std: ', noise_std)
    trajs[:, :, :, :2] += torch.randn_like(trajs[:, :, :, :2]) * noise_std

  trajs = trajs.reshape(-1, T, 4)       # (AM, T, 4)
  init_pos = init_pos.reshape(-1, 2)    # (AM, 2)
  init_heads = init_heads.reshape(-1, 1) # (AM, 1)

  # transform all trajectories to the starting frame's coordinate system
  xys_in_center = batch_rotate_2D(trajs[:, :, :2], init_heads) + init_pos[:, None]
  hs = torch.arctan2(trajs[:, :, 2], trajs[:, :, 3])
  hs_in_centers = wrap_angle(hs + init_heads)

  # transform all trajectories to the world coordinate system
  center_to_world_tf = batch.centered_world_from_agent_tf[0]

  xys_in_world = batch_nd_transform_points_pt(xys_in_center, center_to_world_tf)
  hs_in_world = batch_nd_transform_angles_pt(hs_in_centers, center_to_world_tf)

  rollout_trajs_in_world = torch.cat([xys_in_world, hs_in_world[:, :, None]], axis=-1)
  rollout_trajs_in_world = rollout_trajs_in_world.reshape(A, M, T, 3)

  object_ids_M = []
  for batch_id in set(batch_ids.tolist()):
    batch_mask = batch_ids == batch_id
    object_ids_M.append([object_ids[i] for i in range(len(object_ids)) if batch_mask[i]])

  # return (M, A, T, 3) numpy array and single object_ids list
  return rollout_trajs_in_world.permute(1, 0, 2, 3).detach().cpu().numpy(), object_ids_M[0]

def joint_scene_from_rollout(waymo_scene, rollout_trajs, object_ids, ego_sim_agent_id):
  joint_trajs = rollout_trajs

  ego_idx = object_ids.index('ego')
  control_ids = object_ids.copy()
  control_ids[ego_idx] = str(ego_sim_agent_id)
  control_ids = [int(agent) for agent in control_ids]

  # # use the z from the first frame of the waymo scene
  scene_track_ids = [track.id for track in waymo_scene.tracks]
  track_indices = [scene_track_ids.index(int(agent)) for agent in control_ids]
  z_start = np.array([waymo_scene.tracks[idx].states[submission_specs.CURRENT_TIME_INDEX].center_z for idx in track_indices])
  z_trajs = np.ones_like(joint_trajs[..., :1]) * z_start[:, None, None]

  joint_trajs = np.concatenate([joint_trajs[..., :2], z_trajs, joint_trajs[..., 2:]], axis=-1)

  joint_scene = joint_scene_from_states(joint_trajs, control_ids)

  return joint_scene

def obtain_waymo_scenario_rollouts(waymo_scene, rollout_trajs_in_world_M, object_ids_M, ego_sim_agent_id):
  """Build Waymo scenario rollouts from M rollout results.

  Aligned with reference: rollout_trajs_in_world_M is (M, A, T, 3) numpy array,
  object_ids_M is a single list of agent id strings.
  Handles remainder when M doesn't evenly divide 32.
  """
  M = len(rollout_trajs_in_world_M)

  num_complete_sets = 32 // M
  remainder = 32 % M

  joint_scenes = []
  for j in range(num_complete_sets):
    for i in range(M):
      joint_scene = joint_scene_from_rollout(waymo_scene, rollout_trajs_in_world_M[i], object_ids_M, ego_sim_agent_id)
      joint_scenes.append(joint_scene)

  for i in range(remainder):
    idx = i % M
    joint_scene = joint_scene_from_rollout(waymo_scene, rollout_trajs_in_world_M[idx], object_ids_M, ego_sim_agent_id)
    joint_scenes.append(joint_scene)

  batch_rollouts = sim_agents_submission_pb2.ScenarioRollouts(
    joint_scenes=joint_scenes, scenario_id=waymo_scene.scenario_id)

  submission_specs.validate_scenario_rollouts(batch_rollouts, waymo_scene)

  return batch_rollouts

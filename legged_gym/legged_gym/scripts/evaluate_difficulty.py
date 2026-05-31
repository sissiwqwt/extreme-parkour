# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import code
import argparse
import csv
import json
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
import cv2
from collections import deque
import statistics
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time, sleep
from legged_gym.utils import webviewer
from tqdm import tqdm

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
                    
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
        # code.interact(local=locals())
    else:
        model = "model_{}.pt".format(checkpoint)

    # load_path = root + model
    return model, checkpoint

def resolve_checkpoint_path(log_root, checkpoint):
    if not os.path.isdir(log_root):
        model_name_cand = os.path.basename(log_root)
        model_parent = os.path.dirname(log_root)
        model_names = os.listdir(model_parent)
        model_names = [name for name in model_names if os.path.isdir(os.path.join(model_parent, name))]
        for name in model_names:
            if len(name) >= 6 and name[:6] == model_name_cand[:6]:
                log_root = os.path.join(model_parent, name)
                break
    model, _ = get_load_path(log_root, checkpoint=checkpoint)
    return os.path.abspath(os.path.join(log_root, model))

def safe_filename_part(value):
    return str(value).replace(os.sep, "_").replace(" ", "_")

def checkpoint_id_from_path(checkpoint_path):
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    if stem.startswith("model_"):
        return stem.split("model_", 1)[1]
    return stem

def parse_evaluation_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--terrain_difficulty",
        "--terrain-difficulty",
        choices=["1-5", "3-5"],
        default="1-5",
        help="Terrain difficulty levels used for evaluation.",
    )
    parser.add_argument(
        "--policy_mode",
        "--policy-mode",
        choices=["distill", "base"],
        default="distill",
        help="Policy architecture to evaluate.",
    )
    parser.add_argument(
        "--num_steps",
        "--num-steps",
        type=int,
        default=1500,
        help="Number of rollout steps used for evaluation.",
    )
    eval_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return eval_args

def active_terrain_names(terrain_dict):
    return [name for name, proportion in terrain_dict.items() if proportion > 0]

def equal_active_terrain_dict(terrain_dict):
    active_names = set(active_terrain_names(terrain_dict))
    return {
        terrain_name: 1.0 if terrain_name in active_names else 0.0
        for terrain_name in terrain_dict.keys()
    }

def difficulty_levels(difficulty_arg):
    if difficulty_arg == "3-5":
        return [2, 3, 4]
    return [0, 1, 2, 3, 4]

def evenly_repeated_indices(num_items, num_slots):
    base = num_items // num_slots
    remainder = num_items % num_slots
    indices = []
    for slot in range(num_slots):
        count = base + (1 if slot < remainder else 0)
        indices.extend([slot] * count)
    return indices

def apply_evaluation_terrain_assignment(env, terrain_names, selected_levels):
    terrain_types = evenly_repeated_indices(env.num_envs, len(terrain_names))
    terrain_levels = []
    for terrain_index in range(len(terrain_names)):
        env_count = terrain_types.count(terrain_index)
        repeated_levels = evenly_repeated_indices(env_count, len(selected_levels))
        terrain_levels.extend([selected_levels[level_index] for level_index in repeated_levels])

    env.terrain_types = torch.tensor(terrain_types, dtype=torch.long, device=env.device)
    env.terrain_levels = torch.tensor(terrain_levels, dtype=torch.long, device=env.device)
    env.max_terrain_level = env.cfg.terrain.num_rows
    env.env_origins[:] = env.terrain_origins[env.terrain_levels, env.terrain_types]
    env.env_class[:] = env.terrain_class[env.terrain_levels, env.terrain_types]
    temp = env.terrain_goals[env.terrain_levels, env.terrain_types]
    last_col = temp[:, -1].unsqueeze(1)
    env.env_goals[:] = torch.cat(
        (temp, last_col.repeat(1, env.cfg.env.num_future_goal_obs, 1)),
        dim=1,
    )[:]
    env.cur_goals = env._gather_cur_goals()
    env.next_goals = env._gather_cur_goals(future=1)

    return {
        env_id: {
            "terrain": terrain_names[terrain_types[env_id]],
            "terrain_col": int(terrain_types[env_id]),
            "difficulty_level": int(terrain_levels[env_id] + 1),
        }
        for env_id in range(env.num_envs)
    }

def summarize_values(values):
    if not values:
        return {"count": 0, "mean": None, "std": None}
    if len(values) == 1:
        return {"count": 1, "mean": float(values[0]), "std": 0.0}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }

def build_env_summaries(records_by_env, env_metadata, edge_violation_sums, edge_violation_counts, num_waypoints_normalizer):
    env_summaries = []
    for env_id, metadata in env_metadata.items():
        records = records_by_env[env_id]
        rewards = [record["reward"] for record in records]
        lengths = [record["episode_length"] for record in records]
        waypoints = [record["num_waypoints"] for record in records]
        normalized_waypoints = [waypoint / num_waypoints_normalizer for waypoint in waypoints]
        falls = [record["fell"] for record in records]
        time_outs = [record["time_out"] for record in records]
        edge_count = edge_violation_counts[env_id]
        edge_mean = edge_violation_sums[env_id] / edge_count if edge_count > 0 else 0.0
        env_summaries.append(
            {
                "env_id": env_id,
                "terrain": metadata["terrain"],
                "terrain_col": metadata["terrain_col"],
                "difficulty_level": metadata["difficulty_level"],
                "done": len(records) > 0,
                "num_episodes": len(records),
                "num_falls": int(sum(falls)),
                "num_time_outs": int(sum(time_outs)),
                "reward_mean": summarize_values(rewards)["mean"],
                "reward_std": summarize_values(rewards)["std"],
                "episode_length_mean": summarize_values(lengths)["mean"],
                "episode_length_std": summarize_values(lengths)["std"],
                "waypoint_mean": summarize_values(waypoints)["mean"],
                "waypoint_std": summarize_values(waypoints)["std"],
                "normalized_waypoint_mean": summarize_values(normalized_waypoints)["mean"],
                "normalized_waypoint_std": summarize_values(normalized_waypoints)["std"],
                "edge_violation_mean": float(edge_mean),
                "edge_violation_count": int(edge_count),
            }
        )
    return env_summaries

def build_summary(env_summaries):
    terrain_records = {}
    overall = {
        "rewards": [],
        "episode_lengths": [],
        "normalized_waypoints": [],
        "waypoints": [],
        "edge_violations": [],
        "done": [],
        "num_episodes": [],
    }

    for env_summary in env_summaries:
        terrain_name = env_summary["terrain"]
        if terrain_name not in terrain_records:
            terrain_records[terrain_name] = {
                "rewards": [],
                "episode_lengths": [],
                "waypoints": [],
                "normalized_waypoints": [],
                "edge_violations": [],
                "done": [],
                "num_episodes": [],
            }
        terrain_records[terrain_name]["done"].append(env_summary["done"])
        terrain_records[terrain_name]["num_episodes"].append(env_summary["num_episodes"])
        terrain_records[terrain_name]["edge_violations"].append(env_summary["edge_violation_mean"])
        overall["done"].append(env_summary["done"])
        overall["num_episodes"].append(env_summary["num_episodes"])
        overall["edge_violations"].append(env_summary["edge_violation_mean"])

        if env_summary["reward_mean"] is not None:
            terrain_records[terrain_name]["rewards"].append(env_summary["reward_mean"])
            terrain_records[terrain_name]["episode_lengths"].append(env_summary["episode_length_mean"])
            terrain_records[terrain_name]["waypoints"].append(env_summary["waypoint_mean"])
            terrain_records[terrain_name]["normalized_waypoints"].append(env_summary["normalized_waypoint_mean"])
            overall["rewards"].append(env_summary["reward_mean"])
            overall["episode_lengths"].append(env_summary["episode_length_mean"])
            overall["waypoints"].append(env_summary["waypoint_mean"])
            overall["normalized_waypoints"].append(env_summary["normalized_waypoint_mean"])

    terrains = {}
    for terrain_name, terrain_data in terrain_records.items():
        terrain_num_episodes = int(sum(terrain_data["num_episodes"]))
        terrains[terrain_name] = {
            "num_envs": len(terrain_data["done"]),
            "done_envs": int(sum(terrain_data["done"])),
            "done_rate": float(np.mean(terrain_data["done"])) if terrain_data["done"] else 0.0,
            "episodes_per_env": summarize_values(terrain_data["num_episodes"]),
            "reward": summarize_values(terrain_data["rewards"]),
            "episode_length": summarize_values(terrain_data["episode_lengths"]),
            "waypoint": summarize_values(terrain_data["waypoints"]),
            "normalized_waypoints": summarize_values(terrain_data["normalized_waypoints"]),
            "edge_violation": summarize_values(terrain_data["edge_violations"]),
        }

    return {
        "overall": {
            "num_envs": len(env_summaries),
            "done_envs": int(sum(overall["done"])),
            "done_rate": float(np.mean(overall["done"])) if overall["done"] else 0.0,
            "episodes_per_env": summarize_values(overall["num_episodes"]),
            "reward": summarize_values(overall["rewards"]),
            "episode_length": summarize_values(overall["episode_lengths"]),
            "waypoint": summarize_values(overall["waypoints"]),
            "normalized_waypoints": summarize_values(overall["normalized_waypoints"]),
            "edge_violation": summarize_values(overall["edge_violations"]),
        },
        "terrains": terrains,
    }

def write_evaluation_outputs(output_dir, output_stem, env_summaries, summary):
    os.makedirs(output_dir, exist_ok=True)
    fieldnames = [
        "env_id",
        "terrain",
        "terrain_col",
        "difficulty_level",
        "done",
        "num_episodes",
        "num_falls",
        "num_time_outs",
        "reward_mean",
        "reward_std",
        "episode_length_mean",
        "episode_length_std",
        "waypoint_mean",
        "waypoint_std",
        "normalized_waypoint_mean",
        "normalized_waypoint_std",
        "edge_violation_mean",
        "edge_violation_count",
    ]
    csv_path = os.path.join(output_dir, f"{output_stem}.csv")
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(env_summaries)

    json_path = os.path.join(output_dir, f"{output_stem}.json")
    with open(json_path, "w") as summary_file:
        json.dump(summary, summary_file, indent=2)
    return csv_path, json_path

def infer_actions(policy_mode, ppo_runner, policy, obs, depth_encoder, infos, env):
    if policy_mode == "base":
        return policy(
            obs.detach(),
            hist_encoding=False,
            scandots_latent=None,
        )

    if env.cfg.depth.use_camera:
        depth_latent = None
        if infos["depth"] is not None:
            obs_student = obs[:, :env.cfg.env.n_proprio]
            obs_student[:, 6:8] = 0
            with torch.no_grad():
                depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)
            depth_latent = depth_latent_and_yaw[:, :-2]
            yaw = depth_latent_and_yaw[:, -2:]
            obs[:, 6:8] = 1.5 * yaw
    else:
        depth_latent = None

    if hasattr(ppo_runner.alg, "depth_actor"):
        with torch.no_grad():
            return ppo_runner.alg.depth_actor(
                obs.detach(),
                hist_encoding=True,
                scandots_latent=depth_latent,
            )

    return policy(
        obs.detach(),
        hist_encoding=True,
        scandots_latent=depth_latent,
    )

def play(args, eval_args):
    if args.web:
        web_viewer = webviewer.WebViewer()
    faulthandler.enable()
    exptid = args.exptid
    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = 16
    env_cfg.env.episode_length_s = 20
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {"smooth slope": 0., 
                                    "rough slope up": 0.0,
                                    "rough slope down": 0.0,
                                    "rough stairs up": 0., 
                                    "rough stairs down": 0., 
                                    "discrete": 0., 
                                    "stepping stones": 0.0,
                                    "gaps": 0., 
                                    "smooth flat": 0,
                                    "pit": 0.0,
                                    "wall": 0.0,
                                    "platform": 0.,
                                    "large stairs up": 0.,
                                    "large stairs down": 0.,
                                    "parkour": 0.25,
                                    "parkour_hurdle": 0.25,
                                    "parkour_flat": 0.,
                                    "parkour_step": 0.25,
                                    "parkour_gap": 0.25, 
                                    "demo": 0}
    
    terrain_names = active_terrain_names(env_cfg.terrain.terrain_dict)
    env_cfg.terrain.num_cols = len(terrain_names)
    env_cfg.terrain.terrain_dict = equal_active_terrain_dict(env_cfg.terrain.terrain_dict)
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = env_cfg.terrain.num_rows - 1
    env_cfg.terrain.max_difficulty = False
    
    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False

    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.cfg.terrain.curriculum = False
    selected_levels = difficulty_levels(eval_args.terrain_difficulty)
    env_metadata = apply_evaluation_terrain_assignment(env, terrain_names, selected_levels)
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    obs = env.get_observations()

    if args.web:
        web_viewer.setup(env)

    # load policy
    train_cfg.runner.resume = True
    if eval_args.policy_mode == "base" and hasattr(train_cfg, "depth_encoder"):
        train_cfg.depth_encoder.if_depth = False
    checkpoint_root = os.path.abspath(log_pth)
    if args.resumeid:
        checkpoint_root = os.path.abspath(os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.proj_name, args.resumeid))
    checkpoint_path = resolve_checkpoint_path(checkpoint_root, args.checkpoint)
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)
    
    policy = ppo_runner.get_inference_policy(device=env.device)
    print(f"Evaluating {eval_args.policy_mode} policy mode.")
    depth_encoder = None
    if eval_args.policy_mode == "distill" and env.cfg.depth.use_camera:
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)
    
    total_steps = 1000
    rewbuffer = deque(maxlen=total_steps)
    lenbuffer = deque(maxlen=total_steps)
    num_waypoints_buffer = deque(maxlen=total_steps)
    time_to_fall_buffer = deque(maxlen=total_steps)
    edge_violation_buffer = deque(maxlen=total_steps)

    cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    cur_time_from_start = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    episode_records = {env_id: [] for env_id in range(env.num_envs)}
    edge_violation_sums = np.zeros(env.num_envs, dtype=np.float64)
    edge_violation_counts = np.zeros(env.num_envs, dtype=np.int64)

    num_actions = env.num_actions if hasattr(env, "num_actions") else 12
    actions = torch.zeros(env.num_envs, num_actions, device=env.device, requires_grad=False)
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if eval_args.policy_mode == "distill" and ppo_runner.if_depth else None

    for i in tqdm(range(eval_args.num_steps)):

        actions = infer_actions(
            eval_args.policy_mode,
            ppo_runner,
            policy,
            obs,
            depth_encoder,
            infos,
            env,
        )
            
        cur_goal_idx = env.cur_goal_idx.clone()
        obs, _, rews, dones, infos = env.step(actions.detach())
        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)
        
        id = env.lookat_id
        # Log stuff
        edge_violations = env.feet_at_edge.sum(dim=1).float()
        edge_violations_cpu = edge_violations.detach().cpu().numpy()
        edge_violation_buffer.extend(edge_violations_cpu.tolist())
        edge_violation_sums += edge_violations_cpu
        edge_violation_counts += 1
        cur_reward_sum += rews
        cur_episode_length += 1
        cur_time_from_start += 1

        dones_cpu = dones.detach().cpu().numpy().astype(bool).tolist()
        time_outs_cpu = infos["time_outs"].detach().cpu().numpy().astype(bool).tolist()
        episode_reward_cpu = cur_reward_sum.detach().cpu().numpy().tolist()
        episode_length_cpu = cur_episode_length.detach().cpu().numpy().tolist()
        time_from_start_cpu = cur_time_from_start.detach().cpu().numpy().tolist()
        cur_goal_idx_cpu = cur_goal_idx.detach().cpu().numpy().tolist()

        new_ids = (dones > 0).nonzero(as_tuple=False)
        killed_ids = ((dones > 0) & (~infos["time_outs"])).nonzero(as_tuple=False)
        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
        num_waypoints_buffer.extend(cur_goal_idx[new_ids][:, 0].cpu().numpy().tolist())
        time_to_fall_buffer.extend(cur_time_from_start[killed_ids][:, 0].cpu().numpy().tolist())

        for env_id_tensor in new_ids[:, 0]:
            env_id = int(env_id_tensor.item())
            fell = bool(dones_cpu[env_id] and not time_outs_cpu[env_id])
            episode_records[env_id].append(
                {
                    "env_id": env_id,
                    "terrain": env_metadata[env_id]["terrain"],
                    "terrain_col": env_metadata[env_id]["terrain_col"],
                    "difficulty_level": env_metadata[env_id]["difficulty_level"],
                    "reward": float(episode_reward_cpu[env_id]),
                    "episode_length": float(episode_length_cpu[env_id]),
                    "num_waypoints": int(cur_goal_idx_cpu[env_id]),
                    "time_to_fall": float(time_from_start_cpu[env_id]),
                    "fell": fell,
                    "time_out": bool(time_outs_cpu[env_id]),
                    "finished_at_step": i,
                }
            )

        cur_reward_sum[new_ids] = 0
        cur_episode_length[new_ids] = 0
        cur_time_from_start[killed_ids] = 0
    
    #compute buffer mean and std
    rew_mean = statistics.mean(rewbuffer)
    rew_std = statistics.stdev(rewbuffer)

    len_mean = statistics.mean(lenbuffer)
    len_std = statistics.stdev(lenbuffer)

    num_waypoints_mean = np.mean(np.array(num_waypoints_buffer).astype(float)/7.0)
    num_waypoints_std = np.std(np.array(num_waypoints_buffer).astype(float)/7.0)

    # time_to_fall_mean = statistics.mean(time_to_fall_buffer)
    # time_to_fall_std = statistics.stdev(time_to_fall_buffer)

    edge_violation_mean = np.mean(edge_violation_buffer)
    edge_violation_std = np.std(edge_violation_buffer)

    print("Mean reward: {:.2f}$\\pm${:.2f}".format(rew_mean, rew_std))
    print("Mean episode length: {:.2f}$\\pm${:.2f}".format(len_mean, len_std))
    print("Mean number of waypoints: {:.2f}$\\pm${:.2f}".format(num_waypoints_mean, num_waypoints_std))
    # print("Mean time to fall: {:.2f}$\\pm${:.2f}".format(time_to_fall_mean, time_to_fall_std))
    print("Mean edge violation: {:.2f}$\\pm${:.2f}".format(edge_violation_mean, edge_violation_std))

    env_summaries = build_env_summaries(
        episode_records,
        env_metadata,
        edge_violation_sums,
        edge_violation_counts,
        num_waypoints_normalizer=7.0,
    )
    aggregate_summary = build_summary(env_summaries)
    summary = {
        "checkpoint_path": checkpoint_path,
        "final_result": aggregate_summary["overall"],
        "terrains": aggregate_summary["terrains"],
        "config": {
            "task": args.task,
            "exptid": args.exptid,
            "proj_name": args.proj_name,
            "num_envs": env.num_envs,
            "num_steps": eval_args.num_steps,
            "policy_mode": eval_args.policy_mode,
            "hist_encoding": eval_args.policy_mode == "distill",
            "terrain_difficulty": eval_args.terrain_difficulty,
            "terrain_names": terrain_names,
            "difficulty_levels": [level + 1 for level in selected_levels],
        },
    }
    output_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "evaluations")
    ckpt_name = checkpoint_id_from_path(checkpoint_path)
    output_stem = "{}_{}_{}".format(
        safe_filename_part(eval_args.policy_mode),
        safe_filename_part(ckpt_name),
        safe_filename_part(eval_args.terrain_difficulty),
    )
    csv_path, json_path = write_evaluation_outputs(output_dir, output_stem, env_summaries, summary)
    print(f"Evaluation CSV saved to: {csv_path}")
    print(f"Evaluation JSON saved to: {json_path}")


if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    eval_args = parse_evaluation_args()
    args = get_args()
    play(args, eval_args)


# 038-10 no feet edge
# 038-91 ours
# 043-21 non-inner

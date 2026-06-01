from __future__ import annotations

import numpy as np


def sample_terrain_names(num_envs: int, terrain_weights: dict[str, float], rng: np.random.Generator) -> list[str]:
    names = list(terrain_weights.keys())
    weights = np.asarray([float(terrain_weights[name]) for name in names], dtype=np.float64)
    weights = weights / weights.sum()
    indices = rng.choice(len(names), size=num_envs, p=weights)
    return [names[idx] for idx in indices]


def curriculum_terrain_names(num_cols: int, terrain_weights: dict[str, float]) -> list[str]:
    names = list(terrain_weights.keys())
    weights = np.asarray([float(terrain_weights[name]) for name in names], dtype=np.float64)
    weights = weights / weights.sum()
    cumulative = np.cumsum(weights)

    terrain_names = []
    for col in range(num_cols):
        choice = col / num_cols + 0.001
        terrain_idx = int(np.searchsorted(cumulative, choice, side="right"))
        terrain_idx = min(terrain_idx, len(names) - 1)
        terrain_names.append(names[terrain_idx])
    return terrain_names


def generate_goal_track(
    terrain_name: str,
    num_goals: int,
    env_length: float,
    env_width: float,
    difficulty: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate per-env 2D waypoint tracks mirroring old parkour terrain patterns.

    This ports the waypoint layout logic first, without yet porting the full
    heightfield/trimesh construction from the Isaac Gym terrain generator.
    """
    mid_y = env_width * 0.5
    goals = np.zeros((num_goals, 3), dtype=np.float32)
    goals[0, 0] = 2.5
    goals[0, 1] = mid_y

    if terrain_name == "parkour":
        x_min, x_max = 1.0, 1.1 + 0.3 * difficulty
        y_min, y_max = 0.2, 0.3 + 0.1 * difficulty
        x = goals[0, 0]
        left_right = int(rng.integers(0, 2))
        for i in range(1, num_goals - 1):
            x += rng.uniform(x_min, x_max)
            y = mid_y + (1.0 if left_right else -1.0) * rng.uniform(y_min, y_max)
            goals[i, 0] = x
            goals[i, 1] = np.clip(y, 0.4, env_width - 0.4)
            left_right = 1 - left_right
        goals[-1, 0] = min(x + 2.0 * rng.uniform(x_min, x_max), env_length - 0.5)
        goals[-1, 1] = mid_y
    elif terrain_name in {"parkour_hurdle", "parkour_flat", "parkour_step", "parkour_gap", "alternating_step"}:
        lateral = 0.4 if terrain_name != "alternating_step" else 0.2
        x = goals[0, 0]
        for i in range(1, num_goals - 1):
            if terrain_name == "parkour_step":
                dx = rng.uniform(0.4 + difficulty * 0.1, 1.5 + difficulty * 0.2)
            elif terrain_name == "parkour_gap":
                dx = rng.uniform(0.8, 1.5)
            elif terrain_name == "alternating_step":
                dx = rng.uniform(0.35, 0.8)
            else:
                dx = rng.uniform(1.2, 2.2)
            x += dx
            y = mid_y + rng.uniform(-lateral, lateral)
            goals[i, 0] = x
            goals[i, 1] = np.clip(y, 0.4, env_width - 0.4)
        goals[-1, 0] = min(x + rng.uniform(1.2, 2.0), env_length - 0.5)
        goals[-1, 1] = mid_y
    elif terrain_name in {"beam_gap", "asymmetric_gap", "narrow_gap", "climbing_wall"}:
        x = goals[0, 0]
        for i in range(1, num_goals - 1):
            if terrain_name == "beam_gap":
                dx = rng.uniform(1.0, 1.8)
                y = mid_y + rng.uniform(-0.08, 0.08)
            elif terrain_name == "asymmetric_gap":
                dx = rng.uniform(1.0, 1.6)
                offset = 0.25 + 0.45 * difficulty
                y = mid_y + (1.0 if i % 2 == 0 else -1.0) * offset
            elif terrain_name == "narrow_gap":
                dx = rng.uniform(0.8, 1.4)
                y = mid_y + (1.0 if i % 2 == 0 else -1.0) * 0.75
            else:
                dx = rng.uniform(0.8, 1.4)
                y = mid_y + rng.uniform(-0.4, 0.4)
            x += dx
            goals[i, 0] = x
            goals[i, 1] = np.clip(y, 0.3, env_width - 0.3)
        goals[-1, 0] = min(x + rng.uniform(1.0, 1.8), env_length - 0.5)
        goals[-1, 1] = mid_y
    elif terrain_name == "parkour_v2":
        x = goals[0, 0]
        cursor = 1
        while cursor < num_goals - 1 and x < env_length - 2.0:
            seg_type = rng.choice(
                ["slanted_hurdle", "alternating_step", "beam_gap", "biased_gap", "narrow_gap"]
            )
            seg_len = rng.uniform(1.5, 2.8)
            x += seg_len
            if seg_type == "beam_gap":
                y = mid_y + rng.uniform(-0.08, 0.08)
            elif seg_type == "biased_gap":
                y = mid_y + (1.0 if cursor % 2 == 0 else -1.0) * (0.25 + 0.45 * difficulty)
            elif seg_type == "narrow_gap":
                y = mid_y + (1.0 if cursor % 2 == 0 else -1.0) * 0.75
            else:
                y = mid_y + rng.uniform(-0.35, 0.35)
            goals[cursor, 0] = min(x, env_length - 1.0)
            goals[cursor, 1] = np.clip(y, 0.3, env_width - 0.3)
            cursor += 1
        if cursor < num_goals:
            goals[cursor:, 0] = min(x + 1.0, env_length - 0.5)
            goals[cursor:, 1] = mid_y
        else:
            goals[-1, 0] = min(goals[-2, 0] + 1.0, env_length - 0.5)
            goals[-1, 1] = mid_y
    else:
        forward_steps = np.linspace(2.5, env_length - 0.5, num_goals, dtype=np.float32)
        goals[:, 0] = forward_steps
        goals[:, 1] = mid_y

    goals[0, 2] = 0.0
    return goals

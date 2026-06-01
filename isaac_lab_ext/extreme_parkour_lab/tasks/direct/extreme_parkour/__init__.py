"""Gymnasium registration for the Extreme Parkour Isaac Lab task."""

from __future__ import annotations

import gymnasium as gym

from .extreme_parkour_env import ExtremeParkourEnv, ExtremeParkourEnvCfg


gym.register(
    id="Isaac-Extreme-Parkour-A1-Direct-v0",
    entry_point="extreme_parkour_lab.tasks.direct.extreme_parkour:ExtremeParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ExtremeParkourEnvCfg,
        "rsl_rl_cfg_entry_point": "extreme_parkour_lab.tasks.direct.extreme_parkour.agents.rsl_rl_ppo_cfg:ExtremeParkourPPORunnerCfg",
    },
)


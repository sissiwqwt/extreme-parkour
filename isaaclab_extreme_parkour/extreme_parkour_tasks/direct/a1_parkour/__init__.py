"""A1 parkour task registration for Isaac Lab."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-A1-Parkour-Direct-v0",
    entry_point=f"{__name__}.a1_parkour_env:A1ParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.a1_parkour_env_cfg:A1ParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A1ParkourPPORunnerCfg",
    },
)

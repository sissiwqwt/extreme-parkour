from __future__ import annotations

import isaaclab.terrains as terrain_gen

from isaaclab.terrains import TerrainGeneratorCfg

from .custom_terrains import MeshAsymmetricGapTerrainCfg, MeshBeamGapTerrainCfg, MeshParkourGapTerrainCfg


# This is an intermediate terrain configuration for the parkour port.
# It keeps the terrain ownership local to the task package so the next step can
# replace these stock sub-terrains with custom parkour mesh generators without
# changing the environment wiring again.
PARKOUR_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(18.0, 4.0),
    border_width=5.0,
    num_rows=5,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "parkour_gap": MeshParkourGapTerrainCfg(
            proportion=0.35,
            border_width=0.1,
            border_height=1.0,
            platform_height=0.0,
            spawn_platform_length=2.5,
            final_platform_length=2.0,
            num_segments=6,
            gap_width_range=(0.25, 0.8),
            lane_width_range=(0.6, 1.2),
            step_spacing_range=(0.8, 1.5),
            y_offset_range=(-0.4, 0.4),
        ),
        "beam_gap": MeshBeamGapTerrainCfg(
            proportion=0.20,
            border_width=0.1,
            border_height=1.0,
            platform_height=0.0,
            spawn_platform_length=2.5,
            final_platform_length=2.0,
            num_segments=6,
            gap_width_range=(0.25, 0.8),
            beam_width_range=(0.18, 0.24),
            segment_spacing_range=(1.0, 1.8),
            y_offset_range=(-0.08, 0.08),
        ),
        "asymmetric_gap": MeshAsymmetricGapTerrainCfg(
            proportion=0.15,
            border_width=0.1,
            border_height=1.0,
            platform_height=0.0,
            spawn_platform_length=2.5,
            final_platform_length=2.0,
            num_segments=6,
            gap_width_range=(0.2, 0.35),
            corridor_half_width_range=(0.5, 0.6),
            lateral_offset_range=(0.25, 0.7),
            segment_spacing_range=(1.0, 1.6),
        ),
        "boxes": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.075,
            obstacle_width_range=(0.35, 0.75),
            obstacle_height_range=(0.05, 0.20),
            num_obstacles=30,
            platform_width=2.5,
            border_width=0.25,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.075,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.075,
            slope_range=(0.0, 0.25),
            platform_width=2.5,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.0, 0.25),
            platform_width=2.5,
            border_width=0.25,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.20),
            step_width=0.4,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
    },
)

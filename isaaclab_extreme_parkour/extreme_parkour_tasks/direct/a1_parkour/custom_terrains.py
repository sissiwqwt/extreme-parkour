from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.trimesh.utils import make_border
from isaaclab.utils import configclass


def parkour_gap_terrain(
    difficulty: float,
    cfg: "MeshParkourGapTerrainCfg",
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Trimesh port of a gap-style parkour lane.

    This is the first end-to-end custom terrain family for the Isaac Lab port.
    It creates a sequence of narrow safe platforms separated by full-width gaps,
    with a laterally shifted corridor center similar to the old Isaac Gym
    `parkour_gap_terrain` routine.
    """

    meshes: list[trimesh.Trimesh] = []
    terrain_size = cfg.size
    terrain_center = np.array([terrain_size[0] * 0.5, terrain_size[1] * 0.5, 0.0], dtype=np.float32)

    platform_height = cfg.platform_height
    lane_half_width = cfg.lane_width_range[0] + difficulty * (cfg.lane_width_range[1] - cfg.lane_width_range[0])
    gap_width = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])
    step_spacing_min = cfg.step_spacing_range[0]
    step_spacing_max = cfg.step_spacing_range[1]

    border_height = max(cfg.border_height, 0.1)
    inner_size = (
        max(terrain_size[0] - 2.0 * cfg.border_width, 0.1),
        max(terrain_size[1] - 2.0 * cfg.border_width, 0.1),
    )
    border_center = [terrain_center[0], terrain_center[1], -border_height * 0.5]
    if cfg.border_width > 0.0:
        meshes += make_border(terrain_size, inner_size, border_height, border_center)

    # Spawn platform.
    spawn_len = cfg.spawn_platform_length
    spawn_dim = (spawn_len, terrain_size[1], border_height)
    spawn_pos = (spawn_len * 0.5, terrain_center[1], -border_height * 0.5)
    meshes.append(trimesh.creation.box(spawn_dim, trimesh.transformations.translation_matrix(spawn_pos)))

    corridor_x = spawn_len
    last_x = corridor_x
    num_segments = cfg.num_segments

    rng = np.random.default_rng(cfg.seed if cfg.seed is not None else None)
    for _ in range(num_segments):
        segment_len = float(rng.uniform(step_spacing_min, step_spacing_max))
        corridor_x = min(corridor_x + segment_len, terrain_size[0] - cfg.final_platform_length - gap_width - 0.2)

        lateral_offset = float(rng.uniform(cfg.y_offset_range[0], cfg.y_offset_range[1]))
        center_y = float(np.clip(terrain_center[1] + lateral_offset, lane_half_width, terrain_size[1] - lane_half_width))

        safe_len = max(corridor_x - last_x - gap_width, 0.1)
        if safe_len > 0.0:
            safe_center_x = last_x + safe_len * 0.5
            safe_dim = (safe_len, lane_half_width * 2.0, border_height)
            safe_pos = (safe_center_x, center_y, -border_height * 0.5)
            meshes.append(trimesh.creation.box(safe_dim, trimesh.transformations.translation_matrix(safe_pos)))

        last_x = corridor_x

    final_len = cfg.final_platform_length
    final_center_x = min(last_x + gap_width + final_len * 0.5, terrain_size[0] - final_len * 0.5)
    final_dim = (final_len, terrain_size[1], border_height)
    final_pos = (final_center_x, terrain_center[1], -border_height * 0.5)
    meshes.append(trimesh.creation.box(final_dim, trimesh.transformations.translation_matrix(final_pos)))

    origin = np.array([1.0, terrain_size[1] * 0.5, platform_height], dtype=np.float32)
    return meshes, origin


def beam_gap_terrain(
    difficulty: float,
    cfg: "MeshBeamGapTerrainCfg",
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Trimesh port of the narrow-beam-with-gaps terrain family."""

    meshes: list[trimesh.Trimesh] = []
    terrain_size = cfg.size
    terrain_center = np.array([terrain_size[0] * 0.5, terrain_size[1] * 0.5, 0.0], dtype=np.float32)

    platform_height = cfg.platform_height
    border_height = max(cfg.border_height, 0.1)
    beam_width = cfg.beam_width_range[0] + difficulty * (cfg.beam_width_range[1] - cfg.beam_width_range[0])
    gap_width = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])

    inner_size = (
        max(terrain_size[0] - 2.0 * cfg.border_width, 0.1),
        max(terrain_size[1] - 2.0 * cfg.border_width, 0.1),
    )
    border_center = [terrain_center[0], terrain_center[1], -border_height * 0.5]
    if cfg.border_width > 0.0:
        meshes += make_border(terrain_size, inner_size, border_height, border_center)

    spawn_len = cfg.spawn_platform_length
    spawn_dim = (spawn_len, terrain_size[1], border_height)
    spawn_pos = (spawn_len * 0.5, terrain_center[1], -border_height * 0.5)
    meshes.append(trimesh.creation.box(spawn_dim, trimesh.transformations.translation_matrix(spawn_pos)))

    rng = np.random.default_rng(cfg.seed if cfg.seed is not None else None)
    beam_x = spawn_len
    last_beam_center_y = terrain_center[1]

    for _ in range(cfg.num_segments):
        seg_len = float(rng.uniform(cfg.segment_spacing_range[0], cfg.segment_spacing_range[1]))
        gap_center_x = min(beam_x + seg_len, terrain_size[0] - cfg.final_platform_length - gap_width - 0.2)

        lateral_offset = float(rng.uniform(cfg.y_offset_range[0], cfg.y_offset_range[1]))
        beam_center_y = float(
            np.clip(terrain_center[1] + lateral_offset, beam_width * 0.5 + 0.05, terrain_size[1] - beam_width * 0.5 - 0.05)
        )

        safe_len = max(gap_center_x - beam_x - gap_width * 0.5, 0.1)
        safe_center_x = beam_x + safe_len * 0.5
        safe_dim = (safe_len, beam_width, border_height)
        safe_pos = (safe_center_x, beam_center_y, -border_height * 0.5)
        meshes.append(trimesh.creation.box(safe_dim, trimesh.transformations.translation_matrix(safe_pos)))

        beam_x = gap_center_x + gap_width * 0.5
        last_beam_center_y = beam_center_y

    final_len = cfg.final_platform_length
    final_safe_len = max(terrain_size[0] - beam_x - 0.2, final_len)
    final_center_x = min(beam_x + final_safe_len * 0.5, terrain_size[0] - 0.5 * final_safe_len)
    final_dim = (max(final_safe_len, 0.1), beam_width, border_height)
    final_pos = (final_center_x, last_beam_center_y, -border_height * 0.5)
    meshes.append(trimesh.creation.box(final_dim, trimesh.transformations.translation_matrix(final_pos)))

    landing_center_x = terrain_size[0] - final_len * 0.5
    landing_dim = (final_len, terrain_size[1], border_height)
    landing_pos = (landing_center_x, terrain_center[1], -border_height * 0.5)
    meshes.append(trimesh.creation.box(landing_dim, trimesh.transformations.translation_matrix(landing_pos)))

    origin = np.array([1.0, terrain_size[1] * 0.5, platform_height], dtype=np.float32)
    return meshes, origin


def asymmetric_gap_terrain(
    difficulty: float,
    cfg: "MeshAsymmetricGapTerrainCfg",
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Trimesh port of the alternating laterally-shifted gap platforms."""

    meshes: list[trimesh.Trimesh] = []
    terrain_size = cfg.size
    terrain_center = np.array([terrain_size[0] * 0.5, terrain_size[1] * 0.5, 0.0], dtype=np.float32)

    platform_height = cfg.platform_height
    border_height = max(cfg.border_height, 0.1)
    gap_width = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])
    corridor_half_width = cfg.corridor_half_width_range[0] + difficulty * (
        cfg.corridor_half_width_range[1] - cfg.corridor_half_width_range[0]
    )
    lateral_offset = cfg.lateral_offset_range[0] + difficulty * (
        cfg.lateral_offset_range[1] - cfg.lateral_offset_range[0]
    )

    inner_size = (
        max(terrain_size[0] - 2.0 * cfg.border_width, 0.1),
        max(terrain_size[1] - 2.0 * cfg.border_width, 0.1),
    )
    border_center = [terrain_center[0], terrain_center[1], -border_height * 0.5]
    if cfg.border_width > 0.0:
        meshes += make_border(terrain_size, inner_size, border_height, border_center)

    spawn_len = cfg.spawn_platform_length
    spawn_dim = (spawn_len, terrain_size[1], border_height)
    spawn_pos = (spawn_len * 0.5, terrain_center[1], -border_height * 0.5)
    meshes.append(trimesh.creation.box(spawn_dim, trimesh.transformations.translation_matrix(spawn_pos)))

    rng = np.random.default_rng(cfg.seed if cfg.seed is not None else None)
    seg_start_x = spawn_len

    for segment_idx in range(cfg.num_segments):
        seg_len = float(rng.uniform(cfg.segment_spacing_range[0], cfg.segment_spacing_range[1]))
        seg_end_x = min(seg_start_x + seg_len, terrain_size[0] - cfg.final_platform_length - gap_width - 0.2)

        sign = 1.0 if segment_idx % 2 == 0 else -1.0
        center_y = float(
            np.clip(
                terrain_center[1] + sign * lateral_offset,
                corridor_half_width + 0.05,
                terrain_size[1] - corridor_half_width - 0.05,
            )
        )

        safe_center_x = seg_start_x + max(seg_end_x - seg_start_x, 0.1) * 0.5
        safe_dim = (max(seg_end_x - seg_start_x, 0.1), corridor_half_width * 2.0, border_height)
        safe_pos = (safe_center_x, center_y, -border_height * 0.5)
        meshes.append(trimesh.creation.box(safe_dim, trimesh.transformations.translation_matrix(safe_pos)))

        seg_start_x = seg_end_x + gap_width

    final_len = cfg.final_platform_length
    final_center_x = terrain_size[0] - final_len * 0.5
    final_dim = (final_len, terrain_size[1], border_height)
    final_pos = (final_center_x, terrain_center[1], -border_height * 0.5)
    meshes.append(trimesh.creation.box(final_dim, trimesh.transformations.translation_matrix(final_pos)))

    origin = np.array([1.0, terrain_size[1] * 0.5, platform_height], dtype=np.float32)
    return meshes, origin


@configclass
class MeshParkourGapTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a parkour gap lane built from trimesh boxes."""

    function = parkour_gap_terrain

    border_width: float = 0.0
    border_height: float = 1.0
    platform_height: float = 0.0
    spawn_platform_length: float = 2.5
    final_platform_length: float = 2.0
    num_segments: int = 6
    gap_width_range: tuple[float, float] = (0.25, 0.8)
    lane_width_range: tuple[float, float] = (0.6, 1.2)
    step_spacing_range: tuple[float, float] = (0.8, 1.5)
    y_offset_range: tuple[float, float] = (-0.4, 0.4)
    seed: int | None = None


@configclass
class MeshBeamGapTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a narrow-beam terrain with repeated gaps."""

    function = beam_gap_terrain

    border_width: float = 0.0
    border_height: float = 1.0
    platform_height: float = 0.0
    spawn_platform_length: float = 2.5
    final_platform_length: float = 2.0
    num_segments: int = 6
    gap_width_range: tuple[float, float] = (0.25, 0.8)
    beam_width_range: tuple[float, float] = (0.18, 0.24)
    segment_spacing_range: tuple[float, float] = (1.0, 1.8)
    y_offset_range: tuple[float, float] = (-0.08, 0.08)
    seed: int | None = None


@configclass
class MeshAsymmetricGapTerrainCfg(SubTerrainBaseCfg):
    """Configuration for laterally alternating corridor platforms with gaps."""

    function = asymmetric_gap_terrain

    border_width: float = 0.0
    border_height: float = 1.0
    platform_height: float = 0.0
    spawn_platform_length: float = 2.5
    final_platform_length: float = 2.0
    num_segments: int = 6
    gap_width_range: tuple[float, float] = (0.2, 0.35)
    corridor_half_width_range: tuple[float, float] = (0.5, 0.6)
    lateral_offset_range: tuple[float, float] = (0.25, 0.7)
    segment_spacing_range: tuple[float, float] = (1.0, 1.6)
    seed: int | None = None

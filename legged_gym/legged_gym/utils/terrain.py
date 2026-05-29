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

import numpy as np
from numpy.random import choice
from scipy import interpolate
import random
from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from scipy import ndimage
from pydelatin import Delatin
import pyfqmr
from scipy.ndimage import binary_dilation


A1_TRUNK_WIDTH = 0.194
A1_HIP_Y_OFFSET = 0.047
A1_THIGH_Y_OFFSET = 0.08505
A1_FOOT_COLLISION_RADIUS = 0.02
#A1_TOTAL_WIDTH = 2 * (A1_HIP_Y_OFFSET + A1_THIGH_Y_OFFSET + A1_FOOT_COLLISION_RADIUS)
A1_TOTAL_WIDTH = 0.3041
A1_BEAM_GAP_TARGET_WIDTH = 0.31
A1_NARROW_GAP_TARGET_WIDTH = 0.31


class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width

        cfg.terrain_proportions = np.array(cfg.terrain_proportions) / np.sum(cfg.terrain_proportions)
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]
        self.terrain_names = list(cfg.terrain_dict.keys()) if hasattr(cfg, "terrain_dict") else None
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols))
        # self.env_slope_vec = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.goals = np.zeros((cfg.num_rows, cfg.num_cols, cfg.num_goals, 3))
        self.num_goals = cfg.num_goals

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            if hasattr(cfg, "max_difficulty"):
                self.curiculum(random=True, max_difficulty=cfg.max_difficulty)
            else:
                self.curiculum(random=True)
            # self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            print("Converting heightmap to trimesh...")
            if cfg.hf2mesh_method == "grid":
                self.vertices, self.triangles, self.x_edge_mask = convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                                self.cfg.horizontal_scale,
                                                                                                self.cfg.vertical_scale,
                                                                                                self.cfg.slope_treshold)
                half_edge_width = int(self.cfg.edge_width_thresh / self.cfg.horizontal_scale)
                structure = np.ones((half_edge_width*2+1, 1))
                self.x_edge_mask = binary_dilation(self.x_edge_mask, structure=structure)
                if self.cfg.simplify_grid:
                    mesh_simplifier = pyfqmr.Simplify()
                    mesh_simplifier.setMesh(self.vertices, self.triangles)
                    mesh_simplifier.simplify_mesh(target_count = int(0.05*self.triangles.shape[0]), aggressiveness=7, preserve_border=True, verbose=10)

                    self.vertices, self.triangles, normals = mesh_simplifier.getMesh()
                    self.vertices = self.vertices.astype(np.float32)
                    self.triangles = self.triangles.astype(np.uint32)
            else:
                assert cfg.hf2mesh_method == "fast", "Height field to mesh method must be grid or fast"
                self.vertices, self.triangles = convert_heightfield_to_trimesh_delatin(self.height_field_raw, self.cfg.horizontal_scale, self.cfg.vertical_scale, max_error=cfg.max_error)
            print("Created {} vertices".format(self.vertices.shape[0]))
            print("Created {} triangles".format(self.triangles.shape[0]))

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            # difficulty = np.random.choice([0.5, 0.75, 0.9])
            difficulty = np.random.uniform(-0.2, 1.2)
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self, random=False, max_difficulty=False):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / (self.cfg.num_rows-1)
                choice = j / self.cfg.num_cols + 0.001
                if random:
                    if max_difficulty:
                        terrain = self.make_terrain(choice, np.random.uniform(0.7, 1))
                    else:
                        terrain = self.make_terrain(choice, np.random.uniform(0, 1))
                else:
                    terrain = self.make_terrain(choice, difficulty)

                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.length_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def add_roughness(self, terrain, difficulty=1):
        max_height = (self.cfg.height[1] - self.cfg.height[0]) * difficulty + self.cfg.height[0]
        height = random.uniform(self.cfg.height[0], max_height)
        terrain_utils.random_uniform_terrain(terrain, min_height=-height, max_height=height, step=0.005, downsampled_scale=self.cfg.downsampled_scale)

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.length_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.02 + 0.14 * difficulty
        discrete_obstacles_height = 0.03 + difficulty * 0.15
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        idx = min(int(np.searchsorted(self.proportions, choice, side="right")), len(self.proportions) - 1)
        terrain_name = self.terrain_names[idx] if self.terrain_names is not None else idx
        if terrain_name == "smooth slope":
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            # self.add_roughness(terrain)
        elif terrain_name in ["rough slope up", "rough slope down"]:
            if terrain_name == "rough slope down":
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            self.add_roughness(terrain)
        elif terrain_name in ["rough stairs up", "rough stairs down"]:
            if terrain_name == "rough stairs down":
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
            self.add_roughness(terrain)
        elif terrain_name == "discrete":
            num_rectangles = 20
            rectangle_min_size = 0.5
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
            self.add_roughness(terrain)
        elif terrain_name == "stepping stones":
            stones_size = 1.5 - 1.2*difficulty
            # terrain_utils.stepping_stones_terrain(terrain, stone_size=stones_size, stone_distance=0.1, stone_distance_rand=0, max_height=0.04*difficulty, platform_size=2.)
            half_sloped_terrain(terrain, wall_width=4, start2center=0.5, max_height=0.00)
            stepping_stones_terrain(terrain, stone_size=1.5-0.2*difficulty, stone_distance=0.0+0.4*difficulty, max_height=0.2*difficulty, platform_size=1.2)
            self.add_roughness(terrain)
        elif terrain_name == "gaps":
            # gap_size = random.uniform(self.cfg.gap_size[0], self.cfg.gap_size[1])
            gap_parkour_terrain(terrain, difficulty, platform_size=4)
            self.add_roughness(terrain)
        elif terrain_name == "smooth flat":
            self.add_roughness(terrain)
            # pass
        elif terrain_name == "pit":
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        elif terrain_name == "wall":
            if self.cfg.all_vertical:
                half_slope_difficulty = 1.0
            else:
                difficulty *= 1.3
                if not self.cfg.no_flat:
                    difficulty -= 0.1
                if difficulty > 1:
                    half_slope_difficulty = 1.0
                elif difficulty < 0:
                    self.add_roughness(terrain)
                    terrain.slope_vector = np.array([1, 0., 0]).astype(np.float32)
                    terrain.idx = idx
                    return terrain
                else:
                    half_slope_difficulty = difficulty
            wall_width = 4 - half_slope_difficulty * 4
            # terrain_utils.wall_terrain(terrain, height=1, start2center=0.7)
            # terrain_utils.tanh_terrain(terrain, height=1.0, start2center=0.7)
            if self.cfg.flat_wall:
                half_sloped_terrain(terrain, wall_width=4, start2center=0.5, max_height=0.00)
            else:
                half_sloped_terrain(terrain, wall_width=wall_width, start2center=0.5, max_height=1.5)
            max_height = terrain.height_field_raw.max()
            top_mask = terrain.height_field_raw > max_height - 0.05
            self.add_roughness(terrain, difficulty=1)
            terrain.height_field_raw[top_mask] = max_height
        elif terrain_name == "platform":
            # half platform terrain
            half_platform_terrain(terrain, max_height=0.1 + 0.4 * difficulty )
            self.add_roughness(terrain, difficulty=1)
        elif terrain_name in ["large stairs up", "large stairs down"]:
            height = 0.1 + 0.3 * difficulty
            if terrain_name == "large stairs down":
                height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=1., step_height=height, platform_size=3.)
            self.add_roughness(terrain)
        elif terrain_name == "parkour":
            x_range = [-0.1, 0.1+0.3*difficulty]  # offset to stone_len
            y_range = [0.2, 0.3+0.1*difficulty]
            stone_len = [0.9 - 0.3*difficulty, 1 - 0.2*difficulty]#2 * round((0.6) / 2.0, 1)
            incline_height = 0.25*difficulty
            last_incline_height = incline_height + 0.1 - 0.1*difficulty
            parkour_terrain(terrain,
                            num_stones=self.num_goals - 2,
                            x_range=x_range, 
                            y_range=y_range,
                            incline_height=incline_height,
                            stone_len=stone_len,
                            stone_width=1.0, 
                            last_incline_height=last_incline_height,
                            pad_height=0,
                            pit_depth=[0.2, 1])
            # terrain.height_field_raw[:] = 0
            self.add_roughness(terrain)
        elif terrain_name == "parkour_hurdle":
            parkour_hurdle_terrain(terrain,
                                   num_stones=self.num_goals - 2,
                                   stone_len=0.1+0.3*difficulty,
                                   hurdle_height_range=[0.1+0.1*difficulty, 0.15+0.25*difficulty],
                                   pad_height=0,
                                   x_range=[1.2, 2.2],
                                   y_range=self.cfg.y_range,
                                   half_valid_width=[0.4, 0.8],
                                   )
            # terrain.height_field_raw[:] = 0
            self.add_roughness(terrain)
        elif terrain_name == "parkour_flat":
            parkour_hurdle_terrain(terrain,
                                   num_stones=self.num_goals - 2,
                                   stone_len=0.1+0.3*difficulty,
                                   hurdle_height_range=[0.1+0.1*difficulty, 0.15+0.15*difficulty],
                                   pad_height=0,
                                   y_range=self.cfg.y_range,
                                   half_valid_width=[0.45, 1],
                                   flat=True
                                   )
            self.add_roughness(terrain)
        elif terrain_name == "parkour_step":
            parkour_step_terrain(terrain,
                                   num_stones=self.num_goals - 2,
                                   step_height=0.1 + 0.35*difficulty,
                                   x_range=[0.3,1.5],
                                   y_range=self.cfg.y_range,
                                   half_valid_width=[0.5, 1],
                                   pad_height=0,
                                   )
            self.add_roughness(terrain)
        elif terrain_name == "parkour_gap":
            parkour_gap_terrain(terrain,
                                num_gaps=self.num_goals - 2,
                                gap_size=0.1 + 0.7 * difficulty,
                                gap_depth=[0.2, 1],
                                pad_height=0,
                                x_range=[0.8, 1.5],
                                y_range=self.cfg.y_range,
                                half_valid_width=[0.6, 1.2],
                                # flat=True
                                )
            self.add_roughness(terrain)
        elif terrain_name == "alternating_step":
            alternating_step_terrain(terrain,
                                     platform_len=2.5,
                                     platform_height=0.0,
                                     num_steps=self.num_goals - 2,
                                     x_range=[0.35, 0.8],
                                     y_range=[-0.2, 0.2],
                                     half_valid_width=[0.45, 0.7],
                                     step_height=0.1 + 0.3*difficulty,
                                     pad_width=0.1,
                                     pad_height=0.5)
            self.add_roughness(terrain)
        elif terrain_name in ["bean_gap", "beam_gap"]:
            beam_gap_difficulty = np.clip(difficulty, 0.0, 1.0)
            target_beam_width = max(A1_BEAM_GAP_TARGET_WIDTH, A1_TOTAL_WIDTH)
            #target_gap_size = 0.38 / 2.0
            beam_width_range = [
                0.4 + beam_gap_difficulty * (target_beam_width - 0.4),
                0.8 + beam_gap_difficulty * (target_beam_width - 0.8),
            ]
            beam_gap_terrain(terrain,
                             platform_len=2.5,
                             platform_height=0.0,
                             num_gaps=self.num_goals - 2,
                             gap_size=0.25 + 0.5 * difficulty,
                             x_range=[1.0, 1.8],
                             beam_width_range=beam_width_range,
                             y_offset_range=[-0.04, 0.04],
                             gap_depth=[0.2, 1.0],
                             pad_width=0.1,
                             pad_height=0.5)
            self.add_roughness(terrain)
        elif terrain_name == "asymmetric_gap":
            asymmetric_gap_difficulty = np.clip(difficulty, 0.0, 1.0)
            corridor_half_width = 0.6 - 0.1 * asymmetric_gap_difficulty
            max_edge_distance = 0.38 * 1.5
            target_lateral_offset = corridor_half_width + max_edge_distance / 2.0
            lateral_offset = 0.25 + 0.3*difficulty
            biased_gap_terrain(terrain,
                                   platform_len=2.5,
                                   platform_height=0.0,
                                   num_gaps=self.num_goals - 2,
                                   gap_size=0.2 + 0.15*difficulty,
                                   x_range=[1.0, 1.6],
                                   corridor_half_width=corridor_half_width,
                                   lateral_offset=lateral_offset,
                                   gap_depth=[0.2, 1.0],
                                   pad_width=0.1,
                                   pad_height=0.5)
            self.add_roughness(terrain)
        elif terrain_name == "parkour_v2":
            parkour_v2_terrain(terrain,
                               platform_len=2.5,
                               platform_height=0.0,
                               num_segments=min(int(5 + 3*difficulty), 10),
                               segment_x_range=[1.5, 2.8],
                               difficulty=difficulty,
                               num_goals=self.num_goals,
                               pad_width=0.1,
                               pad_height=0.5)
            # parkour_v2_terrain already handles terrain setup, no need to add_roughness separately
        elif terrain_name == "narrow_gap":
            narrow_gap_difficulty = np.clip(difficulty, 0.0, 1.0)
            target_corridor_half_width = max(A1_NARROW_GAP_TARGET_WIDTH, A1_TOTAL_WIDTH) / 2.0
            corridor_half_width = [
                0.24 + narrow_gap_difficulty * (target_corridor_half_width - 0.24),
                0.32 + narrow_gap_difficulty * (target_corridor_half_width - 0.32),
            ]
            alternating_lateral_terrain(terrain,
                                        platform_len=2.5,
                                        platform_height=0.0,
                                        num_segments=self.num_goals - 2,
                                        x_range=[0.8, 1.4],
                                        corridor_half_width=corridor_half_width,
                                        lateral_offset=0.75,
                                        wall_height=1.0,
                                        pad_width=0.1,
                                        pad_height=0.5)
        elif terrain_name == "climbing_wall":
            climbing_wall_difficulty = np.clip(difficulty, 0.0, 1.0)
            target_hurdle_height = 0.25 * 1.5
            hurdle_height_range = [
                0.18 + climbing_wall_difficulty * (target_hurdle_height - 0.18),
                0.28 + climbing_wall_difficulty * (target_hurdle_height - 0.28),
            ]
            hurdle_depth_range = [
                0.45 + climbing_wall_difficulty * (0.3 - 0.45),
                0.70 + climbing_wall_difficulty * (0.35 - 0.70),
            ]
            top_platform_depth = 0.3 - climbing_wall_difficulty * 0.1
            slanted_hurdle_terrain(terrain,
                                   platform_len=2.5,
                                   platform_height=0.0,
                                   num_stones=self.num_goals - 2,
                                   x_range=[0.8, 1.4],
                                   y_range=self.cfg.y_range,
                                   half_valid_width=[0.45, 0.8],
                                   hurdle_height_range=hurdle_height_range,
                                   hurdle_depth_range=hurdle_depth_range,
                                   top_platform_depth=top_platform_depth,
                                   pad_width=0.1,
                                   pad_height=0.5)
            slanted_hurdle_mask = getattr(terrain, "slanted_hurdle_mask", None)
            slanted_hurdle_heights = terrain.height_field_raw.copy() if slanted_hurdle_mask is not None else None
            self.add_roughness(terrain)
            if slanted_hurdle_mask is not None:
                wall_roughness_scale = 0.25
                roughness_delta = terrain.height_field_raw.astype(np.int32) - slanted_hurdle_heights.astype(np.int32)
                terrain.height_field_raw[slanted_hurdle_mask] = (
                    slanted_hurdle_heights.astype(np.int32)
                    + np.rint(roughness_delta * wall_roughness_scale).astype(np.int32)
                )[slanted_hurdle_mask].astype(terrain.height_field_raw.dtype)
        elif terrain_name == "demo":
            demo_terrain(terrain)
            self.add_roughness(terrain)
        else:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=0.0, platform_size=3.)
        # np.set_printoptions(precision=2)
        # print(np.array(self.proportions), choice)
        terrain.idx = idx
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        # env_origin_x = (i + 0.5) * self.env_length
        env_origin_x = i * self.env_length + 1.0
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 0.5) / terrain.horizontal_scale) # within 1 meter square range
        x2 = int((self.env_length/2. + 0.5) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 0.5) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 0.5) / terrain.horizontal_scale)
        if self.cfg.origin_zero_z:
            env_origin_z = 0
        else:
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        self.terrain_type[i, j] = terrain.idx
        self.goals[i, j, :, :2] = terrain.goals + [i * self.env_length, j * self.env_width]
        # self.env_slope_vec[i, j] = terrain.slope_vector

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def gap_parkour_terrain(terrain, difficulty, platform_size=2.):
    gap_size = 0.1 + 0.3 * difficulty
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -400
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

    slope_angle = 0.1 + difficulty * 1
    offset = 1 + 9 * difficulty#10
    scale = 15
    wall_center_x = [center_x - x1, center_x, center_x + x1]
    wall_center_y = [center_y - y1, center_y, center_y + y1]

    # for i in range(center_y + y1, center_y + y2):
    #     for j in range(center_x-x1, center_x + x1):
    #         for w in wall_center_x:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[j, i] < height:
    #                 terrain.height_field_raw[j, i] = int(height)
    
    # for i in range(center_y - y2, center_y - y1):
    #     for j in range(center_x-x1, center_x + x1):
    #         for w in wall_center_x:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[j, i] < height:
    #                 terrain.height_field_raw[j, i] = int(height)

    # for i in range(center_x + x1, center_x + x2):
    #     for j in range(center_y-y1, center_y + y1):
    #         for w in wall_center_y:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[i, j] < height:
    #                 terrain.height_field_raw[i, j] = int(height)
    
    # for i in range(center_x - x2, center_x - x1):
    #     for j in range(center_y-y1, center_y + y1):
    #         for w in wall_center_y:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[i, j] < height:
    #                 terrain.height_field_raw[i, j] = int(height)

def parkour_terrain(terrain, 
                    platform_len=2.5, 
                    platform_height=0., 
                    num_stones=8, 
                    x_range=[1.8, 1.9], 
                    y_range=[0., 0.1], 
                    z_range=[-0.2, 0.2],
                    stone_len=1.0,
                    stone_width=0.6,
                    pad_width=0.1,
                    pad_height=0.5,
                    incline_height=0.1,
                    last_incline_height=0.6,
                    last_stone_len=1.6,
                    pit_depth=[0.5, 1.]):
    # 1st dimension: x, 2nd dimension: y
    goals = np.zeros((num_stones+2, 2))
    terrain.height_field_raw[:] = -round(np.random.uniform(pit_depth[0], pit_depth[1]) / terrain.vertical_scale)
    
    mid_y = terrain.length // 2  # length is actually y width
    stone_len = np.random.uniform(*stone_len)
    stone_len = 2 * round(stone_len / 2.0, 1)
    stone_len = round(stone_len / terrain.horizontal_scale)
    dis_x_min = stone_len + round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = stone_len + round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)
    dis_z_min = round(z_range[0] / terrain.vertical_scale)
    dis_z_max = round(z_range[1] / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    stone_width = round(stone_width / terrain.horizontal_scale)
    last_stone_len = round(last_stone_len / terrain.horizontal_scale)

    incline_height = round(incline_height / terrain.vertical_scale)
    last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len - np.random.randint(dis_x_min, dis_x_max) + stone_len // 2
    goals[0] = [platform_len -  stone_len // 2, mid_y]
    left_right_flag = np.random.randint(0, 2)
    # dis_z = np.random.randint(dis_z_min, dis_z_max)
    dis_z = 0
    
    for i in range(num_stones):
        dis_x += np.random.randint(dis_x_min, dis_x_max)
        pos_neg = round(2*(left_right_flag - 0.5))
        dis_y = mid_y + pos_neg * np.random.randint(dis_y_min, dis_y_max)
        if i == num_stones - 1:
            dis_x += last_stone_len // 4
            heights = np.tile(np.linspace(-last_incline_height, last_incline_height, stone_width), (last_stone_len, 1)) * pos_neg
            terrain.height_field_raw[dis_x-last_stone_len//2:dis_x+last_stone_len//2, dis_y-stone_width//2: dis_y+stone_width//2] = heights.astype(int) + dis_z
        else:
            heights = np.tile(np.linspace(-incline_height, incline_height, stone_width), (stone_len, 1)) * pos_neg
            terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, dis_y-stone_width//2: dis_y+stone_width//2] = heights.astype(int) + dis_z
        
        goals[i+1] = [dis_x, dis_y]

        left_right_flag = 1 - left_right_flag
    final_dis_x = dis_x + 2*np.random.randint(dis_x_min, dis_x_max)
    final_platform_start = dis_x + last_stone_len // 2 + round(0.05 // terrain.horizontal_scale)
    terrain.height_field_raw[final_platform_start:, :] = platform_height
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height
    
def parkour_gap_terrain(terrain,
                           platform_len=2.5, 
                           platform_height=0., 
                           num_gaps=8,
                           gap_size=0.3,
                           x_range=[1.6, 2.4],
                           y_range=[-1.2, 1.2],
                           half_valid_width=[0.6, 1.2],
                           gap_depth=-200,
                           pad_width=0.1,
                           pad_height=0.5,
                           flat=False):
    goals = np.zeros((num_gaps+2, 2))
    # terrain.height_field_raw[:] = -200
    # import ipdb; ipdb.set_trace()
    mid_y = terrain.length // 2  # length is actually y width

    # dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    # dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    gap_depth = -round(np.random.uniform(gap_depth[0], gap_depth[1]) / terrain.vertical_scale)
    
    # half_gap_width = round(np.random.uniform(0.6, 1.2) / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    # terrain.height_field_raw[:, :mid_y-half_valid_width] = gap_depth
    # terrain.height_field_raw[:, mid_y+half_valid_width:] = gap_depth
    
    terrain.height_field_raw[0:platform_len, :] = platform_height

    gap_size = round(gap_size / terrain.horizontal_scale)
    dis_x_min = round(x_range[0] / terrain.horizontal_scale) + gap_size
    dis_x_max = round(x_range[1] / terrain.horizontal_scale) + gap_size

    dis_x = platform_len
    goals[0] = [platform_len - 1, mid_y]
    last_dis_x = dis_x
    for i in range(num_gaps):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        dis_x += rand_x
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        if not flat:
            # terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, ] = np.random.randint(hurdle_height_min, hurdle_height_max)
            # terrain.height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2, 
            #                          gap_center-half_gap_width:gap_center+half_gap_width] = gap_depth
            terrain.height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2, :] = gap_depth

        terrain.height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = gap_depth
        terrain.height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = gap_depth
        
        last_dis_x = dis_x
        goals[i+1] = [dis_x-rand_x//2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

def parkour_hurdle_terrain(terrain,
                           platform_len=2.5, 
                           platform_height=0., 
                           num_stones=8,
                           stone_len=0.3,
                           x_range=[1.5, 2.4],
                           y_range=[-0.4, 0.4],
                           half_valid_width=[0.4, 0.8],
                           hurdle_height_range=[0.2, 0.3],
                           pad_width=0.1,
                           pad_height=0.5,
                           flat=False):
    goals = np.zeros((num_stones+2, 2))
    # terrain.height_field_raw[:] = -200
    
    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    # half_valid_width = round(np.random.uniform(y_range[1]+0.2, y_range[1]+1) / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    hurdle_height_max = round(hurdle_height_range[1] / terrain.vertical_scale)
    hurdle_height_min = round(hurdle_height_range[0] / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    stone_len = round(stone_len / terrain.horizontal_scale)
    # stone_width = round(stone_width / terrain.horizontal_scale)
    
    # incline_height = round(incline_height / terrain.vertical_scale)
    # last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len
    goals[0] = [platform_len - 1, mid_y]
    last_dis_x = dis_x
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        dis_x += rand_x
        if not flat:
            terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, ] = np.random.randint(hurdle_height_min, hurdle_height_max)
            terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, :mid_y+rand_y-half_valid_width] = 0
            terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, mid_y+rand_y+half_valid_width:] = 0
        last_dis_x = dis_x
        goals[i+1] = [dis_x-rand_x//2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    # terrain.height_field_raw[:, :max(mid_y-half_valid_width, 0)] = 0
    # terrain.height_field_raw[:, min(mid_y+half_valid_width, terrain.height_field_raw.shape[1]):] = 0
    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

def parkour_step_terrain(terrain,
                           platform_len=2.5, 
                           platform_height=0., 
                           num_stones=8,
                        #    x_range=[1.5, 2.4],
                            x_range=[0.2, 0.4],
                           y_range=[-0.15, 0.15],
                           half_valid_width=[0.45, 0.5],
                           step_height = 0.2,
                           pad_width=0.1,
                           pad_height=0.5):
    goals = np.zeros((num_stones+2, 2))
    # terrain.height_field_raw[:] = -200
    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round( (x_range[0] + step_height) / terrain.horizontal_scale)
    dis_x_max = round( (x_range[1] + step_height) / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    step_height = round(step_height / terrain.vertical_scale)

    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    # stone_width = round(stone_width / terrain.horizontal_scale)
    
    # incline_height = round(incline_height / terrain.vertical_scale)
    # last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len
    last_dis_x = dis_x
    stair_height = 0
    goals[0] = [platform_len - round(1 / terrain.horizontal_scale), mid_y]
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        if i < num_stones // 2:
            stair_height += step_height
        elif i > num_stones // 2:
            stair_height -= step_height
        terrain.height_field_raw[dis_x:dis_x+rand_x, ] = stair_height
        dis_x += rand_x
        terrain.height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = 0
        terrain.height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = 0
        
        last_dis_x = dis_x
        goals[i+1] = [dis_x-rand_x//2, mid_y+rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    # terrain.height_field_raw[:, :max(mid_y-half_valid_width, 0)] = 0
    # terrain.height_field_raw[:, min(mid_y+half_valid_width, terrain.height_field_raw.shape[1]):] = 0
    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height


# new terrain

def _sanitize_goals_on_safe_cells(terrain, goals, unsafe_height):
    """Move generated goals off pits while keeping them close to the intended route."""
    safe_goals = []
    search_radius = round(0.5 / terrain.horizontal_scale)
    for goal in np.asarray(goals):
        x = int(round(goal[0]))
        y = int(round(goal[1]))
        x = int(np.clip(x, 0, terrain.width - 1))
        y = int(np.clip(y, 0, terrain.length - 1))
        if terrain.height_field_raw[x, y] > unsafe_height:
            safe_goals.append([x, y])
            continue

        x0 = max(x - search_radius, 0)
        x1 = min(x + search_radius + 1, terrain.width)
        y0 = max(y - search_radius, 0)
        y1 = min(y + search_radius + 1, terrain.length)
        safe_cells = np.argwhere(terrain.height_field_raw[x0:x1, y0:y1] > unsafe_height)
        if len(safe_cells) == 0:
            safe_goals.append([x, y])
            continue

        safe_cells[:, 0] += x0
        safe_cells[:, 1] += y0
        distances = np.sum((safe_cells - np.array([x, y])) ** 2, axis=1)
        safe_goals.append(safe_cells[np.argmin(distances)].tolist())
    return np.asarray(safe_goals)

def slanted_hurdle_terrain(
        terrain,
        platform_len=2.5,
        platform_height=0.0,
        num_stones=8,
        x_range=[1.2, 2.0],
        y_range=[-0.3, 0.3],
        half_valid_width=[0.45, 0.8],
        hurdle_height_range=[0.12, 0.28],
        hurdle_depth_range=[0.35, 0.7],
        top_platform_depth=0.0,
        pad_width=0.1,
        pad_height=0.5,
        flat=False):
    """
    斜顶矮墙：
    与 parkour_hurdle_terrain 类似，但障碍不是竖直薄墙，而是沿 x 方向渐变高度的斜顶障碍。
    """
    goals = np.zeros((num_stones + 2, 2))
    mid_y = terrain.length // 2

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    top_platform_depth = round(top_platform_depth / terrain.horizontal_scale)

    terrain.height_field_raw[0:platform_len, :] = platform_height
    slanted_hurdle_mask = np.zeros_like(terrain.height_field_raw, dtype=bool)

    dis_x = platform_len
    goals[0] = [platform_len / 2, mid_y]

    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        dis_x += rand_x

        hurdle_depth = round(np.random.uniform(hurdle_depth_range[0], hurdle_depth_range[1]) / terrain.horizontal_scale)
        hurdle_height = round(np.random.uniform(hurdle_height_range[0], hurdle_height_range[1]) / terrain.vertical_scale)

        start_x = max(dis_x - hurdle_depth // 2, 0)
        end_x = min(dis_x + hurdle_depth // 2, terrain.width)

        if not flat and end_x > start_x:
            heights = np.linspace(0, hurdle_height, end_x - start_x).astype(np.int16)
            terrain.height_field_raw[start_x:end_x, :] = heights[:, None]

            center_y = mid_y + rand_y
            terrain.height_field_raw[start_x:end_x, :max(center_y - half_valid_width, 0)] = 0
            terrain.height_field_raw[start_x:end_x, min(center_y + half_valid_width, terrain.length):] = 0
            valid_y0 = max(center_y - half_valid_width, 0)
            valid_y1 = min(center_y + half_valid_width, terrain.length)
            slanted_hurdle_mask[start_x:end_x, valid_y0:valid_y1] = True

            if top_platform_depth > 0:
                platform_end_x = min(end_x + top_platform_depth, terrain.width)
                terrain.height_field_raw[end_x:platform_end_x, :] = hurdle_height
                terrain.height_field_raw[end_x:platform_end_x, :max(center_y - half_valid_width, 0)] = 0
                terrain.height_field_raw[end_x:platform_end_x, min(center_y + half_valid_width, terrain.length):] = 0
                slanted_hurdle_mask[end_x:platform_end_x, valid_y0:valid_y1] = True

        if top_platform_depth > 0 and end_x > start_x:
            goals[i + 1] = [(end_x + min(end_x + top_platform_depth, terrain.width)) / 2, mid_y + rand_y]
        else:
            goals[i + 1] = [dis_x, mid_y + rand_y]

    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - round(0.5 / terrain.horizontal_scale)
    goals[-1] = [final_dis_x, mid_y]

    terrain.goals = goals * terrain.horizontal_scale

    pad_width = int(pad_width / terrain.horizontal_scale)
    pad_height = int(pad_height / terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height
    terrain.slanted_hurdle_mask = slanted_hurdle_mask


def alternating_step_terrain(
        terrain,
        platform_len=2.5,
        platform_height=0.0,
        num_steps=8,
        x_range=[0.35, 0.8],
        y_range=[-0.2, 0.2],
        half_valid_width=[0.45, 0.7],
        step_height=0.12,
        pad_width=0.1,
        pad_height=0.5):
    """
    锯齿台阶：
    多个连续台阶，高度上下交替变化（+h, -h, +h, -h...），
    横向通道中心也可轻微漂移。
    """
    goals = np.zeros((num_steps + 2, 2))
    mid_y = terrain.length // 2

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)

    step_height = round(step_height / terrain.vertical_scale)
    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)

    terrain.height_field_raw[0:platform_len, :] = platform_height

    dis_x = platform_len
    last_dis_x = dis_x
    goals[0] = [platform_len - 1, mid_y]

    for i in range(num_steps):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)

        sign = 1 if i % 2 == 0 else -1
        cur_h = sign * step_height

        terrain.height_field_raw[dis_x:dis_x + rand_x, :] = cur_h
        dis_x += rand_x

        center_y = mid_y + rand_y
        terrain.height_field_raw[last_dis_x:dis_x, :max(center_y - half_valid_width, 0)] = 0
        terrain.height_field_raw[last_dis_x:dis_x, min(center_y + half_valid_width, terrain.length):] = 0

        goals[i + 1] = [dis_x - rand_x // 2, center_y]
        last_dis_x = dis_x

    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - round(0.5 / terrain.horizontal_scale)
    goals[-1] = [final_dis_x, mid_y]

    terrain.goals = goals * terrain.horizontal_scale

    pad_width = int(pad_width / terrain.horizontal_scale)
    pad_height = int(pad_height / terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height


def beam_gap_terrain(
        terrain,
        platform_len=2.5,
        platform_height=0.0,
        num_gaps=8,
        gap_size=0.28,
        x_range=[1.0, 1.6],
        beam_width_range=[0.5, 0.8],
        y_offset_range=[-0.08, 0.08],
        gap_depth=[0.2, 1.0],
        pad_width=0.1,
        pad_height=0.5):
    """
    窄梁缺口：
    大部分区域为深坑，只保留一条窄梁；沿窄梁连续挖多个 gap。
    """
    goals = np.zeros((num_gaps + 2, 2))
    mid_y = terrain.length // 2

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    beam_width = np.random.uniform(beam_width_range[0], beam_width_range[1])
    gap_size = round(gap_size / terrain.horizontal_scale)
    gap_depth = -round(np.random.uniform(gap_depth[0], gap_depth[1]) / terrain.vertical_scale)

    dis_x_min = round(x_range[0] / terrain.horizontal_scale) + gap_size
    dis_x_max = round(x_range[1] / terrain.horizontal_scale) + gap_size
    beam_width = max(int(np.ceil(beam_width / terrain.horizontal_scale)), 1)
    beam_half_width = beam_width // 2

    terrain.height_field_raw[:, :] = gap_depth
    terrain.height_field_raw[0:platform_len, :] = platform_height
    beam_start = mid_y - beam_half_width
    beam_end = beam_start + beam_width
    terrain.height_field_raw[0:platform_len, beam_start:beam_end] = platform_height

    dis_x = platform_len
    goals[0] = [platform_len - 1, mid_y]
    last_beam_center = mid_y

    for i in range(num_gaps):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = round(np.random.uniform(y_offset_range[0], y_offset_range[1]) / terrain.horizontal_scale)
        dis_x += rand_x

        beam_center = int(np.clip(mid_y + rand_y, beam_half_width + 1, terrain.length - (beam_width - beam_half_width) - 1))
        last_beam_center = beam_center
        beam_start = beam_center - beam_half_width
        beam_end = beam_start + beam_width

        last_safe_x = max(dis_x - rand_x, 0)
        terrain.height_field_raw[last_safe_x:dis_x, beam_start:beam_end] = platform_height

        gap_start = max(dis_x - gap_size // 2, 0)
        gap_end = min(gap_start + gap_size, terrain.width)
        terrain.height_field_raw[gap_start:gap_end, beam_start:beam_end] = gap_depth

        goals[i + 1] = [dis_x - rand_x // 2, beam_center]

    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - round(0.5 / terrain.horizontal_scale)

    beam_start = last_beam_center - beam_half_width
    beam_end = beam_start + beam_width
    terrain.height_field_raw[dis_x:final_dis_x, beam_start:beam_end] = platform_height
    goals[-1] = [(dis_x + final_dis_x) / 2, last_beam_center]

    terrain.goals = goals * terrain.horizontal_scale

    pad_width = int(pad_width / terrain.horizontal_scale)
    pad_height = int(pad_height / terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height


def biased_gap_terrain(
        terrain,
        platform_len=2.5,
        platform_height=0.0,
        num_gaps=8,
        gap_size=0.3,
        x_range=[1.0, 1.6],
        corridor_half_width=0.6,
        lateral_offset=0.8,
        gap_depth=[0.2, 1.0],
        pad_width=0.1,
        pad_height=0.5):
    """
    完全错开缺口平台：
    相邻两个平台完全左右错开（交替偏移），goal点放在相邻平台可通行区域之间的最近处。
    """
    goals = np.zeros((num_gaps + 2, 2))
    mid_y = terrain.length // 2

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    gap_size = round(gap_size / terrain.horizontal_scale)
    gap_depth = -round(np.random.uniform(gap_depth[0], gap_depth[1]) / terrain.vertical_scale)

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    corridor_half_width = round(corridor_half_width / terrain.horizontal_scale)
    lateral_offset = round(lateral_offset / terrain.horizontal_scale)

    terrain.height_field_raw[0:platform_len, :] = platform_height

    dis_x = platform_len
    last_dis_x = dis_x
    goals[0] = [platform_len / 2, mid_y]

    for i in range(num_gaps):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        dis_x += rand_x

        # 相邻平台完全交替错开：左-右-左-右...
        sign = 1 if i % 2 == 0 else -1
        center_y = int(round(np.clip(
            mid_y + sign * lateral_offset,
            corridor_half_width + 1,
            terrain.length - corridor_half_width - 1,
        )))
        corridor_start = max(center_y - corridor_half_width, 0)
        corridor_end = min(center_y + corridor_half_width, terrain.length)

        # 当前平台：设置缺口为除去可通行区域外的部分
        terrain.height_field_raw[last_dis_x:dis_x, :corridor_start] = gap_depth
        terrain.height_field_raw[last_dis_x:dis_x, corridor_end:] = gap_depth

        goals[i + 1] = [(last_dis_x + dis_x) / 2, center_y]

        # 平台端缺口：使机器狗必须精确跳跃
        gap_start = max(dis_x - gap_size // 2, 0)
        gap_end = min(dis_x + gap_size // 2, terrain.width)
        terrain.height_field_raw[gap_start:gap_end, :] = gap_depth

        last_dis_x = dis_x

    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - round(0.5 / terrain.horizontal_scale)
    goals[-1] = [final_dis_x, mid_y]

    terrain.goals = goals * terrain.horizontal_scale

    pad_width = int(pad_width / terrain.horizontal_scale)
    pad_height = int(pad_height / terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height


def alternating_lateral_terrain(
        terrain,
        platform_len=2.5,
        platform_height=0.0,
        num_segments=8,
        x_range=[0.8, 1.4],
        corridor_half_width=[0.5, 0.8],
        lateral_offset=0.6,
        wall_height=0.25,
        pad_width=0.1,
        pad_height=0.5):
    """
    交替侧跳赛道：
    多个方形凹坑左右交替分布，相邻凹坑仅由狭窄斜向走廊连接。
    """
    goals = np.zeros((num_segments + 2, 2))
    mid_y = terrain.length // 2

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    corridor_width = max(int(np.ceil(np.random.uniform(corridor_half_width[0], corridor_half_width[1]) * 2 / terrain.horizontal_scale)), 1)
    corridor_half_width = corridor_width // 2
    lateral_offset = round(lateral_offset / terrain.horizontal_scale)
    wall_height = round(wall_height / terrain.vertical_scale)
    pit_half_size = max(round(0.45 / terrain.horizontal_scale), corridor_half_width * 3)

    terrain.height_field_raw[:, :] = wall_height
    terrain.height_field_raw[0:platform_len, :] = platform_height

    def carve_square(center_x, center_y, half_size):
        x0 = max(center_x - half_size, 0)
        x1 = min(center_x + half_size + 1, terrain.width)
        y0 = max(center_y - half_size, 0)
        y1 = min(center_y + half_size + 1, terrain.length)
        terrain.height_field_raw[x0:x1, y0:y1] = platform_height

    def carve_diagonal_corridor(start_x, start_y, end_x, end_y):
        num_points = max(abs(end_x - start_x), abs(end_y - start_y), 1) + 1
        xs = np.linspace(start_x, end_x, num_points).round().astype(int)
        ys = np.linspace(start_y, end_y, num_points).round().astype(int)
        for x, y in zip(xs, ys):
            x0 = max(x - corridor_half_width, 0)
            x1 = min(x0 + corridor_width, terrain.width)
            y0 = max(y - corridor_half_width, 0)
            y1 = min(y0 + corridor_width, terrain.length)
            terrain.height_field_raw[x0:x1, y0:y1] = platform_height

    dis_x = platform_len
    last_dis_x = dis_x
    goals[0] = [platform_len // 2, mid_y]
    last_center_x = platform_len - 1
    last_center_y = mid_y

    for i in range(num_segments):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        sign = 1 if i % 2 == 0 else -1
        target_center_y = int(round(np.clip(
            mid_y + sign * lateral_offset,
            corridor_half_width + 1,
            terrain.length - corridor_half_width - 1,
        )))

        pit_center_x = min(last_dis_x + rand_x, terrain.width - pit_half_size - 1)
        carve_diagonal_corridor(last_center_x, last_center_y, pit_center_x, target_center_y)
        carve_square(pit_center_x, target_center_y, pit_half_size)

        dis_x = pit_center_x
        goals[i + 1] = [pit_center_x, target_center_y]
        last_dis_x = dis_x
        last_center_x = pit_center_x
        last_center_y = target_center_y

    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    if final_dis_x > terrain.width - pit_half_size - 1:
        final_dis_x = terrain.width - pit_half_size - 1
    carve_diagonal_corridor(last_center_x, last_center_y, final_dis_x, mid_y)
    carve_square(final_dis_x, mid_y, pit_half_size)
    goals[-1] = [final_dis_x, mid_y]

    terrain.goals = goals * terrain.horizontal_scale

    pad_width = int(pad_width / terrain.horizontal_scale)
    pad_height = int(pad_height / terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height


def cliff_side_terrain(
        terrain,
        platform_len=2.5,
        platform_height=0.0,
        num_segments=8,
        x_range=[0.8, 1.4],
        corridor_width_range=[1.0, 1.5],
        cliff_depth=[0.3, 1.0],
        cliff_side=None,
        add_obstacles=True,
        obstacle_height_range=[0.08, 0.18],
        obstacle_depth_range=[0.18, 0.35],
        pad_width=0.1,
        pad_height=0.5):
    """
    单侧悬崖赛道：
    一侧持续为悬崖，另一侧为可行走走廊；走廊上可加一些低矮小障碍。
    """
    goals = np.zeros((num_segments + 2, 2))
    mid_y = terrain.length // 2

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    cliff_depth = -round(np.random.uniform(cliff_depth[0], cliff_depth[1]) / terrain.vertical_scale)

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)

    corridor_width = round(np.random.uniform(corridor_width_range[0], corridor_width_range[1]) / terrain.horizontal_scale)
    obstacle_h_min = round(obstacle_height_range[0] / terrain.vertical_scale)
    obstacle_h_max = round(obstacle_height_range[1] / terrain.vertical_scale)
    obstacle_d_min = round(obstacle_depth_range[0] / terrain.horizontal_scale)
    obstacle_d_max = round(obstacle_depth_range[1] / terrain.horizontal_scale)

    if cliff_side is None:
        cliff_side = "left" if np.random.rand() < 0.5 else "right"

    terrain.height_field_raw[:, :] = 0
    terrain.height_field_raw[0:platform_len, :] = platform_height

    if cliff_side == "left":
        corridor_start = terrain.length - corridor_width
        corridor_end = terrain.length
        terrain.height_field_raw[:, :corridor_start] = cliff_depth
        corridor_center = (corridor_start + corridor_end) // 2
    else:
        corridor_start = 0
        corridor_end = corridor_width
        terrain.height_field_raw[:, corridor_end:] = cliff_depth
        corridor_center = (corridor_start + corridor_end) // 2

    dis_x = platform_len
    goals[0] = [platform_len - 1, corridor_center]

    for i in range(num_segments):
        seg_len = np.random.randint(dis_x_min, dis_x_max)
        start_x = dis_x
        end_x = min(dis_x + seg_len, terrain.width)

        if add_obstacles:
            obs_depth = np.random.randint(obstacle_d_min, obstacle_d_max + 1)
            obs_height = np.random.randint(obstacle_h_min, obstacle_h_max + 1)
            obs_x = min(start_x + seg_len // 2, terrain.width - 1)

            if cliff_side == "left":
                safe_margin = max(corridor_start + round(0.15 / terrain.horizontal_scale), 0)
                obs_y0 = safe_margin
                obs_y1 = min(corridor_end - round(0.2 / terrain.horizontal_scale), terrain.length)
            else:
                obs_y0 = max(corridor_start + round(0.2 / terrain.horizontal_scale), 0)
                obs_y1 = max(corridor_end - round(0.15 / terrain.horizontal_scale), obs_y0 + 1)

            terrain.height_field_raw[max(obs_x - obs_depth // 2, 0):min(obs_x + obs_depth // 2, terrain.width),
                                     obs_y0:obs_y1] = obs_height

        goals[i + 1] = [start_x + (end_x - start_x) // 2, corridor_center]
        dis_x = end_x

    final_dis_x = min(dis_x + np.random.randint(dis_x_min, dis_x_max), terrain.width - round(0.5 / terrain.horizontal_scale))
    goals[-1] = [final_dis_x, corridor_center]

    terrain.goals = goals * terrain.horizontal_scale

    pad_width = int(pad_width / terrain.horizontal_scale)
    pad_height = int(pad_height / terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = max(terrain.height_field_raw[:, :pad_width].max(), pad_height)
    terrain.height_field_raw[:, -pad_width:] = max(terrain.height_field_raw[:, -pad_width:].max(), pad_height)
    terrain.height_field_raw[:pad_width, :] = max(terrain.height_field_raw[:pad_width, :].max(), pad_height)
    terrain.height_field_raw[-pad_width:, :] = max(terrain.height_field_raw[-pad_width:, :].max(), pad_height)


def parkour_v2_terrain(
        terrain,
        platform_len=2.5,
        platform_height=0.0,
        num_segments=6,
        segment_x_range=[1.5, 2.8],
        difficulty=0.0,
        num_goals=8,
        pad_width=0.1,
        pad_height=0.5):
    """
    parkour_v2:
    终极混合赛道。将多种障碍片段随机组合：
      - slanted_hurdle
      - alternating_step
      - beam_gap
      - biased_gap
      - narrow_gap

    注意：
    这里采用“在同一个高度图上按段拼接”的写法，而不是递归调用别的 terrain 函数，
    避免这些函数各自把整张图重置掉。
    """
    terrain.height_field_raw[:, :] = 0
    difficulty = np.clip(difficulty, 0.0, 1.0)

    mid_y = terrain.length // 2
    platform_len_px = round(platform_len / terrain.horizontal_scale)
    platform_height_px = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len_px, :] = platform_height_px

    goals = [[platform_len_px // 2, mid_y]]
    key_goal_ids = {0}

    def add_goal(goal, key=False):
        goals.append(goal)
        if key:
            key_goal_ids.add(len(goals) - 1)

    cur_x = platform_len_px

    gap_depth = -round(np.random.uniform(0.25, 1.0) / terrain.vertical_scale)
    target_hurdle_height = 0.4
    hurdle_height_range = [
        0.18 + difficulty * (target_hurdle_height - 0.18),
        0.28 + difficulty * (target_hurdle_height - 0.28),
    ]
    hurdle_depth_range = [
        0.45 + difficulty * (0.3 - 0.45),
        0.70 + difficulty * (0.35 - 0.70),
    ]
    top_platform_depth = 0.35 - 0.1 * difficulty
    target_beam_width = max(A1_BEAM_GAP_TARGET_WIDTH, A1_TOTAL_WIDTH)
    beam_width_range = [
        0.4 + difficulty * (target_beam_width - 0.4),
        0.8 + difficulty * (target_beam_width - 0.8),
    ]
    beam_gap_size = 0.25 + difficulty * 0.55
    target_corridor_half_width = max(A1_NARROW_GAP_TARGET_WIDTH, A1_TOTAL_WIDTH) / 2.0
    narrow_corridor_half_width = np.random.uniform(
        0.24 + difficulty * (target_corridor_half_width - 0.24),
        0.32 + difficulty * (target_corridor_half_width - 0.32),
    )

    last_transition_goal = None
    last_transition_goal_added = False

    for seg_id in range(num_segments):
        seg_len = round(np.random.uniform(segment_x_range[0], segment_x_range[1]) / terrain.horizontal_scale)
        if cur_x + seg_len >= terrain.width - round(1.0 / terrain.horizontal_scale):
            break

        seg_type = np.random.choice(["slanted_hurdle", "alternating_step", "beam_gap", "biased_gap", "narrow_gap"])
        segment_backup = terrain.height_field_raw.copy()
        goals_backup = list(goals)
        key_goal_ids_backup = set(key_goal_ids)
        cur_x_backup = cur_x
        last_transition_goal_backup = last_transition_goal
        last_transition_goal_added_backup = last_transition_goal_added

        start_x = cur_x
        end_x = min(cur_x + seg_len, terrain.width)

        if seg_type == "beam_gap" and last_transition_goal is not None and not last_transition_goal_added:
            add_goal(last_transition_goal, key=True)
            last_transition_goal_added = True

        if seg_type == "slanted_hurdle":
            # 随机障碍数量（1-3个）
            num_hurdles = np.random.randint(1, 4)
            center_y = mid_y + np.random.randint(
                round(-0.35 / terrain.horizontal_scale),
                round(0.35 / terrain.horizontal_scale) + 1
            )
            half_valid_width = round(np.random.uniform(0.5, 0.8) / terrain.horizontal_scale)
            
            # 在segment内分布多个障碍
            dis_x = start_x
            hurdle_spacing_min = round(0.4 / terrain.horizontal_scale)
            hurdle_spacing_max = round(0.8 / terrain.horizontal_scale)
            
            for hurdle_idx in range(num_hurdles):
                if dis_x >= end_x - round(0.5 / terrain.horizontal_scale):
                    break
                    
                # 每个障碍之间的随机间距
                if hurdle_idx > 0:
                    spacing = np.random.randint(hurdle_spacing_min, hurdle_spacing_max + 1)
                    dis_x += spacing
                
                if dis_x >= end_x:
                    break
                
                hurdle_depth = max(round(np.random.uniform(hurdle_depth_range[0], hurdle_depth_range[1]) / terrain.horizontal_scale), 2)
                hurdle_height = round(np.random.uniform(hurdle_height_range[0], hurdle_height_range[1]) / terrain.vertical_scale)
                
                obs_x0 = max(dis_x - hurdle_depth // 2, start_x)
                obs_x1 = min(dis_x + hurdle_depth // 2, end_x)
                
                if obs_x1 > obs_x0:
                    heights = np.linspace(0, hurdle_height, max(obs_x1 - obs_x0, 1)).astype(np.int16)
                    terrain.height_field_raw[obs_x0:obs_x1, :] = heights[:, None]
                    terrain.height_field_raw[obs_x0:obs_x1, :max(center_y - half_valid_width, 0)] = 0
                    terrain.height_field_raw[obs_x0:obs_x1, min(center_y + half_valid_width, terrain.length):] = 0

                    platform_len = round(top_platform_depth / terrain.horizontal_scale)
                    platform_x1 = min(obs_x1 + platform_len, end_x)
                    if platform_x1 > obs_x1:
                        terrain.height_field_raw[obs_x1:platform_x1, :] = hurdle_height
                        terrain.height_field_raw[obs_x1:platform_x1, :max(center_y - half_valid_width, 0)] = 0
                        terrain.height_field_raw[obs_x1:platform_x1, min(center_y + half_valid_width, terrain.length):] = 0
                    
                    add_goal([(obs_x1 + max(platform_x1, obs_x1)) // 2, center_y], key=True)

        elif seg_type == "alternating_step":
            center_y = mid_y + np.random.randint(
                round(-0.25 / terrain.horizontal_scale),
                round(0.25 / terrain.horizontal_scale) + 1
            )
            half_valid_width = round(np.random.uniform(0.45, 0.7) / terrain.horizontal_scale)
            num_local = np.random.randint(2, 5)
            local_w = max(seg_len // num_local, 1)
            step_h = round(np.random.uniform(0.08, 0.16) / terrain.vertical_scale)

            for k in range(num_local):
                sx0 = start_x + k * local_w
                sx1 = min(start_x + (k + 1) * local_w, end_x)
                sign = 1 if k % 2 == 0 else -1
                terrain.height_field_raw[sx0:sx1, :] = sign * step_h
                terrain.height_field_raw[sx0:sx1, :max(center_y - half_valid_width, 0)] = 0
                terrain.height_field_raw[sx0:sx1, min(center_y + half_valid_width, terrain.length):] = 0

                add_goal([(sx0 + sx1) // 2, center_y], key=True)

        elif seg_type == "beam_gap":
            beam_width = max(int(np.ceil(np.random.uniform(beam_width_range[0], beam_width_range[1]) / terrain.horizontal_scale)), 1)
            beam_half_width = beam_width // 2
            beam_center = int(np.clip(
                mid_y + np.random.randint(
                    round(-0.08 / terrain.horizontal_scale),
                    round(0.08 / terrain.horizontal_scale) + 1
                ),
                beam_half_width + 1,
                terrain.length - (beam_width - beam_half_width) - 1,
            ))
            beam_start = beam_center - beam_half_width
            beam_end = beam_start + beam_width
            terrain.height_field_raw[start_x:end_x, :] = gap_depth
            terrain.height_field_raw[start_x:end_x, beam_start:beam_end] = 0

            num_local = np.random.randint(1, 3)
            gaps = []
            for _ in range(num_local):
                gap_w = round(beam_gap_size / terrain.horizontal_scale)
                gx = np.random.randint(start_x + gap_w, end_x - gap_w) if end_x - start_x > 2 * gap_w else start_x + (end_x - start_x) // 2
                gap_x0 = max(gx - gap_w // 2, start_x)
                gap_x1 = min(gap_x0 + gap_w, end_x)
                terrain.height_field_raw[gap_x0:gap_x1, beam_start:beam_end] = gap_depth
                gaps.append((gap_x0, gap_x1))

            safe_start = start_x
            for gap_x0, gap_x1 in sorted(gaps):
                if gap_x0 - safe_start > round(0.25 / terrain.horizontal_scale):
                    add_goal([(safe_start + gap_x0) // 2, beam_center], key=True)
                safe_start = max(safe_start, gap_x1)
            if end_x - safe_start > round(0.25 / terrain.horizontal_scale):
                add_goal([(safe_start + end_x) // 2, beam_center], key=True)

        elif seg_type == "biased_gap":
            # 在一个segment内放置3个相邻的小平台，左右交替错开
            num_platforms = 3
            platform_size = seg_len // num_platforms
            corridor_half_width_m = 0.6 - 0.1 * difficulty
            max_edge_distance = 0.38 * 1.5
            target_lateral_offset_m = corridor_half_width_m + max_edge_distance / 2.0
            lateral_offset_m = 0.25 + 0.3 *difficulty
            corridor_half_width = round(corridor_half_width_m / terrain.horizontal_scale)
            lateral_offset = round(lateral_offset_m / terrain.horizontal_scale)
            local_gap_size = round((0.2 + 0.15 * difficulty) / terrain.horizontal_scale)
            
            for plat_id in range(num_platforms):
                # 每个平台的位置
                plat_start = start_x + plat_id * platform_size
                plat_end = min(start_x + (plat_id + 1) * platform_size, end_x)
                
                # 交替的左右偏移
                sign = 1 if plat_id % 2 == 0 else -1
                center_y = int(round(np.clip(
                    mid_y + sign * lateral_offset,
                    corridor_half_width + 1,
                    terrain.length - corridor_half_width - 1,
                )))
                
                # 设置该小平台的可通行区域（中心），两侧为缺口
                terrain.height_field_raw[plat_start:plat_end, :max(center_y - corridor_half_width, 0)] = gap_depth
                terrain.height_field_raw[plat_start:plat_end, min(center_y + corridor_half_width, terrain.length):] = gap_depth
                add_goal([plat_start + (plat_end - plat_start) // 2, center_y], key=True)
                if plat_id < num_platforms - 1:
                    gap_x0 = max(plat_end - local_gap_size // 2, start_x)
                    gap_x1 = min(gap_x0 + local_gap_size, end_x)
                    terrain.height_field_raw[gap_x0:gap_x1, :] = gap_depth

        elif seg_type == "narrow_gap":
            corridor_width = max(int(np.ceil(narrow_corridor_half_width * 2 / terrain.horizontal_scale)), 1)
            corridor_half_width = corridor_width // 2
            lateral_offset = round(0.75 / terrain.horizontal_scale)
            wall_height = round((0.55 + 0.35 * difficulty) / terrain.vertical_scale)
            pit_half_size = max(round(0.45 / terrain.horizontal_scale), corridor_half_width * 3)

            def carve_square(center_x, center_y):
                x0 = max(center_x - pit_half_size, start_x)
                x1 = min(center_x + pit_half_size + 1, end_x)
                y0 = max(center_y - pit_half_size, 0)
                y1 = min(center_y + pit_half_size + 1, terrain.length)
                terrain.height_field_raw[x0:x1, y0:y1] = platform_height_px

            def carve_diagonal_corridor(x0, y0, x1, y1):
                num_points = max(abs(x1 - x0), abs(y1 - y0), 1) + 1
                xs = np.linspace(x0, x1, num_points).round().astype(int)
                ys = np.linspace(y0, y1, num_points).round().astype(int)
                for x, y in zip(xs, ys):
                    cx0 = max(x - corridor_half_width, start_x)
                    cx1 = min(cx0 + corridor_width, end_x)
                    cy0 = max(y - corridor_half_width, 0)
                    cy1 = min(cy0 + corridor_width, terrain.length)
                    terrain.height_field_raw[cx0:cx1, cy0:cy1] = platform_height_px

            terrain.height_field_raw[start_x:end_x, :] = wall_height
            pit_centers = []
            for pit_id in range(2):
                sign = 1 if pit_id % 2 == 0 else -1
                center_x = start_x + (pit_id + 1) * seg_len // 3
                center_y = int(round(np.clip(
                    mid_y + sign * lateral_offset,
                    pit_half_size + 1,
                    terrain.length - pit_half_size - 1,
                )))
                pit_centers.append((center_x, center_y))

            prev_x, prev_y = start_x, mid_y
            for center_x, center_y in pit_centers:
                carve_diagonal_corridor(prev_x, prev_y, center_x, center_y)
                carve_square(center_x, center_y)
                add_goal([center_x, center_y], key=True)
                prev_x, prev_y = center_x, center_y
            final_room_x = max(end_x - pit_half_size - 1, start_x)
            carve_diagonal_corridor(prev_x, prev_y, final_room_x, mid_y)
            carve_square(final_room_x, mid_y)
            add_goal([final_room_x, mid_y], key=True)

        cur_x = end_x
        
        # 在障碍段之后添加过渡平台
        if seg_id < num_segments - 1:
            transition_len = round(np.random.uniform(1.0, 1.5) / terrain.horizontal_scale)
            transition_start = cur_x
            transition_end = min(cur_x + transition_len, terrain.width - round(1.0 / terrain.horizontal_scale))
            
            # 创建安全的平台
            terrain.height_field_raw[transition_start:transition_end, :] = platform_height_px
            
            # 在过渡平台中心添加goal点
            transition_mid = transition_start + (transition_end - transition_start) // 2
            last_transition_goal = [transition_mid, mid_y]
            last_transition_goal_added = False
            if seg_type == "beam_gap":
                add_goal(last_transition_goal, key=True)
                last_transition_goal_added = True
            
            cur_x = transition_end

        if len(goals) + 1 > num_goals:
            terrain.height_field_raw[:, :] = segment_backup
            goals = goals_backup
            key_goal_ids = key_goal_ids_backup
            cur_x = cur_x_backup
            last_transition_goal = last_transition_goal_backup
            last_transition_goal_added = last_transition_goal_added_backup
            break

    final_x = min(cur_x + round(1.0 / terrain.horizontal_scale), terrain.width - round(0.5 / terrain.horizontal_scale))
    add_goal([final_x, mid_y], key=True)

    goals = np.array(goals)
    if len(goals) > num_goals:
        key_ids = np.array(sorted(key_goal_ids))
        if len(key_ids) >= num_goals:
            keep_ids = key_ids[np.round(np.linspace(0, len(key_ids) - 1, num_goals)).astype(int)]
        else:
            remaining = np.array([i for i in range(len(goals)) if i not in key_goal_ids])
            num_remaining = num_goals - len(key_ids)
            if len(remaining) > num_remaining:
                remaining = remaining[np.round(np.linspace(0, len(remaining) - 1, num_remaining)).astype(int)]
            keep_ids = np.sort(np.concatenate([key_ids, remaining]))
        goals = goals[keep_ids]
    elif len(goals) < num_goals:
        goals = np.concatenate([goals, np.repeat(goals[-1][None, :], num_goals - len(goals), axis=0)], axis=0)
    goals = _sanitize_goals_on_safe_cells(terrain, goals, gap_depth)

    terrain.goals = goals * terrain.horizontal_scale

    pad_width = int(pad_width / terrain.horizontal_scale)
    pad_height = int(pad_height / terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = np.maximum(terrain.height_field_raw[:, :pad_width], pad_height)
    terrain.height_field_raw[:, -pad_width:] = np.maximum(terrain.height_field_raw[:, -pad_width:], pad_height)
    terrain.height_field_raw[:pad_width, :] = np.maximum(terrain.height_field_raw[:pad_width, :], pad_height)
    terrain.height_field_raw[-pad_width:, :] = np.maximum(terrain.height_field_raw[-pad_width:, :], pad_height)



# end of new terrain

def demo_terrain(terrain):
    goals = np.zeros((8, 2))
    mid_y = terrain.length // 2
    
    # hurdle
    platform_length = round(2 / terrain.horizontal_scale)
    hurdle_depth = round(np.random.uniform(0.35, 0.4) / terrain.horizontal_scale)
    hurdle_height = round(np.random.uniform(0.3, 0.36) / terrain.vertical_scale)
    hurdle_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[0] = [platform_length + hurdle_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+hurdle_depth, round(mid_y-hurdle_width/2):round(mid_y+hurdle_width/2)] = hurdle_height
    
    # step up
    platform_length += round(np.random.uniform(1.5, 2.5) / terrain.horizontal_scale)
    first_step_depth = round(np.random.uniform(0.45, 0.8) / terrain.horizontal_scale)
    first_step_height = round(np.random.uniform(0.35, 0.45) / terrain.vertical_scale)
    first_step_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[1] = [platform_length+first_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+first_step_depth, round(mid_y-first_step_width/2):round(mid_y+first_step_width/2)] = first_step_height
    
    platform_length += first_step_depth
    second_step_depth = round(np.random.uniform(0.45, 0.8) / terrain.horizontal_scale)
    second_step_height = first_step_height
    second_step_width = first_step_width
    goals[2] = [platform_length+second_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+second_step_depth, round(mid_y-second_step_width/2):round(mid_y+second_step_width/2)] = second_step_height
    
    # gap
    platform_length += second_step_depth
    gap_size = round(np.random.uniform(0.5, 0.8) / terrain.horizontal_scale)
    
    # step down
    platform_length += gap_size
    third_step_depth = round(np.random.uniform(0.25, 0.6) / terrain.horizontal_scale)
    third_step_height = first_step_height
    third_step_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[3] = [platform_length+third_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+third_step_depth, round(mid_y-third_step_width/2):round(mid_y+third_step_width/2)] = third_step_height
    
    platform_length += third_step_depth
    forth_step_depth = round(np.random.uniform(0.25, 0.6) / terrain.horizontal_scale)
    forth_step_height = first_step_height
    forth_step_width = third_step_width
    goals[4] = [platform_length+forth_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+forth_step_depth, round(mid_y-forth_step_width/2):round(mid_y+forth_step_width/2)] = forth_step_height
    
    # parkour
    platform_length += forth_step_depth
    gap_size = round(np.random.uniform(0.1, 0.4) / terrain.horizontal_scale)
    platform_length += gap_size
    
    left_y = mid_y + round(np.random.uniform(0.15, 0.3) / terrain.horizontal_scale)
    right_y = mid_y - round(np.random.uniform(0.15, 0.3) / terrain.horizontal_scale)
    
    slope_height = round(np.random.uniform(0.15, 0.22) / terrain.vertical_scale)
    slope_depth = round(np.random.uniform(0.75, 0.85) / terrain.horizontal_scale)
    slope_width = round(1.0 / terrain.horizontal_scale)
    
    platform_height = slope_height + np.random.randint(0, 0.2 / terrain.vertical_scale)

    goals[5] = [platform_length+slope_depth/2, left_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * 1
    terrain.height_field_raw[platform_length:platform_length+slope_depth, left_y-slope_width//2: left_y+slope_width//2] = heights.astype(int) + platform_height
    
    platform_length += slope_depth + gap_size
    goals[6] = [platform_length+slope_depth/2, right_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * -1
    terrain.height_field_raw[platform_length:platform_length+slope_depth, right_y-slope_width//2: right_y+slope_width//2] = heights.astype(int) + platform_height
    
    platform_length += slope_depth + gap_size + round(0.4 / terrain.horizontal_scale)
    goals[-1] = [platform_length, left_y]
    terrain.goals = goals * terrain.horizontal_scale

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

def half_sloped_terrain(terrain, wall_width=4, start2center=0.7, max_height=1):
    wall_width_int = max(int(wall_width / terrain.horizontal_scale), 1)
    max_height_int = int(max_height / terrain.vertical_scale)
    slope_start = int(start2center / terrain.horizontal_scale + terrain.length // 2)
    terrain_length = terrain.length
    height2width_ratio = max_height_int / wall_width_int
    xs = np.arange(slope_start, terrain_length)
    heights = (height2width_ratio * (xs - slope_start)).clip(max=max_height_int).astype(np.int16)
    terrain.height_field_raw[slope_start:terrain_length, :] = heights[:, None]
    terrain.slope_vector = np.array([wall_width_int*terrain.horizontal_scale, 0., max_height]).astype(np.float32)
    terrain.slope_vector /= np.linalg.norm(terrain.slope_vector)
    # print(terrain.slope_vector, wall_width)
    # import matplotlib.pyplot as plt
    # plt.imsave('test.png', terrain.height_field_raw, cmap='gray')

def half_platform_terrain(terrain, start2center=2, max_height=1):
    max_height_int = int(max_height / terrain.vertical_scale)
    slope_start = int(start2center / terrain.horizontal_scale + terrain.length // 2)
    terrain_length = terrain.length
    terrain.height_field_raw[:, :] = max_height_int
    terrain.height_field_raw[-slope_start:slope_start, -slope_start:slope_start] = 0
    # print(terrain.slope_vector, wall_width)
    # import matplotlib.pyplot as plt
    # plt.imsave('test.png', terrain.height_field_raw, cmap='gray')

def stepping_stones_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1., depth=-1):
    """
    Generate a stepping stones terrain

    Parameters:
        terrain (terrain): the terrain
        stone_size (float): horizontal size of the stepping stones [meters]
        stone_distance (float): distance between stones (i.e size of the holes) [meters]
        max_height (float): maximum height of the stones (positive and negative) [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        depth (float): depth of the holes (default=-10.) [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    def get_rand_dis_int(scale):
        return np.random.randint(int(- scale / terrain.horizontal_scale + 1), int(scale / terrain.horizontal_scale))
    # switch parameters to discrete units
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(-max_height-1, max_height, step=1)

    start_x = 0
    start_y = 0
    terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)
    if terrain.length >= terrain.width:
        while start_y < terrain.length:
            stop_y = min(terrain.length, start_y + stone_size)
            start_x = np.random.randint(0, stone_size)
            # fill first hole
            stop_x = max(0, start_x - stone_distance - get_rand_dis_int(0.2))
            terrain.height_field_raw[0: stop_x, start_y: stop_y] = np.random.choice(height_range)
            # fill row
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_x += stone_size + stone_distance + get_rand_dis_int(0.2)
            start_y += stone_size + stone_distance + get_rand_dis_int(0.2)
    elif terrain.width > terrain.length:
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)
            # fill first hole
            stop_y = max(0, start_y - stone_distance)
            terrain.height_field_raw[start_x: stop_x, 0: stop_y] = np.random.choice(height_range)
            # fill column
            while start_y < terrain.length:
                stop_y = min(terrain.length, start_y + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_y += stone_size + stone_distance
            start_x += stone_size + stone_distance

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain

def convert_heightfield_to_trimesh_delatin(height_field_raw, horizontal_scale, vertical_scale, max_error=0.01):
    mesh = Delatin(np.flip(height_field_raw, axis=1).T, z_scale=vertical_scale, max_error=max_error)
    vertices = np.zeros_like(mesh.vertices)
    vertices[:, :2] = mesh.vertices[:, :2] * horizontal_scale
    vertices[:, 2] = mesh.vertices[:, 2]
    return vertices, mesh.triangles

def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

    return vertices, triangles, move_x != 0

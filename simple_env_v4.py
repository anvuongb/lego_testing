import gymnasium as gym
from gymnasium import spaces
import numpy as np
from lego import Brick, LegoModel
from stud_control import update_occupied_stud_matrx, get_all_possible_placements
import helpers
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, List
import json

DEFAULT_TRANSFORM = helpers.build_translation_matrix(0, -24, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)

BRICK_TYPE_MAP = {
    0: [(2, 2), "3003.dat", 30, False],
    1: [(2, 4), "3001.dat", 50, False],
    2: [(2, 4), "3001.dat", 50, True],
    3: [(2, 3), "3002.dat", 70, False],
    4: [(2, 3), "3002.dat", 70, True],
    5: [(2, 6), "2456.dat", 90, False],
    6: [(2, 6), "2456.dat", 90, True],
    7: [(1, 2), "3004.dat", 110, False],
    8: [(1, 1), "3005.dat", 130, False],
    9: [(1, 3), "3009.dat", 150, False],
}

DEFAULT_COLORS = [14, 24, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]


def _brick_footprint(grid_h, brick_h, brick_w, x, z):
    row_start = grid_h - brick_h - z
    row_end = grid_h - z
    col_start = x
    col_end = x + brick_w
    return row_start, row_end, col_start, col_end


def _check_support(occ_grid_3d, layer, brick_h, brick_w, x, z, threshold):
    if layer == 0:
        return True
    if layer >= occ_grid_3d.shape[0]:
        return False
    H = occ_grid_3d.shape[1]
    rs, re, cs, ce = _brick_footprint(H, brick_h, brick_w, x, z)
    rs = max(0, rs)
    re = min(occ_grid_3d.shape[1], re)
    cs = max(0, cs)
    ce = min(occ_grid_3d.shape[2], ce)
    if rs >= re or cs >= ce:
        return False
    support_region = occ_grid_3d[layer - 1, rs:re, cs:ce]
    if support_region.size == 0:
        return False
    support_ratio = np.sum(support_region) / (brick_h * brick_w)
    return support_ratio >= threshold


def _check_collision(occ_grid_3d, layer, brick_h, brick_w, x, z):
    H = occ_grid_3d.shape[1]
    rs, re, cs, ce = _brick_footprint(H, brick_h, brick_w, x, z)
    rs = max(0, rs)
    re = min(occ_grid_3d.shape[1], re)
    cs = max(0, cs)
    ce = min(occ_grid_3d.shape[2], ce)
    if rs >= re or cs >= ce:
        return True
    return np.any(occ_grid_3d[layer, rs:re, cs:ce] > 0)


def _generate_brick_placements(base_occupancy, brick_h, brick_w, mode="valid", collide_type="brick"):
    brick_stud_mat = np.ones((brick_h, brick_w))
    return get_all_possible_placements(base_occupancy, brick_stud_mat, mode=mode, collide_type=collide_type)


def _target_from_ldr(filepath, grid_height, grid_width, max_levels):
    model = LegoModel(filepath=filepath)
    target = np.zeros((max_levels, grid_height, grid_width), dtype=np.float32)
    unit_height = 24

    if not model.bricks:
        return target

    min_x = min(b.center_x for b in model.bricks)
    min_z = min(b.center_z for b in model.bricks)
    offset_x = max(0, -min_x // 20 * 20) if min_x < 0 else 0
    offset_z = max(0, -min_z // 20 * 20) if min_z < 0 else 0

    for brick in model.bricks:
        layer = int(round(abs(brick.center_y_origin) / unit_height))
        if layer >= max_levels:
            continue
        bs = brick.block_size
        cx = brick.center_x + offset_x
        cz = brick.center_z + offset_z
        half_w = bs[1] * 20 / 2
        half_h = bs[0] * 20 / 2
        x_start = int(round((cx - half_w) / 20))
        x_end = int(round((cx + half_w) / 20))
        z_start_raw = int(round((cz - half_h) / 20))
        z_end_raw = int(round((cz + half_h) / 20))
        z_start = grid_height - z_end_raw
        z_end = grid_height - z_start_raw
        x_start = max(0, x_start)
        x_end = min(grid_width, x_end)
        z_start = max(0, z_start)
        z_end = min(grid_height, z_end)
        if x_start < x_end and z_start < z_end:
            target[layer, z_start:z_end, x_start:x_end] = 1.0
    return target


def _generate_pyramid_target(levels, grid_height, grid_width, max_levels):
    target = np.zeros((max_levels, grid_height, grid_width), dtype=np.float32)
    base_w = levels * 2
    base_h = levels * 2
    idx_w = (grid_width - base_w) // 2
    idx_h = (grid_height - base_h) // 2
    for i in range(levels):
        tmp = np.ones((base_h, base_w))
        tmp[:i, :] = 0
        tmp[base_h - i:, :] = 0
        tmp[:, :i] = 0
        tmp[:, base_w - i:] = 0
        target[i, idx_h:idx_h + base_h, idx_w:idx_w + base_w] = tmp
    return target


@dataclass
class LegoSetConfig:
    grid_height: int = 20
    grid_width: int = 20
    max_levels: int = 10
    brick_type_indices: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8])
    structural_support_ratio: float = 0.5
    rand_levels: bool = True
    min_levels: int = 3
    max_levels_config: int = 10
    reward_weights: dict = field(default_factory=lambda: {
        "shape": 1.0,
        "structural": 0.5,
        "efficiency": 0.1,
        "diversity": 0.2,
        "layer_complete": 5.0,
        "model_complete": 10.0,
        "invalid_placement": -2.0,
    })
    budget_per_type: int = 20
    use_budget: bool = False
    target_path: Optional[str] = None
    free_form: bool = False


def base_action_generation_multiple_bricks(base_stud_mat, brick_indices, mode="valid", collide_type="brick", budget=None):
    actions = []
    for idx in brick_indices:
        if budget is not None and budget[idx] <= 0:
            continue
        brick_stud_mat, rotate = np.ones(BRICK_TYPE_MAP[idx][0]), BRICK_TYPE_MAP[idx][-1]
        if rotate:
            brick_stud_mat = brick_stud_mat.T
        tmp = get_all_possible_placements(base_stud_mat, brick_stud_mat, mode=mode, collide_type=collide_type)
        for a in tmp:
            a.insert(0, idx)
        actions += tmp
    return actions


def mask_fn(env):
    layer_occ = env.occupancy_mat_list[env.current_layer_idx]
    placements_list = base_action_generation_multiple_bricks(
        layer_occ, env.brick_type_indices, mode="valid", collide_type="brick", budget=env.budget_map
    )
    for idx, val in enumerate(placements_list):
        placements_list[idx] = tuple(val)

    placements_control_list = base_action_generation_multiple_bricks(
        env.action_control_stud_mat, env.brick_type_indices, mode="valid", collide_type="brick", budget=env.budget_map
    )
    for idx, val in enumerate(placements_control_list):
        placements_control_list[idx] = tuple(val)
    placements_list = list(set(placements_control_list).intersection(placements_list))

    if env.current_layer_idx > 0:
        placements_partial_list = base_action_generation_multiple_bricks(
            1 - env.occupancy_mat_list[env.current_layer_idx - 1],
            env.brick_type_indices, mode="valid", collide_type="hole", budget=env.budget_map
        )
        for idx, val in enumerate(placements_partial_list):
            placements_partial_list[idx] = tuple(val)
        placements_list = list(set(placements_partial_list).intersection(placements_list))

    # Filter by structural support
    filtered = []
    for p in placements_list:
        brick_idx, x, z = p
        brick_h, brick_w = BRICK_TYPE_MAP[brick_idx][0]
        # Check if rotated
        if BRICK_TYPE_MAP[brick_idx][-1]:
            bh, bw = brick_w, brick_h
        else:
            bh, bw = brick_h, brick_w
        if _check_support(env.occ_grid_3d, env.current_layer_idx, bh, bw, x, z, env.structural_support_ratio):
            filtered.append(p)
    placements_list = filtered

    masked_actions_dict = {i: False for i in range(len(env.all_actions))}

    if env.bricks_per_level[env.current_layer_idx] > 0:
        masked_actions_dict[len(env.all_actions) - 1] = True

    for k, v in env.actions_map.items():
        if tuple(v) in placements_list:
            masked_actions_dict[k] = True

    return np.array([masked_actions_dict[i] for i in range(len(env.all_actions))])


class SimpleLegoEnv(gym.Env):
    def __init__(self, config: Optional[LegoSetConfig] = None):
        super().__init__()
        self.config = config or LegoSetConfig()
        self._load_config()

    def _load_config(self):
        c = self.config
        self.grid_height = c.grid_height
        self.grid_width = c.grid_width
        self.max_levels = c.max_levels
        self.brick_type_indices = c.brick_type_indices
        self.structural_support_ratio = c.structural_support_ratio
        self.rand_levels = c.rand_levels
        self.min_levels = c.min_levels
        self.max_levels_config = c.max_levels_config
        self.reward_weights = c.reward_weights
        self.use_budget = c.use_budget
        self.budget_per_type = c.budget_per_type
        self.target_path = c.target_path
        self.free_form = c.free_form

        self.budget_map = None
        self.pyramid_levels = self.min_levels
        self._setup_target_and_actions()

    def _setup_target_and_actions(self):
        self.pyramid_base_width = self.pyramid_levels * 2
        self.pyramid_base_height = self.pyramid_levels * 2
        self.model_base_width = self.max_levels_config * 2
        self.model_base_height = self.max_levels_config * 2

        self.action_control_stud_mat = np.zeros((self.model_base_height, self.model_base_width))
        idx_w = (self.model_base_width - self.pyramid_base_width) // 2
        idx_h = (self.model_base_height - self.pyramid_base_height) // 2
        if idx_h > 0:
            self.action_control_stud_mat[:idx_h, :] = 1
            self.action_control_stud_mat[self.model_base_height - idx_h:, :] = 1
        if idx_w > 0:
            self.action_control_stud_mat[:, :idx_w] = 1
            self.action_control_stud_mat[:, self.model_base_width - idx_w:] = 1

        if self.target_path:
            self.target_voxel_grid = _target_from_ldr(
                self.target_path, self.grid_height, self.grid_width, self.max_levels
            )
            actual_levels = 0
            for l in range(self.max_levels):
                if np.any(self.target_voxel_grid[l] > 0):
                    actual_levels = l + 1
            self.pyramid_levels = max(1, actual_levels)
        elif self.free_form:
            self.target_voxel_grid = np.zeros((self.max_levels, self.grid_height, self.grid_width), dtype=np.float32)
        else:
            self.target_voxel_grid = _generate_pyramid_target(
                self.pyramid_levels, self.grid_height, self.grid_width, self.max_levels
            )

        self.target_stud_mat_list = [self.target_voxel_grid[l].copy() for l in range(self.max_levels)]

        self.base_brick_stud_mat_list = []
        for idx in self.brick_type_indices:
            brick_stud_mat, rotate = np.ones(BRICK_TYPE_MAP[idx][0]), BRICK_TYPE_MAP[idx][-1]
            if rotate:
                brick_stud_mat = brick_stud_mat.T
            self.base_brick_stud_mat_list.append(brick_stud_mat)

        self.all_actions = base_action_generation_multiple_bricks(
            np.zeros((self.grid_height, self.grid_width)), self.brick_type_indices, "valid"
        )
        self.all_actions += ["moveup"]
        self.n_discrete_actions = len(self.all_actions)
        self.actions_map = dict(zip(range(self.n_discrete_actions), self.all_actions))

        self.action_space = spaces.Discrete(self.n_discrete_actions)
        self.observation_space = spaces.Box(
            low=-1, high=10,
            shape=(2 * self.max_levels * self.grid_height * self.grid_width + 1 + len(self.brick_type_indices),),
            dtype=np.float32
        )

        self.bricks_list = []
        self.bricks_per_level = [0 for _ in range(self.max_levels)]
        self.occupancy_mat_list = [np.zeros((self.grid_height, self.grid_width)) for _ in range(self.max_levels)]
        self.occ_grid_3d = np.zeros((self.max_levels, self.grid_height, self.grid_width), dtype=np.float32)
        self.current_layer_idx = 0
        self.brick_usage_count = {idx: 0 for idx in self.brick_type_indices}
        self.total_studs_placed = 0

    def _get_obs(self):
        obs = []
        for mat in self.occupancy_mat_list:
            obs.extend(mat.ravel().tolist())
        for mat in self.target_stud_mat_list:
            obs.extend(mat.ravel().tolist())
        obs.append(self.pyramid_levels)
        for idx in self.brick_type_indices:
            if self.budget_map:
                obs.append(self.budget_map[idx])
            else:
                obs.append(5.0)
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        action_decode = self.actions_map[action]
        terminated = False
        truncated = False
        info = {"valid": True}

        if action_decode == "moveup":
            if self.current_layer_idx < self.pyramid_levels - 1:
                target_sum = np.sum(self.target_stud_mat_list[self.current_layer_idx])
                if target_sum > 0:
                    fill_percent = np.sum(self.occupancy_mat_list[self.current_layer_idx]) / target_sum
                else:
                    fill_percent = 1.0
                if fill_percent < 0.5:
                    reward = self.reward_weights.get("moveup_early", -1.0)
                elif fill_percent >= 0.8:
                    reward = 1.0
                else:
                    reward = 0.0
                self.current_layer_idx += 1
            else:
                terminated = True
                model_complete = True
                for l in range(self.pyramid_levels):
                    target_sum = np.sum(self.target_stud_mat_list[l])
                    if target_sum > 0:
                        if not np.all(np.equal(self.occupancy_mat_list[l], self.target_stud_mat_list[l])):
                            model_complete = False
                            break
                if model_complete:
                    reward = self.reward_weights.get("model_complete", 10.0)
                else:
                    reward = 0.0
        else:
            brick_idx, xunit, zunit = action_decode
            brick_h, brick_w = BRICK_TYPE_MAP[brick_idx][0]
            rotated = BRICK_TYPE_MAP[brick_idx][-1]
            if rotated:
                bh, bw = brick_w, brick_h
            else:
                bh, bw = brick_h, brick_w

            # 3D collision check
            if _check_collision(self.occ_grid_3d, self.current_layer_idx, bh, bw, xunit, zunit):
                reward = self.reward_weights.get("invalid_placement", -2.0)
                info["valid"] = False
                return self._get_obs(), reward, terminated, truncated, info

            # Structural support check
            if not _check_support(self.occ_grid_3d, self.current_layer_idx, bh, bw, xunit, zunit, self.structural_support_ratio):
                reward = self.reward_weights.get("invalid_placement", -2.0)
                info["valid"] = False
                return self._get_obs(), reward, terminated, truncated, info

            # Budget check
            if self.use_budget and self.budget_map is not None:
                if self.budget_map[brick_idx] <= 0:
                    reward = self.reward_weights.get("invalid_placement", -2.0)
                    info["valid"] = False
                    return self._get_obs(), reward, terminated, truncated, info
                self.budget_map[brick_idx] -= 1

            # Place brick
            brick_color = self.target_stud_mat_list[self.current_layer_idx][0, 0] * 10 + 30 or 30
            _, brick_type, base_color, _ = BRICK_TYPE_MAP[brick_idx]
            brick = Brick(0, -24, 0, DEFAULT_TRANSFORM, brick_type, int(base_color), "#8A12A8")
            if rotated:
                brick.rotate_yaxis_90deg_align_origin()
            brick.translate_corner_to_brick_relative(
                Brick(0, -24, 0, DEFAULT_TRANSFORM, f"base{self.grid_width}x{self.grid_height}", 324, "#8A12A8")
            )
            brick_mat = np.ones(brick.stud_matrix.shape)
            brick.unit_translate(xunit, -self.current_layer_idx, zunit)
            self.bricks_list.append(brick)
            self.bricks_per_level[self.current_layer_idx] += 1
            self.brick_usage_count[brick_idx] = self.brick_usage_count.get(brick_idx, 0) + 1
            self.total_studs_placed += bh * bw

            # Update occupancy
            self.occupancy_mat_list[self.current_layer_idx] = update_occupied_stud_matrx(
                self.occupancy_mat_list[self.current_layer_idx], brick_mat, xunit, zunit
            )
            H = self.grid_height
            rs, re, cs, ce = _brick_footprint(H, bh, bw, xunit, zunit)
            rs = max(0, rs)
            re = min(self.grid_height, re)
            cs = max(0, cs)
            ce = min(self.grid_width, ce)
            if rs < re and cs < ce:
                self.occ_grid_3d[self.current_layer_idx, rs:re, cs:ce] = 1.0

            # --- Multi-objective reward ---
            # 1. Shape reward
            mat = np.zeros((self.grid_height, self.grid_width))
            mat = update_occupied_stud_matrx(mat, brick_mat, xunit, zunit)
            target = self.target_stud_mat_list[self.current_layer_idx]
            if self.free_form or np.sum(target) == 0:
                shape_reward = 0.0
            else:
                overlap = np.sum(np.multiply(mat, target))
                expected = bh * bw
                if overlap == expected:
                    shape_reward = self.reward_weights.get("shape", 1.0)
                elif overlap > 0:
                    shape_reward = self.reward_weights.get("shape", 1.0) * (overlap / expected)
                else:
                    shape_reward = self.reward_weights.get("shape", 1.0) * -1.0

            # 2. Structural reward
            if _check_support(self.occ_grid_3d, self.current_layer_idx, bh, bw, xunit, zunit, 0.75):
                structural_reward = self.reward_weights.get("structural", 0.5)
            elif _check_support(self.occ_grid_3d, self.current_layer_idx, bh, bw, xunit, zunit, self.structural_support_ratio):
                structural_reward = self.reward_weights.get("structural", 0.5) * 0.5
            else:
                structural_reward = -self.reward_weights.get("structural", 0.5)

            # 3. Efficiency reward (encourage using larger bricks)
            area = bh * bw
            efficiency_reward = self.reward_weights.get("efficiency", 0.1) * (area / 4.0)

            # 4. Diversity reward
            total_actions = sum(self.brick_usage_count.values())
            if total_actions > 1:
                usage_ratio = self.brick_usage_count[brick_idx] / total_actions
                if usage_ratio < 0.5 / len(self.brick_type_indices):
                    diversity_reward = self.reward_weights.get("diversity", 0.2)
                elif usage_ratio > 3.0 / len(self.brick_type_indices):
                    diversity_reward = -self.reward_weights.get("diversity", 0.2)
                else:
                    diversity_reward = 0.0
            else:
                diversity_reward = 0.0

            reward = shape_reward + structural_reward + efficiency_reward + diversity_reward

            # 5. Layer complete bonus
            if not self.free_form:
                target_sum = np.sum(target)
                if target_sum > 0 and np.all(np.equal(self.occupancy_mat_list[self.current_layer_idx], target)):
                    reward += self.reward_weights.get("layer_complete", 5.0)

        observation = self._get_obs()

        if terminated:
            if len(self.bricks_list) > 1:
                model = LegoModel(brick=self.bricks_list[0])
                for brick in self.bricks_list[1:]:
                    model.add_brick(brick)
                model.generate_ldr_file("test_v4.ldr")

        return observation, reward, terminated, truncated, info

    def reset(self, seed=42, options=None):
        if self.rand_levels and not self.target_path:
            self.pyramid_levels = np.random.randint(self.min_levels, self.max_levels_config + 1)

        if self.use_budget:
            self.budget_map = {idx: self.budget_per_type for idx in self.brick_type_indices}
        else:
            self.budget_map = None

        self._setup_target_and_actions()

        self.bricks_list = []
        self.bricks_per_level = [0 for _ in range(self.max_levels)]
        self.occupancy_mat_list = [np.zeros((self.grid_height, self.grid_width)) for _ in range(self.max_levels)]
        self.occ_grid_3d = np.zeros((self.max_levels, self.grid_height, self.grid_width), dtype=np.float32)
        self.current_layer_idx = 0
        self.brick_usage_count = {idx: 0 for idx in self.brick_type_indices}
        self.total_studs_placed = 0

        info = {}
        return self._get_obs(), info

    def render(self, mode='human'):
        pass

    def close(self):
        ...

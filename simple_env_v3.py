import gymnasium as gym
from gymnasium import spaces
import subprocess
import numpy as np
from lego import Brick, LegoModel
from stud_control import update_occupied_stud_matrx, get_all_possible_placements
import helpers
from collections import deque
from copy import deepcopy

# default transform for lego brick
DEFAULT_TRANSFORM = helpers.build_translation_matrix(0, -24, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)

# brick index: (height, width), ldr_code, color, perform 90deg rotate
BRICK_TYPE_MAP = {
    0: [(2,2), "3003.dat", 30, False],
    1: [(2,4), "3001.dat", 50, False],
    2: [(2,4), "3001.dat", 50, True],
    3: [(2,3), "3002.dat", 70, False],
    4: [(2,3), "3002.dat", 70, True],
    5: [(2,6), "2456.dat", 90, False],
    6: [(2,6), "2456.dat", 90, True]

}

def base_action_generation(base_stud_mat, brick_stud_mat, mode="valid"):
    actions = get_all_possible_placements(base_stud_mat, brick_stud_mat, mode=mode)
    return actions

def base_action_generation_multiple_bricks(base_stud_mat, brick_indices, mode="valid"):
    # similar to base_action_generation but input is a list of different brick types
    actions = []

    for idx in brick_indices:
        brick_stud_mat, rotate = np.ones(BRICK_TYPE_MAP[idx][0]), BRICK_TYPE_MAP[idx][-1]
        if rotate:
            brick_stud_mat = brick_stud_mat.transpose()
        tmp = get_all_possible_placements(base_stud_mat, brick_stud_mat, mode=mode)
        for a in tmp:
            a.insert(0, idx)
        actions += tmp

    return actions


def mask_fn(env: gym.Env) -> np.ndarray:
    placements_list = get_all_possible_placements(env.occupancy_mat_list[env.current_layer_idx], env.brick_base_indices, mode="valid", collide_type="brick")
    # stud_control_mat
    placements_control_list = get_all_possible_placements(env.action_control_stud_mat, env.brick_base_indices, mode="valid", collide_type="brick")
    placements_list = list(set(placements_control_list).intersection(placements_list))

    if env.current_layer_idx > 0:
        # allow placement over holes if possible
        placements_partial_list = get_all_possible_placements(1 - env.occupancy_mat_list[env.current_layer_idx-1], env.brick_base_indices, mode="valid", collide_type="hole")
        placements_list = list(set(placements_partial_list).intersection(placements_list))
    masked_actions_dict = {}

    for i in range(len(env.all_actions)):
        masked_actions_dict[i] = False

    # check if moveup allowed
    if env.bricks_per_level[env.current_layer_idx] > 0:
        masked_actions_dict[len(env.all_actions)-1] = True

    # check if other actions allowed:
    for k, v in env.actions_map.items():
        if v in placements_list:
            masked_actions_dict[k] = True

    masked_actions_list = [masked_actions_dict[i] for i in range(len(env.all_actions))]
    return np.array(masked_actions_list)
    
class SimpleLegoEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, brick_base_indices=[0, 1, 2, 3, 4, 5, 6], pyramid_levels=4, min_pyramid_level=3, max_pyramid_level=10, rand_levels=False):
        super(SimpleLegoEnv, self).__init__()
        self.rand_levels = rand_levels
        self.pyramid_levels = pyramid_levels
        self.min_pyramid_level = min_pyramid_level
        self.max_pyramid_level = max_pyramid_level
        self.pyramid_base_width = self.pyramid_levels*2
        self.pyramid_base_height = self.pyramid_levels*2
        self.model_base_width = self.max_pyramid_level*2
        self.model_base_height = self.max_pyramid_level*2

        # masked actions for base smaller than max bas
        self.action_control_stud_mat = np.zeros((self.model_base_height, self.model_base_width))
        idx_w = int((self.model_base_width-self.pyramid_base_width)/2)
        idx_h = int((self.model_base_height-self.pyramid_base_height)/2)

        self.action_control_stud_mat[:idx_h,:] = 1
        self.action_control_stud_mat[self.model_base_height-idx_h:,:] = 1
        self.action_control_stud_mat[:,:idx_w] = 1
        self.action_control_stud_mat[:,self.model_base_width-idx_w:] = 1

        # generate target stud mat for pyramid
        self.target_stud_mat_list = []
        for i in range(self.pyramid_levels):
            # generate base
            tmp = np.ones((self.pyramid_base_height, self.pyramid_base_width))
            # fill zero based on current level
            tmp[:i,:] = 0
            tmp[self.pyramid_base_height-i:,:] = 0
            tmp[:,:i] = 0
            tmp[:,self.pyramid_base_width-i:] = 0
            tmp_2 = np.zeros((self.model_base_height, self.model_base_width))
            tmp_2[idx_h:self.model_base_height-idx_h, idx_w:self.model_base_width-idx_w] = tmp
            self.target_stud_mat_list.append(tmp_2)
        # pad till reach max height
        for i in range(self.pyramid_levels, self.max_pyramid_level):
            self.target_stud_mat_list.append(np.zeros((self.model_base_height, self.model_base_width)))

        # all possible actions
        self.brick_base_indices = brick_base_indices
        self.base_brick_stud_mat_list = []
        for idx in self.brick_base_indices:
            brick_stud_mat, rotate = np.ones(BRICK_TYPE_MAP[idx][0]), BRICK_TYPE_MAP[idx][-1]
            if rotate:
                brick_stud_mat = brick_stud_mat.transpose()
            self.base_brick_stud_mat_list.append(brick_stud_mat) 
        self.all_actions = base_action_generation_multiple_bricks(np.zeros((self.model_base_height, self.model_base_width)), self.brick_base_indices, "valid")
        self.all_actions += ["moveup"]
        self.n_discrete_actions = len(self.all_actions)
        self.actions_map = dict(zip(range(self.n_discrete_actions), self.all_actions))

        self.action_space = spaces.Discrete(self.n_discrete_actions)
        # N_CHANNELS -> pyramid height
        # HEIGHT = WIDTH = base size
        self.observation_space = spaces.Box(low=-1, high=10,
                                            shape=(2*self.max_pyramid_level*self.model_base_height*self.model_base_width + 1,), dtype=np.int16) # include targe stud mat and pyramid levels
        self.bricks_list = []
        self.bricks_per_level = [0 for _ in range(self.pyramid_levels)]

        self.occupancy_mat_list = [np.zeros((self.model_base_height, self.model_base_height)) for _ in range(self.max_pyramid_level)]
        self.current_layer_idx = 0
    
    def step(self, action):
        # assumming masked -> action is always valid
        action_decode = self.actions_map[action]
        terminated = False
        truncated = False
        info = {}
        if action_decode == "moveup":
            # check if already on top
            if self.current_layer_idx < self.pyramid_levels-1:
                # move up to next layer
                fill_percent = np.sum(self.occupancy_mat_list[self.current_layer_idx])/np.sum(self.target_stud_mat_list[self.current_layer_idx])
                if fill_percent < 0.5:
                    # move up too soon, penalize
                    reward = -1
                elif fill_percent > 0.8:
                    # kind good idk
                    reward = 1
                else:
                    reward = 0
                self.current_layer_idx += 1
            else:
                terminated = True
                reward = 0
        else:
            # apply action and update occupancy matrix
            brick_idx, xunit, zunit = action_decode
            _, brick_type, brick_color, brick_rotate = BRICK_TYPE_MAP[brick_idx]

            # create brick and save bricks ldr for visualizing
            brick = Brick(0,-24,0, DEFAULT_TRANSFORM, brick_type, brick_color + self.current_layer_idx, "#8A12A8")
            if brick_rotate:
                brick.rotate_yaxis_90deg_align_origin()
            brick_mat = np.ones(brick.stud_matrix.shape)
            brick.unit_translate(xunit, -self.current_layer_idx, zunit)
            self.bricks_list.append(brick)
            self.bricks_per_level[self.current_layer_idx] += 1

            # update occupancy matrix
            self.occupancy_mat_list[self.current_layer_idx] = update_occupied_stud_matrx(self.occupancy_mat_list[self.current_layer_idx], 
                                                                                        brick_mat, xunit, zunit)
            
            # mat to calculate if brick is with desired region
            mat = np.zeros((self.model_base_height, self.model_base_width))
            mat = update_occupied_stud_matrx(mat, brick_mat, xunit, zunit)

            # calculate reward
            if np.sum(np.multiply(mat, self.target_stud_mat_list[self.current_layer_idx])) == brick_mat.shape[0]*brick_mat.shape[1]:
                # brick fits with desired region
                reward = 1
                # if entire layer is filled, big reward
                if np.all(np.equal(self.occupancy_mat_list[self.current_layer_idx], self.target_stud_mat_list[self.current_layer_idx])):
                    reward += 10
                    # if entire model is filled, even bigger reward, and terminate
                    if self.current_layer_idx == self.pyramid_levels - 1:
                        compared = [np.equal(self.occupancy_mat_list[idx], self.target_stud_mat_list[idx]) for idx in range(self.pyramid_levels)]
                        if np.all(compared):
                            reward += 10
                            terminated = True
            else: 
                reward = -1

        observation = []
        for mat in self.occupancy_mat_list:
            observation += list(mat.ravel())
        for mat in self.target_stud_mat_list:
            observation += list(mat.ravel())
        observation.append(self.pyramid_levels)
        observation = np.array(observation, dtype=np.int16)

        if terminated:
            if len(self.bricks_list) > 1:
                model = LegoModel(brick=self.bricks_list[0])
                for brick in self.bricks_list[1:]:
                    model.add_brick(brick)
                if not self.rand_levels:
                    model.generate_ldr_file("test_v2_new_fixed_{}_levels.ldr".format(self.pyramid_levels))
                else:
                    model.generate_ldr_file("test_v2_new_{}_levels.ldr".format(self.pyramid_levels))

        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=42, options=None):
        # np.random.seed(seed)
        if self.rand_levels:
            self.pyramid_levels = np.random.randint(self.min_pyramid_level, self.max_pyramid_level+1)
        # don't change these vars
        # self.pyramid_levels = pyramid_levels
        # self.min_pyramid_level = min_pyramid_level
        # self.max_pyramid_level = max_pyramid_level
        self.pyramid_base_width = self.pyramid_levels*2
        self.pyramid_base_height = self.pyramid_levels*2
        self.model_base_width = self.max_pyramid_level*2
        self.model_base_height = self.max_pyramid_level*2

        # masked actions for base smaller than max bas
        self.action_control_stud_mat = np.zeros((self.model_base_height, self.model_base_width))
        idx_w = int((self.model_base_width-self.pyramid_base_width)/2)
        idx_h = int((self.model_base_height-self.pyramid_base_height)/2)

        self.action_control_stud_mat[:idx_h,:] = 1
        self.action_control_stud_mat[self.model_base_height-idx_h:,:] = 1
        self.action_control_stud_mat[:,:idx_w] = 1
        self.action_control_stud_mat[:,self.model_base_width-idx_w:] = 1

        # generate target stud mat for pyramid
        self.target_stud_mat_list = []
        for i in range(self.pyramid_levels):
            # generate base
            tmp = np.ones((self.pyramid_base_height, self.pyramid_base_width))
            # fill zero based on current level
            tmp[:i,:] = 0
            tmp[self.pyramid_base_height-i:,:] = 0
            tmp[:,:i] = 0
            tmp[:,self.pyramid_base_width-i:] = 0
            tmp_2 = np.zeros((self.model_base_height, self.model_base_width))
            tmp_2[idx_h:self.model_base_height-idx_h, idx_w:self.model_base_width-idx_w] = tmp
            self.target_stud_mat_list.append(tmp_2)
        # pad till reach max height
        for i in range(self.pyramid_levels, self.max_pyramid_level):
            self.target_stud_mat_list.append(np.zeros((self.model_base_height, self.model_base_width)))

        # all possible actions

        # all possible actions
        self.base_brick_stud_mat_list = []
        for idx in self.brick_base_indices:
            brick_stud_mat, rotate = np.ones(BRICK_TYPE_MAP[idx][0]), BRICK_TYPE_MAP[idx][-1]
            if rotate:
                brick_stud_mat = brick_stud_mat.transpose()
            self.base_brick_stud_mat_list.append(brick_stud_mat) 
        self.all_actions = base_action_generation_multiple_bricks(np.zeros((self.model_base_height, self.model_base_width)), self.brick_base_indices, "valid")
        self.all_actions += ["moveup"]
        self.n_discrete_actions = len(self.all_actions)
        self.actions_map = dict(zip(range(self.n_discrete_actions), self.all_actions))

        self.action_space = spaces.Discrete(self.n_discrete_actions)
        # N_CHANNELS -> pyramid height
        # HEIGHT = WIDTH = base size
        self.observation_space = spaces.Box(low=-1, high=10,
                                            shape=(2*self.max_pyramid_level*self.model_base_height*self.model_base_width + 1,), dtype=np.int16) # include targe stud mat and pyramid levels
        self.bricks_list = []
        self.bricks_per_level = [0 for _ in range(self.pyramid_levels)]

        self.occupancy_mat_list = [np.zeros((self.model_base_height, self.model_base_height)) for _ in range(self.max_pyramid_level)]
        self.current_layer_idx = 0
        observation = []
        for mat in self.occupancy_mat_list:
            observation += list(mat.ravel())
        for mat in self.target_stud_mat_list:
            observation += list(mat.ravel())
        observation.append(self.pyramid_levels)
        observation = np.array(observation, dtype=np.int16)
        info = {}
        return observation, info  # reward, done, info can't be included
    
    def render(self, mode='human'):
        pass

    def close (self):
        ...



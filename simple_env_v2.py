import gymnasium as gym
from gymnasium import spaces
import subprocess
import numpy as np
from lego import Brick, LegoModel
from stud_control import update_occupied_stud_matrx, get_all_possible_placements
import helpers
from collections import deque
from copy import deepcopy

# # valid mode placement
# ALL_ACTIONS = [(0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6), 
#   (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5), 
#   (0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), 
#   (0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3), 
#   (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), 
#   (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), 
#   (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
#      "moveup"] # move to next layer
# full mode placement
ALL_ACTIONS = [(-1, 7), (0, 7), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7), (-1, 6), (0, 6), (1, 6), (2, 6), 
               (3, 6), (4, 6), (5, 6), (6, 6), (7, 6), (-1, 5), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5), 
               (7, 5), (-1, 4), (0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4), (-1, 3), (0, 3), (1, 3), 
               (2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (-1, 2), (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), 
               (6, 2), (7, 2), (-1, 1), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (-1, 0), (0, 0), 
               (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (-1, -1), (0, -1), (1, -1), (2, -1), (3, -1), 
               (4, -1), (5, -1), (6, -1), (7, -1),
               "moveup"] # move to next layer
N_DISCRETE_ACTIONS = len(ALL_ACTIONS)

ACTIONS_MAP = dict(zip(range(N_DISCRETE_ACTIONS), ALL_ACTIONS))
PREV_ACTIONS_QUEUE_LEN = 8

N_CHANNELS = 10
HEIGHT = N_CHANNELS*2
WIDTH = N_CHANNELS*2

BASE_BRICK_SHAPE = (2,2)

# generat target stud mat for pyramid
target_stud_mat_list = []
for i in range(N_CHANNELS):
    # generate base
    tmp = np.ones((HEIGHT, WIDTH))
    # fill zero based on current level
    tmp[:i,:] = 0
    tmp[HEIGHT-i:,:] = 0
    tmp[:,:i] = 0
    tmp[:,WIDTH-i:] = 0
    target_stud_mat_list.append(tmp)

# default transform for lego brick
DEFAULT_TRANSFORM = helpers.build_translation_matrix(0, -24, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)
BRICK_TYPE = "3003.dat" # 2x2 brick code
BRICK_COLOR = 60 # base layer color
    
class SimpleLegoEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        super(SimpleLegoEnv, self).__init__()
        self.base_brick_stud_mat = np.ones(BASE_BRICK_SHAPE)

        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # N_CHANNELS -> pyramid height
        # HEIGHT = WIDTH = base size
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(N_CHANNELS*HEIGHT*WIDTH,), dtype=np.int16)
        self.bricks_list = []
        self.bricks_per_level = [0 for _ in range(N_CHANNELS)]

        self.occupancy_mat_list = [np.zeros((HEIGHT, WIDTH)) for _ in range(N_CHANNELS)]
        self.current_layer_idx = 0
    
    def step(self, action):
        # assumming masked -> action is always valid
        action_decode = ACTIONS_MAP[action]
        terminated = False
        truncated = False
        info = {}
        if action_decode == "moveup":
            # check if already on top
            if self.current_layer_idx < N_CHANNELS-1:
                # move up to next layer
                fill_percent = np.sum(self.occupancy_mat_list[self.current_layer_idx])/np.sum(target_stud_mat_list[self.current_layer_idx])
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
            xunit, zunit = action_decode

            # create brick and save bricks ldr for visualizing
            brick = Brick(0,-24,0, DEFAULT_TRANSFORM, BRICK_TYPE, BRICK_COLOR + self.current_layer_idx, "#8A12A8")
            brick.unit_translate(xunit, -self.current_layer_idx, zunit)
            self.bricks_list.append(brick)
            self.bricks_per_level[self.current_layer_idx] += 1

            # update occupancy matrix
            self.occupancy_mat_list[self.current_layer_idx] = update_occupied_stud_matrx(self.occupancy_mat_list[self.current_layer_idx], 
                                                                                        self.base_brick_stud_mat, xunit, zunit)
            
            # mat to calculate if brick is with desired region
            mat = np.zeros((HEIGHT, WIDTH))
            mat = update_occupied_stud_matrx(mat, np.ones(BASE_BRICK_SHAPE), xunit, zunit)
            
            # calculate reward
            if np.sum(np.multiply(mat, target_stud_mat_list[self.current_layer_idx])) == BASE_BRICK_SHAPE[0]*BASE_BRICK_SHAPE[1]:
                # brick fits with desired region
                reward = 1
            else: 
                reward = -1
        observation = []
        for mat in self.occupancy_mat_list:
            observation += list(mat.ravel())
        observation = np.array(observation, dtype=np.int16)

        if terminated:
            if len(self.bricks_list) > 1:
                model = LegoModel(brick=self.bricks_list[0])
                for brick in self.bricks_list[1:]:
                    model.add_brick(brick)
                model.generate_ldr_file("test_v2_{}_levels.ldr".format(N_CHANNELS))

        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=42, options=None):
        # HEIGHT = WIDTH = base size
        self.bricks_list = []
        self.bricks_per_level = [0 for _ in range(N_CHANNELS)]

        self.occupancy_mat_list = [np.zeros((HEIGHT, WIDTH)) for _ in range(N_CHANNELS)]
        self.current_layer_idx = 0 
        observation = []
        for mat in self.occupancy_mat_list:
            observation += list(mat.ravel())
        observation = np.array(observation, dtype=np.int16)
        info = {}
        return observation, info  # reward, done, info can't be included
    
    def render(self, mode='human'):
        pass

    def close (self):
        ...



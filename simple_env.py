import gymnasium as gym
from gymnasium import spaces
import subprocess
import numpy as np
from lego import Brick, LegoModel
from stud_control import update_occupied_stud_matrx, get_all_possible_placements
import helpers
from collections import deque

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
# print(ACTIONS_MAP)
PREV_ACTIONS_QUEUE_LEN = 32



HEIGHT = 8
WIDTH = 8
N_CHANNELS = 4

BASE_BRICK_SHAPE = (2,2)

target_stud_mat_list = [
    np.array([[1., 1., 1., 1., 1., 1., 1., 1.],
              [1., 1., 1., 1., 1., 1., 1., 1.],
              [1., 1., 1., 1., 1., 1., 1., 1.],
              [1., 1., 1., 1., 1., 1., 1., 1.],
              [1., 1., 1., 1., 1., 1., 1., 1.],
              [1., 1., 1., 1., 1., 1., 1., 1.],
              [1., 1., 1., 1., 1., 1., 1., 1.],
              [1., 1., 1., 1., 1., 1., 1., 1.]]),
    np.array([[-1., -1., -1., -1., -1., -1., -1., -1.],
              [-1., 1., 1., 1., 1., 1., 1., -1.],
              [-1., 1., 1., 1., 1., 1., 1., -1.],
              [-1., 1., 1., 1., 1., 1., 1., -1.],
              [-1., 1., 1., 1., 1., 1., 1., -1.],
              [-1., 1., 1., 1., 1., 1., 1., -1.],
              [-1., 1., 1., 1., 1., 1., 1., -1.],
              [-1., -1., -1., -1., -1., -1., -1., -1.]]),
    np.array([[-1., -1., -1., -1., -1., -1., -1., -1.],
              [-1., -1., -1., -1., -1., -1., -1., 0],
              [-1., -1., 1., 1., 1., 1., -1., -1.],
              [-1., -1., 1., 1., 1., 1., -1., -1.],
              [-1., -1., 1., 1., 1., 1., -1., -1.],
              [-1., -1., 1., 1., 1., 1., -1., -1.],
              [-1., -1., -1., -1., -1., -1., -1., -1.],
              [-1., -1., -1., -1., -1., -1., -1., -1.]]),
    np.array([[-1., -1., -1., -1., -1., -1., -1., -1.],
              [-1., -1., -1., -1., -1., -1., -1., 0],
              [-1., -1., -1., -1., -1., -1., -1., -1.],
              [-1., -1., -1., 1., 1., -1., -1., -1.],
              [-1., -1., -1., 1., 1., -1., -1., -1.],
              [-1., -1., -1., -1., -1., -1., -1., -1.],
              [-1., -1., -1., -1., -1., -1., -1., -1.],
              [-1., -1., -1., -1., -1., -1., -1., -1.]])
]
# default transform for lego brick
DEFAULT_TRANSFORM = helpers.build_translation_matrix(0, -24, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)
BRICK_TYPE = "3003.dat" # 2x2 brick code
BRICK_COLOR = 60 # base layer color

def calculate_total_value(current_stud_mat_list, target_stud_mat_list):
    value = 0
    for current, target in zip(current_stud_mat_list, target_stud_mat_list):
        value += np.sum(np.multiply(current, target)) 
    return int(value)

def generate_image():
    subprocess.run('''QT_DEBUG_PLUGINS=1 /usr/bin/leocad -i test.png -w 400 -h 400 --camera-angles 30 30 test.ldr''', shell=True)
    
class SimpleLegoEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        super(SimpleLegoEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # N_CHANNELS -> pyramid height
        # HEIGHT = WIDTH = base size
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(N_CHANNELS*HEIGHT*WIDTH,), dtype=np.int16)
        self.current_layer = 0
        self.stud_mat_list = [np.zeros((HEIGHT, WIDTH)) for i in range(N_CHANNELS) ]
        self.current_stud_mat = self.stud_mat_list[self.current_layer]

        self.stud_mat_layer_list = [np.zeros((HEIGHT, WIDTH)) for i in range(N_CHANNELS) ]
        self.current_stud_mat_layer = self.stud_mat_layer_list[self.current_layer]

        self.base_brick_stud_mat = np.ones(BASE_BRICK_SHAPE)
        self.bricks_list = []
        self.bricks_per_level = [0 for _ in range(HEIGHT)]
        self.prev_actions = deque([-1. for _ in range(PREV_ACTIONS_QUEUE_LEN)], maxlen=PREV_ACTIONS_QUEUE_LEN) # save last 32 actions
    
    def step(self, action):
        last_value = calculate_total_value(self.stud_mat_list, target_stud_mat_list)
        self.prev_actions.append(action)
        info = {}
        truncated = False
        action_decode = ACTIONS_MAP[action]
        terminated = False
        if action_decode == "moveup":
            if self.current_layer < N_CHANNELS-1:
                # make sure current layer has at least one brick before move up
                if self.bricks_per_level[self.current_layer] > 0:
                    next_stud_mat_layer = 1 - self.current_stud_mat
                    placements_list = get_all_possible_placements(next_stud_mat_layer.copy(), self.base_brick_stud_mat, mode="full")
                    # only calculate reward for move up if it's actually possible to place next brick
                    if len(placements_list) > 0:
                        perc = np.sum(self.stud_mat_list[self.current_layer])/np.sum(target_stud_mat_list[self.current_layer])
                        if perc < -1.6:
                            reward = -5 # penalize for jumping to next layer to soon
                        elif perc < -1.8:
                            reward = 10 # good reward when knowing when to move up
                        else:
                            reward = 20
                        # rewards when moveup = filled holes of current layer
                        # reward = np.sum(np.multiply(self.stud_mat_list[self.current_layer], target_stud_mat_list[self.current_layer]))
                        self.current_layer += 1
                        # important fix, next layer depends on prev layer
                        next_stud_mat = np.zeros((HEIGHT, WIDTH))
                        next_stud_mat_layer = 1 - self.current_stud_mat
                    else:
                        # moving up to soon, no next possible move, stay on current level
                        reward = -10
                        next_stud_mat = self.current_stud_mat
                        next_stud_mat_layer = self.current_stud_mat_layer
                        # terminated = True
                else:
                    # no brick at current layer, i.e next brick will be floating, big violation
                    reward = -10
                    next_stud_mat = self.current_stud_mat
                    next_stud_mat_layer = self.current_stud_mat_layer
                    # terminated = True
            else:
                # no more layer
                terminated = True
                next_stud_mat = self.current_stud_mat
                next_stud_mat_layer = self.current_stud_mat_layer
                reward = 0
            # current_value = calculate_total_value(self.stud_mat_list, target_stud_mat_list)
            # reward = current_value - last_value
            # reward = current_value
        else:
            placements_list = get_all_possible_placements(self.current_stud_mat_layer, self.base_brick_stud_mat, mode="full")
            if action_decode in placements_list:
                xunit, zunit = action_decode

                # save bricks ldr for visualizing
                brick = Brick(0,-24,0, DEFAULT_TRANSFORM, BRICK_TYPE, BRICK_COLOR + self.current_layer, "#8A12A8")
                brick.unit_translate(xunit, -self.current_layer, zunit)
                self.bricks_list.append(brick)
                self.bricks_per_level[self.current_layer] += 1

                next_stud_mat = update_occupied_stud_matrx(self.current_stud_mat.copy(), 
                            np.ones(BASE_BRICK_SHAPE), xunit, zunit)
                next_stud_mat_layer = update_occupied_stud_matrx(self.current_stud_mat_layer.copy(), 
                            np.ones(BASE_BRICK_SHAPE), xunit, zunit)
                current_value = calculate_total_value(self.stud_mat_list, target_stud_mat_list)
                reward = current_value - last_value
                # reward = current_value
                # reward = 4 # base size
            else:
                # immediate terminate when taking invalid placement
                next_stud_mat = self.current_stud_mat
                next_stud_mat_layer = self.current_stud_mat_layer
                reward = -10 # illegal move
                # terminated = True
        # print(self.current_layer)
        # print(next_stud_mat.shape)
        self.stud_mat_list[self.current_layer] = next_stud_mat
        self.current_stud_mat = self.stud_mat_list[self.current_layer]
        # print(self.stud_mat_list)
        self.stud_mat_layer_list[self.current_layer] = next_stud_mat_layer
        self.current_stud_mat_layer = self.stud_mat_layer_list[self.current_layer]

        observation = []
        for mat in self.stud_mat_list:
            observation += list(mat.ravel())
        # observation += list(self.prev_actions)
        observation = np.array(observation, dtype=np.int16)

        if terminated:
            if len(self.bricks_list) > 1:
                model = LegoModel(brick=self.bricks_list[0])
                for brick in self.bricks_list[1:]:
                    model.add_brick(brick)
                model.generate_ldr_file("test.ldr")
                # generate_image()

        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=42):
        self.bricks_list = []

        self.stud_mat_list = [np.zeros((HEIGHT, WIDTH)) for i in range(N_CHANNELS) ]
        self.current_stud_mat = self.stud_mat_list[self.current_layer]

        self.stud_mat_layer_list = [np.zeros((HEIGHT, WIDTH)) for i in range(N_CHANNELS) ]
        self.current_stud_mat_layer = self.stud_mat_layer_list[self.current_layer]

        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(N_CHANNELS*HEIGHT*WIDTH ,), dtype=np.int16)
        self.current_layer = 0
        self.bricks_per_level = [0 for _ in range(HEIGHT)]
        
        self.prev_actions = deque([-1. for _ in range(PREV_ACTIONS_QUEUE_LEN)], maxlen=PREV_ACTIONS_QUEUE_LEN) 
        observation = []
        for mat in self.stud_mat_list:
            observation += list(mat.ravel())
        # observation += list(self.prev_actions)
        observation = np.array(observation, dtype=np.int16)
        info = {}
        return observation, info  # reward, done, info can't be included
    
    def render(self, mode='human'):
        pass

    def close (self):
        ...



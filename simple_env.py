import gymnasium as gym
from gymnasium import spaces
import numpy as np
from lego import Brick, LegoModel
from stud_control import update_occupied_stud_matrx, get_all_possible_placements
import helpers

ALL_ACTIONS = [(0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6), 
  (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5), 
  (0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), 
  (0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3), 
  (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), 
  (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), 
  (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
     "moveup", "done"] # move to next layer
N_DISCRETE_ACTIONS = len(ALL_ACTIONS)

ACTIONS_MAP = dict(zip(range(N_DISCRETE_ACTIONS), ALL_ACTIONS))
# print(ACTIONS_MAP)



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
    np.array([[1., 1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 1.]]),
    np.array([[1., 1., 1., 1.],
    [1., 1., 1., 1.],
    [1., 1., 1., 1.],
    [1., 1., 1., 1.]]),
    np.array([[1., 1.],
    [1., 1.]])
]
# default transform for lego brick
DEFAULT_TRANSFORM = helpers.build_translation_matrix(0, -24, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)
BRICK_TYPE = "3003.dat" # 2x2 brick code
BRICK_COLOR = 60 # base layer color

def calculate_reward(current_stud_mat_list, target_stud_mat_list):
    reward = 0
    for current, target in zip(current_stud_mat_list, target_stud_mat_list):
        reward += np.sum(np.multiply(current, target)) 
    return int(reward)
    
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
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(120,), dtype=np.uint8)
        self.current_layer = 0
        self.stud_mat_list = [np.zeros((HEIGHT-i*2, WIDTH-i*2)) for i in range(N_CHANNELS)]
        self.current_stud_mat = self.stud_mat_list[self.current_layer]
        self.base_brick_stud_mat = np.ones(BASE_BRICK_SHAPE)
        self.brick_list = []
    
    def step(self, action):
        info = {}
        truncated = False
        action_decode = ACTIONS_MAP[action]
        if action_decode != "done":
            terminated = False
            if action_decode == "moveup":
                if self.current_layer < N_CHANNELS-1:
                    self.current_layer += 1
                    next_stud_mat = np.zeros((HEIGHT-self.current_layer*2, WIDTH-self.current_layer*2))
                else:
                    terminated = True
                    next_stud_mat = self.current_stud_mat
                reward = calculate_reward(self.stud_mat_list, target_stud_mat_list)
            else:
                placements_list = get_all_possible_placements(self.current_stud_mat, self.base_brick_stud_mat, mode="valid")
                if action_decode in placements_list:
                    brick = Brick(0,-24,0, DEFAULT_TRANSFORM, BRICK_TYPE, BRICK_COLOR + self.current_layer, "#8A12A8")
                    self.brick_list.append(brick)
                    xunit, zunit = action_decode
                    next_stud_mat = update_occupied_stud_matrx(self.current_stud_mat.copy(), 
                                np.ones(BASE_BRICK_SHAPE), xunit, zunit)
                    reward = calculate_reward(self.stud_mat_list, target_stud_mat_list)
                else:
                    next_stud_mat = self.current_stud_mat
                    reward = -100
            self.current_stud_mat = next_stud_mat
            self.stud_mat_list[self.current_layer] = self.current_stud_mat
        else:
            terminated = True
            reward = calculate_reward(self.stud_mat_list, target_stud_mat_list)
        observation = []
        for mat in self.stud_mat_list:
            observation += list(mat.ravel())
        observation = np.array(observation, dtype=np.uint8)

        if len(self.brick_list) > 1:
            model = LegoModel(brick=self.bricks_list[0])
            for brick in self.bricks_list[1:]:
                model.add_brick(brick)
            model.generate_ldr_file("test.ldr")

        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=42):
        self.brick_list = []
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(120,), dtype=np.uint8)
        self.current_layer = 0
        self.stud_mat_list = [np.zeros((HEIGHT-i*2, WIDTH-i*2)) for i in range(N_CHANNELS)]
        self.current_stud_mat = self.stud_mat_list[self.current_layer]
        # print(self.stud_mat_list)
        observation = []
        for mat in self.stud_mat_list:
            observation += list(mat.ravel())
        observation = np.array(observation, dtype=np.uint8)
        info = {}
        return observation, info  # reward, done, info can't be included
    
    def render(self, mode='human'):
        pass

    def close (self):
        ...



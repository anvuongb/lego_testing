from stud_control import get_all_possible_placements, update_occupied_stud_matrx
import numpy as np
from scipy.signal import correlate2d

from simple_env_v3 import base_action_generation, base_action_generation_multiple_bricks, BRICK_TYPE_MAP

base = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 1, 1, 1, 0, 0],
                 [0, 0, 0, 0, 0, 1, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]])
# base = np.array([[0,0,1,0],
#                  [0,1,1,1],
#                  [0,1,0,0],
#                  [0,0,0,0]])
stud1 = np.array([[1,1],
                  [1,1]])
stud2 = np.array([[1,1,1],
                  [1,1,1]])

PLACEMENT_MODE = "full"
actions = base_action_generation_multiple_bricks(base, [0, 1, 2], mode=PLACEMENT_MODE)
print(actions)
print(len(actions))
for action in actions:
    print(action)
    idx, xunit, zunit = action
    stud, rotate = np.ones(BRICK_TYPE_MAP[idx][0]), BRICK_TYPE_MAP[idx][-1]
    if rotate:
        stud = stud.transpose()
    print(stud)
    stud_mat = update_occupied_stud_matrx(base.copy(), np.ones(stud.shape), xunit, zunit)
    print(base)
    print(stud_mat)
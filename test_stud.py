from stud_control import get_all_possible_placements, update_occupied_stud_matrx
import numpy as np
from scipy.signal import correlate2d

base = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]])
base = np.array([[0,0,1,0],
                 [0,1,1,1],
                 [0,1,0,0],
                 [0,0,0,0]])
stud = np.array([[1,1],
                 [1,1]])

corr = correlate2d(base, stud, mode="valid", boundary='fill', fillvalue=0.0)
print(corr)

def base_action_generation(base_stud_mat, mode="valid"):
    actions = get_all_possible_placements(base, base_stud_mat, mode=mode)
    return actions
# PLACEMENT_MODE = "full"
# actions = get_all_possible_placements(base, stud, mode=PLACEMENT_MODE)
# print(actions)
# print(len(actions))
# for action in actions:
#     print(action)
#     xunit, zunit = action
#     stud_mat = update_occupied_stud_matrx(base.copy(), np.ones(stud.shape), xunit, zunit)
#     print(base)
#     print(stud_mat)
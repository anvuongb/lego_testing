from stud_control import get_all_possible_placements, update_occupied_stud_matrx
import numpy as np

base = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]])
stud = np.array([[1,1],
                 [1,1]])
PLACEMENT_MODE = "valid"
actions = get_all_possible_placements(base, stud, mode=PLACEMENT_MODE)
print(actions)
print(len(actions))
# for action in actions:
#     print(action)
#     xunit, zunit = action
#     stud_mat = update_occupied_stud_matrx(base.copy(), np.ones(stud.shape), xunit, zunit)
#     print(base)
#     print(stud_mat)
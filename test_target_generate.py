import numpy as np

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

for m in target_stud_mat_list:
    print(m)
    print("\n")
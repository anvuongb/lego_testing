import numpy as np
from scipy.signal import correlate2d

def get_all_possible_placements(stud_mat1, stud_mat2, mode='full'):
    '''
    Create a list of all possible translation (no rotation) of Brick2
    to fit on top of Brick1
    stud_mat1: stud matrix of brick 1
    stud_mat2: stud matrix of brick 2
    return: a list of possible unit translations for brick 2
    '''
    corr = correlate2d(stud_mat1, stud_mat2, mode=mode, boundary='fill', fillvalue=0.0)
    # print("corr matrix")
    # print(corr)
    row_idx, col_idx = np.where(corr == 0)
    if len(row_idx) == 0 or len(col_idx) == 0:
        return []
    # print(list(zip(row_idx, col_idx)))
    if mode == "full":
        row_idx = -(row_idx - stud_mat1.shape[0] + 1) # reverse since matrix representation is different than coordinate representation for this direction
        col_idx = col_idx - stud_mat2.shape[1] + 1
    if mode == "valid":
        row_idx = stud_mat1.shape[0]-stud_mat2.shape[0]-row_idx  # reverse since matrix representation is different than coordinate representation for this direction
        col_idx = col_idx
    # print(list(zip(row_idx, col_idx)))
    return list(zip(col_idx, row_idx))

def update_occupied_stud_matrx(stud_mat, top_mat, xunit, zunit):
    '''
    update occupancy matrix when placing Brick2 (top_mat) on top of Brick1 (stud_mat)
    Brick2 is translated by xunit and zunit
    '''
    rowmax_orig = stud_mat.shape[0]
    rowmin_orig = stud_mat.shape[0] - top_mat.shape[0]

    colmin_orig = 0
    colmax_orig = top_mat.shape[1]

    # print(rowmax_orig, rowmin_orig)
    # print(colmin_orig, colmax_orig)

    rowmin, rowmax = np.clip(rowmin_orig - zunit, a_min=0, a_max=stud_mat.shape[0]), np.clip(rowmax_orig - zunit, a_min=0, a_max=stud_mat.shape[0])
    colmin, colmax = np.clip(colmin_orig + xunit, a_min=0, a_max=stud_mat.shape[1]), np.clip(colmax_orig + xunit, a_min=0, a_max=stud_mat.shape[1])

    # print(rowmin, rowmax)
    # print(colmin, colmax)

    stud_mat[rowmin:rowmax, colmin:colmax] = 1
    return stud_mat
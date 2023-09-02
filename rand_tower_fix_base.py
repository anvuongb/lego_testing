import numpy as np
import plotly.graph_objects as go
import itertools
import copy
import helpers
from lego import Brick, LegoModel, LayerBrick
from stud_control import get_all_possible_placements, update_occupied_stud_matrx

if __name__ == "__main__":
    opacity = 0.5
    marker_size = 3
    line_width = 4

    # bricks_bank = ["2456.dat", "3001.dat", "3002.dat", "3003.dat", "3004.dat", "3006.dat", "3008.dat"]
    bricks_bank = ["3001.dat", "3002.dat", "3003.dat", "3004.dat", "3005.dat"]
    bricks_bank_budget = {
        "3001.dat":5, 
        "3002.dat":5, 
        "3003.dat":5, 
        "3004.dat":5, 
        "3005.dat":5
    }
    colors_bank = [60, 62, 64, 66, 68, 72, 75, 77, 79, 81, 83, 85, 87]
    default_tm = helpers.build_translation_matrix(0, -24, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)

    bricks_height = 3
    bricks_per_level = sum(bricks_bank_budget.values())

    bricks_list = []
    # brick_type = bricks_bank[np.random.randint(0, len(bricks_bank))]
    brick_type = "base10x10"
    # brick_type = "base20x20"
    brick_color = colors_bank[np.random.randint(0, len(colors_bank))]
    base_brick = Brick(0,-24, 0, default_tm, brick_type, brick_color, "#8A12A8")
    base_size = 10
    hole_mat = np.zeros((base_size,base_size))
    stud_mat = np.zeros((base_size,base_size))
    if helpers.coin_flip():
        base_brick.rotate_yaxis_90deg_align_origin()
    else:
        base_brick.translate_corner_to_origin()
    # bricks_list.append(base_brick)
    for idx_h in range(0, bricks_height):
        tmp_list = []
        bricks_bank_budget_tmp = copy.deepcopy(bricks_bank_budget)
        for idx_w in range(bricks_per_level):
            found = False
            for brick_type in bricks_bank:
                if bricks_bank_budget_tmp[brick_type] > 0:
                    # try brick from biggest to smallest
                    # brick_type = bricks_bank[np.random.randint(0, len(bricks_bank))]
                    brick_color = colors_bank[np.random.randint(0, len(colors_bank))]
                    rots = [False, True]
                    if helpers.coin_flip():
                        rots = [True, False]
                    for rotation in rots:
                        # try rotation
                        lev_brick = Brick(0,-24,0, default_tm, brick_type, brick_color, "#8A12A8")
                        if rotation:
                            lev_brick.rotate_yaxis_90deg_align_origin()
                        lev_brick.translate_corner_to_brick_relative(base_brick)

                        placements_full_list = get_all_possible_placements(stud_mat, np.ones(lev_brick.stud_matrix.shape), mode="valid", collide_type="brick")
                        placements_partial_list = get_all_possible_placements(hole_mat, np.ones(lev_brick.stud_matrix.shape), mode="valid", collide_type="hole")
                        placements_list = list(set(placements_partial_list).intersection(placements_full_list))

                        # print(placements_full_list)
                        # print(placements_partial_list)
                        # print(placements_list)
                        if len(placements_list) > 0:
                            # randomly pick placement
                            # xunit, zunit = placements_list[np.random.randint(0,len(placements_list))]
                            # pick first placement
                            xunit, zunit = placements_list[0]
                            # update stud mat current layer placements
                            stud_mat = update_occupied_stud_matrx(stud_mat, np.ones(lev_brick.stud_matrix.shape), xunit, zunit)
                            # translate brick2
                            lev_brick.unit_translate(xunit, -(idx_h), zunit)
                            # print("base stud mat")
                            # print(curr_layer.layer_stud_mat)
                            tmp_list.append(lev_brick)
                            bricks_list.append(lev_brick)
                            found = True
                            bricks_bank_budget_tmp[brick_type] = bricks_bank_budget_tmp[brick_type] - 1
                            # print(bricks_bank_budget_tmp)
                            break
                        else:
                            pass
                            # print(f"Error no possible placement found, layer {idx_h}, brick {idx_w}, brick type {brick_type}, trying a smaller brick")
                else:
                    pass
                    # print(f"No more brick for type {brick_type}")
                if found:
                    break
            if not found:
                pass
                # print(f"Error no possible placement found, layer {idx_h}, brick {idx_w}")
        # reset stud mat for new layer
        hole_mat = 1 - stud_mat.copy()
        stud_mat = np.zeros((base_size, base_size))
        # base_brick = tmp_list[np.random.randint(0, len(tmp_list))]
        

    # create model to generate .ldr file
    model = LegoModel(brick=bricks_list[0])
    for brick in bricks_list[1:]:
        model.add_brick(brick)

    model.generate_ldr_file("tower.ldr")

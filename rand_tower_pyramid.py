import numpy as np
import plotly.graph_objects as go
import itertools
import copy
import os
import helpers
import ast
from lego import Brick, LegoModel, LayerBrick
from stud_control import get_all_possible_placements, update_occupied_stud_matrx
import sys

if __name__ == "__main__":
    # get args
    output_dir = None
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    output_filename = "pyramid.ldr"
    if len(sys.argv) > 2:
        output_filename = sys.argv[2]
    opacity = 0.5
    marker_size = 3
    line_width = 4

    # bricks_bank = ["2456.dat", "3001.dat", "3002.dat", "3003.dat", "3004.dat", "3006.dat", "3008.dat"]
    bricks_bank = ["3002.dat", "3003.dat", "3004.dat", "3005.dat"]
    bricks_bank_budget = {
        "3001.dat":100, 
        "3002.dat":100, 
        "3003.dat":100, 
        "3004.dat":100, 
        "3005.dat":100
    }
    colors_bank = [60, 62, 64, 66, 68, 72, 75, 77, 79, 81, 83, 85, 87, 302, 339, 52, 285, 324, 273]
    # colors_bank = [35]
    default_tm = helpers.build_translation_matrix(0, -24, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)

    bricks_height = 10
    if len(sys.argv) > 3:
        bricks_height = ast.literal_eval(sys.argv[3])
    bricks_per_level = sum(bricks_bank_budget.values())

    bricks_list = []
    # brick_type = bricks_bank[np.random.randint(0, len(bricks_bank))]
    # brick_type = "base6x6"
    base = bricks_height*2
    brick_type = f"base{base}x{base}"
    brick_color = colors_bank[np.random.randint(0, len(colors_bank))]
    base_brick = Brick(0,-24, 0, default_tm, brick_type, brick_color, "#8A12A8")
    curr_layer = LayerBrick([base_brick], base_brick.stud_matrix)
    if helpers.coin_flip():
        base_brick.rotate_yaxis_90deg_align_origin()
    else:
        base_brick.translate_corner_to_origin()
    # bricks_list.append(base_brick)
    for idx_h in range(0, bricks_height):
        tmp_list = []
        next_stud_mat = np.zeros((base_brick.stud_matrix.shape[0], base_brick.stud_matrix.shape[1]))
        print("next stud size", next_stud_mat.shape)
        bricks_bank_budget_tmp = copy.deepcopy(bricks_bank_budget)
        layer_color = colors_bank[np.random.randint(0, len(colors_bank))]
        for idx_w in range(bricks_per_level):
            found = False
            for brick_type in bricks_bank:
                if bricks_bank_budget_tmp[brick_type] > 0:
                    # try brick from biggest to smallest
                    # brick_type = bricks_bank[np.random.randint(0, len(bricks_bank))]
                    brick_color = layer_color
                    rots = [False, True]
                    if helpers.coin_flip():
                        rots = [True, False]
                    for rotation in rots:
                        # try rotation
                        lev_brick = Brick(0,-24,0, default_tm, brick_type, brick_color, "#8A12A8")
                        if rotation:
                            lev_brick.rotate_yaxis_90deg_align_origin()
                        lev_brick.translate_corner_to_brick_relative(base_brick)
                        # lev_brick.unit_translate(idx_w, 0, idx_w)

                        placements_list = get_all_possible_placements(curr_layer.layer_stud_mat, np.ones(lev_brick.stud_matrix.shape))
                        if len(placements_list) > 0:
                            # randomly pick placement
                            # xunit, zunit = placements_list[np.random.randint(0,len(placements_list))]
                            # pick first placement
                            xunit, zunit = placements_list[0]
                            # update stud mat current layer placements
                            curr_layer.layer_stud_mat = update_occupied_stud_matrx(curr_layer.layer_stud_mat, np.ones(lev_brick.stud_matrix.shape), xunit, zunit)
                            # update stud mat for next layer
                            next_stud_mat = update_occupied_stud_matrx(next_stud_mat, np.ones(lev_brick.stud_matrix.shape), xunit, zunit)
                            # translate brick2
                            lev_brick.unit_translate(xunit, -(idx_h), zunit)
                            print("base stud mat")
                            print(curr_layer.layer_stud_mat)
                            tmp_list.append(lev_brick)
                            bricks_list.append(lev_brick)
                            found = True
                            bricks_bank_budget_tmp[brick_type] = bricks_bank_budget_tmp[brick_type] - 1
                            print(bricks_bank_budget_tmp)
                            break
                        else:
                            print(f"Error no possible placement found, layer {idx_h}, brick {idx_w}, brick type {brick_type}, trying a smaller brick")
                else:
                    print(f"[WARNING] No more brick for type {brick_type}")
                if found:
                    break
            if not found:
                print(f"Error no possible placement found, layer {idx_h}, brick {idx_w}")
        next_stud_mat = next_stud_mat[1:base_brick.stud_matrix.shape[0]-1,1:base_brick.stud_matrix.shape[1]-1]
        curr_layer = LayerBrick(tmp_list, 1-next_stud_mat)
        new_base_brick = Brick(0,-24, 0, default_tm, "base"+str(base_brick.stud_matrix.shape[0]-2)+"x"+str(base_brick.stud_matrix.shape[1]-2), brick_color, "#8A12A8")
        new_base_brick.translate_corner_to_brick_relative(base_brick)
        new_base_brick.unit_translate(1, 0, 1)
        base_brick = new_base_brick
        

    # create model to generate .ldr file
    model = LegoModel(brick=bricks_list[0])
    for brick in bricks_list[1:]:
        model.add_brick(brick)

    model.generate_ldr_file(os.path.join(output_dir, f"base_{base}"+"_"+output_filename)+".ldr")
    print("exporting", os.path.join(output_dir, f"base_{base}"+"_"+output_filename)+".ldr")

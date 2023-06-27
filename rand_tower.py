import numpy as np
import plotly.graph_objects as go
import itertools
import helpers
from lego import Brick, LegoModel
from stud_control import get_all_possible_placements, update_occupied_stud_matrx

if __name__ == "__main__":
    opacity = 0.5
    marker_size = 3
    line_width = 4

    bricks_bank = ["3001.dat", "3003.dat", "3004.dat", "3006.dat", "3008.dat"]
    colors_bank = [72, 75, 77, 79, 81, 83, 85, 87]
    default_tm = helpers.build_translation_matrix(0, -24, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)

    bricks_height = 3
    bricks_per_level = 2

    bricks_list = []
    brick_type = bricks_bank[np.random.randint(0, len(bricks_bank))]
    brick_color = colors_bank[np.random.randint(0, len(colors_bank))]
    base_brick = Brick(0,-24, 0, default_tm, brick_type, brick_color, "#8A12A8")
    base_brick.translate_corner_to_origin()
    bricks_list.append(base_brick)
    for idx_h in range(1, bricks_height+1):
        tmp_list = []
        for idx_w in range(bricks_per_level):
            brick_type = bricks_bank[np.random.randint(0, len(bricks_bank))]
            brick_color = colors_bank[np.random.randint(0, len(colors_bank))]
            lev_brick = Brick(0,-24,0, default_tm, brick_type, brick_color, "#8A12A8")
            lev_brick.translate_corner_to_brick_relative(base_brick)

            placements_list = get_all_possible_placements(base_brick.stud_matrix, np.ones(lev_brick.stud_matrix.shape))
            if len(placements_list) > 0:
                xunit, zunit = placements_list[np.random.randint(0,len(placements_list))]
                # update brick1 occupancy stud mat
                base_brick.stud_matrix = update_occupied_stud_matrx(base_brick.stud_matrix, np.ones(lev_brick.stud_matrix.shape), xunit, zunit)
                # translate brick2
                lev_brick.unit_translate(xunit, -(idx_h), zunit)
                print("base stud mat")
                print(base_brick.stud_matrix)
                tmp_list.append(lev_brick)
                bricks_list.append(lev_brick)
            else:
                print(f"Error no possible placement found, layer {idx_h}, brick {idx_w}")
        base_brick = tmp_list[np.random.randint(0, len(tmp_list))]
        

    # create model to generate .ldr file
    model = LegoModel(brick=bricks_list[0])
    for brick in bricks_list[1:]:
        model.add_brick(brick)

    model.generate_ldr_file("test.ldr")

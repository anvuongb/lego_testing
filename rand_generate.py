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

    tm = helpers.build_translation_matrix(0, -24, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)
    brick1 = Brick(0,-24,0, tm, "3001.dat", 72, "#8A12A8")

    tm = helpers.build_translation_matrix(0, -24, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)
    brick2 = Brick(0,-24,0, tm, "3003.dat", 75, "#8A12A8")
    tm = helpers.build_translation_matrix(0, -24, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)
    brick3 = Brick(0,-24,0, tm, "3001.dat", 77, "#8A12A8")
    tm = helpers.build_translation_matrix(0, -24, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)
    brick4 = Brick(0,-24,0, tm, "3004.dat", 79, "#8A12A8")
    tm = helpers.build_translation_matrix(0, -24, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)
    brick5 = Brick(0,-24,0, tm, "3006.dat", 81, "#8A12A8")
    tm = helpers.build_translation_matrix(0, -24, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)
    brick6 = Brick(0,-24,0, tm, "3008.dat", 83, "#8A12A8")

    pre_list = [brick1, brick2, brick3, brick4, brick5, brick6]
    rand_indices = np.random.permutation(len(pre_list))
    bricks_list = [pre_list[idx] for idx in rand_indices]
    bricks_list[0].translate_corner_to_origin()

    # randomly put brick on top of each other
    for idx in range(len(bricks_list)-1):
        bricks_list[idx+1].translate_corner_to_brick_relative(bricks_list[idx])
        placements_list = get_all_possible_placements(bricks_list[idx].stud_matrix, np.ones(bricks_list[idx+1].stud_matrix.shape))
        if len(placements_list) > 0:
            xunit, zunit = placements_list[np.random.randint(0,len(placements_list))]
            # update brick1 occupancy stud mat
            bricks_list[idx].stud_matrix = update_occupied_stud_matrx(bricks_list[idx].stud_matrix, np.ones(bricks_list[idx+1].stud_matrix.shape), xunit, zunit)
            # translate brick2
            bricks_list[idx+1].unit_translate(xunit, -(idx+1), zunit)
            print("Placement")
            print(xunit, zunit)
            print(f"Brick {idx} stud matrix after placing Brick {idx+1}")
            print(bricks_list[idx].stud_matrix)
            print("\n")
        else:
            print("Error no possible placement found")

    # create model to generate .ldr file
    model = LegoModel(brick=bricks_list[0])
    for brick in bricks_list[1:]:
        model.add_brick(brick)

    model.generate_ldr_file("test.ldr")

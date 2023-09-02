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
    brick2 = Brick(0,-24,0, tm, "3006.dat", 83, "#8A12A8")
    brick2.rotate_yaxis_90deg_align_origin()
    
    brick1.translate_corner_to_origin()
    print(brick2.center_x, brick2.center_z)
    brick2.translate_corner_to_brick_relative(brick1)
    placements_list = get_all_possible_placements(brick1.stud_matrix, np.ones(brick2.stud_matrix.shape))
    
    if len(placements_list) > 0:
        xunit, zunit = placements_list[np.random.randint(0,len(placements_list))]
        # if brick2.rotated:
        #     xunit = -xunit 
        # update brick1 occupancy stud mat
        brick1.stud_matrix = update_occupied_stud_matrx(brick1.stud_matrix, np.ones(brick2.stud_matrix.shape), xunit, zunit)
        # translate brick2
        brick2.unit_translate(xunit, -1, zunit)
        print("Placement")
        print(xunit, zunit)
        print(brick1.stud_matrix)
        print("\n")
    else:
        print("Error no possible placement found")
        
    # create model to generate .ldr file
    model = LegoModel(brick=brick1)
    model.add_brick(brick2)

    model.generate_ldr_file("test.ldr")
    
    plotly_data = model.build_plotly(opacity, marker_size, line_width)
    layout = go.Layout(
                scene=dict(
                aspectmode='data'
            ))
    fig = go.Figure(data=plotly_data, layout=layout)
    fig.write_html('plotly_render.html', auto_open=False)

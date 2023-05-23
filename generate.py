import numpy as np
import pandas as pd
import json
import plotly.graph_objects as go
import helpers
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

if __name__ == "__main__":
    # load color dict
    with open("color_codes.json", "r") as f:
        color_dict = json.load(f)

    # ldr_filename = "./ldr_files/dataset/2blocks-perpendicular_15.ldr"
    # ldr_filename = "./ldr_files/dataset/wall_augmented270_18.ldr"
    # ldr_filename = "./ldr_files/block_fake.ldr"
    # ldr_filename = "./ldr_files/2bricks_cross.ldr"
    # ldr_filename = "./ldr_files/5bricks_rotate.ldr"
    # ldr_filename = "./ldr_files/7bricks_rotate.ldr"
    ldr_filename = "./ldr_files/different_bricks.ldr"
    print(f"constructing model for {ldr_filename}")

    columns = ["line_type", "color_code", "x", "y", "z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "file_block"]
    df = pd.read_csv(ldr_filename, sep=" ", names=columns)
    df["block_size"] = df["file_block"].apply(lambda x: helpers.decode_file_block(x))

    plotly_data = []
    opacity = 0.5
    marker_size = 3
    line_width = 4

    for idx, row in df.iterrows():
        # get default translation matrix from ldr
        tm = helpers.build_translation_matrix(row.x, row.y, row.z, 
                                              row.a, row.b, row.c, 
                                              row.d, row.e, row.f, 
                                              row.g, row.h, row.i)

        # construct a default brick
        brick = helpers.Brick(row.x, row.y, row.z, 
                            tm, 
                            row.file_block,
                            color_dict[str(row.color_code)])

        # apply test transformation
        # if idx == 8:
            # brick.rotate(30)
            # brick.translate(40, 24, -20)
            # brick.unit_translate(2,1,-1)

        # get 3d coordinates
        vertices = brick.get_vertices()

        plotly_data += brick.build_plotly(opacity, marker_size, line_width)
        
    layout = go.Layout(
                scene=dict(
                aspectmode='data'
            ))
    fig = go.Figure(data=plotly_data, layout=layout)
    fig.show()

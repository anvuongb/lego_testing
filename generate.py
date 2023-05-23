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
    ldr_filename = "./ldr_files/dataset/wall_augmented270_18.ldr"
    # ldr_filename = "./ldr_files/block_fake.ldr"
    # ldr_filename = "./ldr_files/2bricks_cross.ldr"
    # ldr_filename = "./ldr_files/5bricks_rotate.ldr"
    # ldr_filename = "./ldr_files/7bricks_rotate.ldr"
    print(ldr_filename)

    columns = ["line_type", "color_code", "x", "y", "z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "file_block"]
    df = pd.read_csv(ldr_filename, sep=" ", names=columns)
    df["block_size"] = df["file_block"].apply(lambda x: helpers.decode_file_block(x))

    all_vertices_list = []
    plotly_data = []
    opacity = 0.5
    marker_size = 3
    line_width = 4

    for idx, row in df.iterrows():
        tm = np.array([[row.a, row.b, row.c, row.x],
                    [row.d, row.e, row.f, row.y],
                    [row.g, row.h, row.i, row.z],
                    [0, 0, 0, 1]])
        print(tm)
        vertices = helpers.build_vertices(0, 0, 0, df.block_size[idx])
        print(vertices.shape, vertices)
        vertices = np.hstack([vertices, np.ones((8,1))])
        vertices = np.matmul(tm, vertices.T)
        print(vertices.shape, vertices)
        vertices = vertices[:3,:].T
        all_vertices_list += list(vertices)
        print(vertices.shape, vertices)
        print('\n\n')

        plotly_data += helpers.build_plotly(vertices, opacity, color_dict, row.color_code, marker_size, line_width)
        

    fig = go.Figure(data=plotly_data)

    # fig.update_layout(
    #     scene = dict(
    #         xaxis = dict(nticks=4, range=[-100,100],),
    #                      yaxis = dict(nticks=4, range=[-50,100],),
    #                      zaxis = dict(nticks=4, range=[-100,100],),),
    #     width=700,
    #     margin=dict(r=20, l=10, b=10, t=10))
    fig.update_layout(
        scene = dict(
            aspectratio=dict(x=1, y=1, z=1), # <---- tried this too
            aspectmode='cube'
        ),
        # template='plotly_dark',
    )
    fig.show()

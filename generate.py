import numpy as np
import pandas as pd
import json
import plotly.graph_objects as go
import helpers
from lego import Brick, LegoModel
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

if __name__ == "__main__":
    # load color dict
    # with open("color_codes.json", "r") as f:
    #     color_dict = json.load(f)

    # ldr_filename = "./ldr_files/dataset/2blocks-perpendicular_15.ldr"
    # ldr_filename = "./ldr_files/dataset/wall_augmented270_18.ldr"
    # ldr_filename = "./ldr_files/block_fake.ldr"
    # ldr_filename = "./ldr_files/2bricks_cross.ldr"
    # ldr_filename = "./ldr_files/5bricks_rotate.ldr"
    # ldr_filename = "./ldr_files/7bricks_rotate.ldr"
    # ldr_filename = "./ldr_files/different_bricks.ldr"

    opacity = 0.5
    marker_size = 3
    line_width = 4
    model1 = LegoModel(filepath="./ldr_files/2bricks_cross.ldr", color_code_file="color_codes.json", save_transformation_history=False)
    model2 = LegoModel(filepath="./ldr_files/7bricks_rotate.ldr", color_code_file="color_codes.json", save_transformation_history=False)
    model1.unit_translate(-2, -2, 5)
    model1.rotate_yaxis(45)
    model1.generate_ldr_file("test.ldr")
    plotly_data1 = model1.build_plotly(opacity, marker_size, line_width)
    plotly_data2 = model2.build_plotly(opacity, marker_size, line_width)
    plotly_data = plotly_data1 + plotly_data2

    layout = go.Layout(
                scene=dict(
                aspectmode='data'
            ))
    fig = go.Figure(data=plotly_data, layout=layout)
    fig.show()

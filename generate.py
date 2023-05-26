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
    model_2brick_cross = LegoModel(filepath="./ldr_files/2bricks_cross.ldr", 
                                   color_code_file="color_codes.json", 
                                   save_transformation_history=False)
    model_7brick_rotate = LegoModel(filepath="./ldr_files/7bricks_rotate.ldr", 
                                    color_code_file="color_codes.json", 
                                    save_transformation_history=False)
    model_9brick_different = LegoModel(filepath="./ldr_files/different_bricks.ldr", 
                                    color_code_file="color_codes.json", 
                                    save_transformation_history=False)
    # model_1brick_float = LegoModel(filepath="./ldr_files/1brick_float.ldr", color_code_file="color_codes.json", save_transformation_history=False)

    # rotate 2bricks_cross by 45deg and translate z-axis by 3 lego unit
    model_2brick_cross.rotate_yaxis(45)
    model_2brick_cross.unit_translate(-1, 0, 5)

    # add 1 brick to this model
    tm = helpers.build_translation_matrix(20, -96, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)
    brick = Brick(20, -96, 0, tm, "3001.dat", 69, "#8A12A8")
    model_8brick_rotate = model_7brick_rotate + brick # the type here is LegoModel + Brick, these types overloading may cause confusion later ...
    # rotate 8brick_rotate by 45deg
    model_8brick_rotate.rotate_yaxis(-20)

    # rotate 9bricks 90 deg then shift right a bit
    model_9brick_different.rotate_yaxis(90)
    model_9brick_different.unit_translate(2, 0, 0)
    # move the top brick 
    b = model_9brick_different.get_top_brick()
    # b.unit_translate(-2, 0, 0)
    # b.rotate_yaxis(90)
    model_9brick_different.recalculate_center()
    model_9brick_different.update_sorted_bricks_by_height()
    
    # combine into 1 model 
    model_combine = model_8brick_rotate + model_2brick_cross + model_9brick_different # the type here is LegoModel + LegoModel

    # generate ldr files
    model_combine.generate_ldr_file("model_combine.ldr")

    # plot in plotly
    plotly_data = model_combine.build_plotly(opacity, marker_size, line_width)
    plotly_data_center = model_combine.build_plotly_center(opacity, marker_size, line_width)
    plotly_data_bricks_center = model_combine.build_plotly_bricks_center(opacity, marker_size, line_width)
    plotly_data = plotly_data + plotly_data_center + plotly_data_bricks_center 

    layout = go.Layout(
                scene=dict(
                aspectmode='data'
            ))
    fig = go.Figure(data=plotly_data, layout=layout)
    fig.write_html('plotly_render.html', auto_open=False)
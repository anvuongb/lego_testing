import numpy as np
import pandas as pd
import plotly.graph_objects as go
import helpers
import json
from typing import Type

class LegoModel(object):
    def __init__(self, filepath=None, color_code_file=None, save_transformation_history=False):
        self.filepath = filepath
        self.bricks = []
        self.transformation_list = []
        self.save_transformation_history = save_transformation_history

        if color_code_file != None:
            self.color_code_file = color_code_file
        else:
            self.color_code_file = "color_codes.json"
        
        with open(self.color_code_file, "r") as f:
            self.color_dict = json.load(f)

        if self.filepath is not None:
            self.load_from_ldr(self.filepath)
        return
    
    def recalculate_center(self):
        # calculate center
        self.center_x = np.average([b.center_x for b in self.bricks])
        self.center_y = np.average([b.center_y for b in self.bricks])
        self.center_z = np.average([b.center_z for b in self.bricks])

    def __add__(self, other):
        '''
        combine 2 lego mode, all transformations history will be deleted
        save_transformation_history is set to False
        color_dict are merged
        filepath is deleted
        bricks are merged
        '''
        if type(other) == LegoModel:
            model = LegoModel()

            model.bricks = self.bricks + other.bricks
            model.color_code_file = self.color_code_file
            self.color_dict.update(other.color_dict)
            model.color_dict = self.color_dict
            model.filepath = None
            model.transformation_list = []
            model.save_transformation_history = False
            model.recalculate_center()

            return model
        
        if type(other) == Brick:
            self.add_brick(other)
            return self

    def add_brick(self, brick):
        self.bricks.append(brick)
        self.clear_transformation_history()
        self.recalculate_center()

    def clear_transformation_history(self):
        '''
        delete transformation history to avoid memory leak
        '''
        self.transformation_list = []

    def load_from_ldr(self, filepath):
        '''
        build model from ldr file
        '''
        print(f"Building model from {filepath}")
        columns = ["line_type", "color_code", "x", "y", "z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "file_block"]
        df = pd.read_csv(filepath, sep=" ", names=columns)
        df["block_size"] = df["file_block"].apply(lambda x: helpers.decode_file_block(x))
        for _, row in df.iterrows():
            # get default translation matrix from ldr
            tm = helpers.build_translation_matrix(row.x, row.y, row.z, 
                                                row.a, row.b, row.c, 
                                                row.d, row.e, row.f, 
                                                row.g, row.h, row.i)

            # construct a brick
            brick = Brick(row.x, row.y, row.z, 
                        tm, 
                        row.file_block,
                        row.color_code, 
                        self.color_dict[str(row.color_code)])
            self.bricks.append(brick)

        # calculate center
        self.center_x = np.average([b.center_x for b in self.bricks])
        self.center_y = np.average([b.center_y for b in self.bricks])
        self.center_z = np.average([b.center_z for b in self.bricks])

    def build_plotly(self, opacity=0.5, marker_size=3, line_width=4):
        '''
        return a list of plotly data for all bricks
        '''
        data = []
        for brick in self.bricks:
            data += brick.build_plotly(opacity, marker_size, line_width)
        return data
    
    def build_plotly_center(self, opacity=0.5, marker_size=3, line_width=4):
        return [self.bricks[0]._build_plotly_data(
            [self.center_x],
            [self.center_y],
            [self.center_z],
            1, "#000000", marker_size, line_width
        )]
    
    def build_plotly_bricks_center(self, opacity=0.5, marker_size=3, line_width=4):
        '''
        return a list of plotly data for all bricks
        '''
        data = []
        for brick in self.bricks:
            data += brick.build_plotly_center(opacity, marker_size, line_width)
        return data
    
    def rotate_yaxis(self, angle):
        '''
        rotate this model around y-axis, which is the up dimension in ldr standard
        translate the center to origin, pivot point
        '''
        for idx, _ in enumerate(self.bricks):
            self.bricks[idx].rotate_yaxis(angle, pivot_point=[self.center_x, self.center_y, self.center_z])
        self.recalculate_center()

    def translate(self, x, y, z):
        '''
        translate this model by x, y, z amount
        '''
        tm = np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])
        self.apply_transformation(tm)
        self.recalculate_center()

    def unit_translate(self, x, y, z):
        '''
        this is the same as translate() but the unit is scaled to lego dimension
        x, z (width, depth) unit increment of 20
        -y (height) unit increment of 24
        '''
        # apply transform to each brick
        for idx, _ in enumerate(self.bricks):
            self.bricks[idx].unit_translate(x, y, z)

        if self.save_transformation_history:
            self.transformation_list.append(tm)
            tm = np.array([[1, 0, 0, x],
                        [0, 1, 0, y],
                        [0, 0, 1, z],
                        [0, 0, 0, 1]])
        self.recalculate_center()

    def apply_transformation(self, tm):
        '''
        apply a given transformation matrix tm to this model (all bricks)
        '''
        # apply transform to each brick
        for idx, _ in enumerate(self.bricks):
            self.bricks[idx].apply_transformation(tm)

        # # faster, gather all vertices then apply transformation, but this loses information for individual brick
        # vertices = np.vstack([b.get_vertices() for b in self.bricks])
        # vertices = np.hstack([vertices, np.ones((vertices.shape[0],1))])
        # vertices = np.matmul(tm, vertices.T)
        # for idx, _ in enumerate(self.bricks):
        #     self.bricks[idx].vertices = vertices[:3,idx:idx+8].T

        if self.save_transformation_history:
            self.transformation_list.append(tm)
        self.recalculate_center()
    
    def generate_ldr_file(self, filepath):
        with open(filepath, "w") as f:
            for brick in self.bricks:
                f.write(brick.generate_ldr_line() + "\n")

class Brick(object):
    def __init__(self, x, y, z, default_transform_matrix, block_type, color_code=-1, color_hex="#ffffff", lego_unit_length=20, lego_unit_height=24, save_transformation_history=False):
        self.block_type = block_type
        self.block_size = helpers.decode_file_block(block_type)
        self.color_hex = color_hex
        self.color_code = color_code
        self.unit_length = lego_unit_length
        self.unit_height = lego_unit_height
        self.tm = default_transform_matrix
        self.transformation_list = []
        self.save_transformation_history = save_transformation_history

        # these coordinates are at the center of the brick
        self.center_x = x
        self.center_y = y - self.unit_height/2 # ldr format is nailed to top surface
        self.center_z = z

        # this stores the ldr format
        self.center_x_origin = x
        self.center_y_origin = y
        self.center_z_origin = z

        # initialize numpy array of vertices
        self.vertices = helpers.build_vertices(0, 0, 0, self.block_size, lego_unit_length=self.unit_length, lego_unit_height=self.unit_height)
        self.apply_transformation(self.tm, update_tm=False) # don't update when initialization
    
    def recalculate_center(self):
        # calculate center
        self.center_x, self.center_y, self.center_z = np.average(self.vertices, axis=0)
        self.center_x_origin = self.center_x
        self.center_y_origin = self.center_y - self.unit_height/2
        self.center_z_origin = self.center_z

    def get_current_center(self):
        # center = np.array([self.center_x_origin, self.center_y_origin, self.center_z_origin, 1]).reshape([4,1])
        # return np.matmul(self.tm, center)
        return self.center_x, self.center_y, self.center_z

    def generate_ldr_line(self):
        s = f"1 {self.color_code}" # type 1 + color code
        s += f" {self.center_x_origin} {self.center_y_origin} {self.center_z_origin}"
        s += f" {self.tm[0][0]} {self.tm[0][1]} {self.tm[0][2]}"
        s += f" {self.tm[1][0]} {self.tm[1][1]} {self.tm[1][2]}"
        s += f" {self.tm[2][0]} {self.tm[2][1]} {self.tm[2][2]}"
        s += f" {self.block_type}"
        return s

    def clear_transformation_history(self):
        '''
        delete transformation history to avoid memory leak
        '''
        self.transformation_list = []

    def rotate_yaxis(self, angle, pivot_point = None):
        '''
        rotate this brick around y-axis, which is the up dimension in ldr standard
        translate the center to origin, pivot point
        '''
        if pivot_point is None:
            center_x = self.center_x
            center_y = self.center_y
            center_z = self.center_z
        else:
            center_x = pivot_point[0]
            center_y = self.center_y
            center_z = pivot_point[2]

        self.translate(-center_x, -center_y, -center_z)

        #rotate
        tm = np.array([[np.cos(angle*np.pi/180), 0, -np.sin(angle*np.pi/180), 0],
                     [0, 1, 0, 0],
                     [np.sin(angle*np.pi/180), 0, np.cos(angle*np.pi/180), 0],
                     [0, 0, 0, 1]])
        self.apply_transformation(tm)

        # translate the center back, pivot point
        self.translate(center_x, center_y, center_z)

    def translate(self, x, y, z):
        '''
        translate this brick by x, y, z amount
        '''
        tm = np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])
        # TODO: something is wrong when updating center during translation
        # self.center_x_origin += x
        # self.center_y_origin += y
        # self.center_z_origin += z
        self.apply_transformation(tm)

    def unit_translate(self, x, y, z):
        '''
        this is the same as translate() but the unit is scaled to lego dimension
        x, z (width, depth) unit increment of 20
        -y (height) unit increment of 24
        '''
        self.translate(x*self.unit_length, y*self.unit_height, z*self.unit_length)

    def apply_transformation(self, tm, update_tm=True):
        '''
        apply a given transformation matrix tm to this brick
        '''
        vertices = np.hstack([self.vertices, np.ones((8,1))])
        vertices = np.matmul(tm, vertices.T)
        self.vertices = vertices[:3,:].T
        self.recalculate_center()
        
        # update internal transform
        if update_tm:
            self.tm = np.matmul(self.tm, tm)
            # self.tm = tm

        if self.save_transformation_history:
            self.transformation_list.append(tm)

    def get_transformation_history(self):
        '''
        return a list of transformation performed on this brick
        note that rotation composes of 3 transformation: translate -> rotate -> translate
        '''
        return self.transformation_list

    def get_vertices(self):
        '''
        return a numpy array containing all the vertices
        '''
        return self.vertices
    
    def _build_plotly_data(self, x, y, z, opacity, line_color, marker_size=2, line_width=4):
        d = go.Scatter3d(x=x,
                        y=y,
                        z=z,
                        opacity=opacity,
                        mode='markers+lines',
                        line={"color":line_color, "width":line_width},
                        marker={"size":marker_size}
                        )
        return d
    
    def build_plotly_center(self, opacity=0.5, marker_size=3, line_width=4):
        # return [self._build_plotly_data(
        #     [self.center_x, self.center_x_origin],
        #     [self.center_y, self.center_y_origin],
        #     [self.center_z, self.center_z_origin],
        #     opacity, self.color_hex, marker_size, line_width
        # )]
        return [self._build_plotly_data(
            [self.center_x],
            [self.center_y],
            [self.center_z],
            opacity, self.color_hex, marker_size, line_width
        )]

    def build_plotly(self, opacity=0.5, marker_size=3, line_width=4):
        '''
        return a list of plotly data for this brick
        '''
        plotly_data = []
        # build plotly data
        x_vertices_upper = [self.vertices[0][0], self.vertices[1][0], self.vertices[2][0], self.vertices[3][0], self.vertices[0][0]]
        y_vertices_upper = [self.vertices[0][1], self.vertices[1][1], self.vertices[2][1], self.vertices[3][1], self.vertices[0][1]]
        z_vertices_upper = [self.vertices[0][2], self.vertices[1][2], self.vertices[2][2], self.vertices[3][2], self.vertices[0][2]]
        plotly_data.append(self._build_plotly_data(x_vertices_upper, 
                                            y_vertices_upper, 
                                            z_vertices_upper, 
                                            opacity, self.color_hex, marker_size, line_width))
        
        x_vertices_lower = [self.vertices[4][0], self.vertices[5][0], self.vertices[6][0], self.vertices[7][0], self.vertices[4][0]]
        y_vertices_lower = [self.vertices[4][1], self.vertices[5][1], self.vertices[6][1], self.vertices[7][1], self.vertices[4][1]]
        z_vertices_lower = [self.vertices[4][2], self.vertices[5][2], self.vertices[6][2], self.vertices[7][2], self.vertices[4][2]]
        plotly_data.append(self._build_plotly_data(x_vertices_lower, 
                                            y_vertices_lower, 
                                            z_vertices_lower, 
                                            opacity, self.color_hex, marker_size, line_width))
        
        x_vertices_col1 = [self.vertices[0][0], self.vertices[4][0]]
        y_vertices_col1 = [self.vertices[0][1], self.vertices[4][1]]
        z_vertices_col1 = [self.vertices[0][2], self.vertices[4][2]]
        plotly_data.append(self._build_plotly_data(x_vertices_col1, 
                                            y_vertices_col1, 
                                            z_vertices_col1, 
                                            opacity, self.color_hex, marker_size, line_width))

        x_vertices_col2 = [self.vertices[1][0], self.vertices[5][0]]
        y_vertices_col2 = [self.vertices[1][1], self.vertices[5][1]]
        z_vertices_col2 = [self.vertices[1][2], self.vertices[5][2]]
        plotly_data.append(self._build_plotly_data(x_vertices_col2, 
                                            y_vertices_col2, 
                                            z_vertices_col2, 
                                            opacity, self.color_hex, marker_size, line_width))

        x_vertices_col3 = [self.vertices[2][0], self.vertices[6][0]]
        y_vertices_col3 = [self.vertices[2][1], self.vertices[6][1]]
        z_vertices_col3 = [self.vertices[2][2], self.vertices[6][2]]
        plotly_data.append(self._build_plotly_data(x_vertices_col3, 
                                            y_vertices_col3, 
                                            z_vertices_col3, 
                                            opacity, self.color_hex, marker_size, line_width))

        x_vertices_col4 = [self.vertices[3][0], self.vertices[7][0]]
        y_vertices_col4 = [self.vertices[3][1], self.vertices[7][1]]
        z_vertices_col4 = [self.vertices[3][2], self.vertices[7][2]]
        plotly_data.append(self._build_plotly_data(x_vertices_col4, 
                                            y_vertices_col4, 
                                            z_vertices_col4, 
                                            opacity, self.color_hex, marker_size, line_width))
        
        return plotly_data
import numpy as np
import plotly.graph_objects as go

class Brick(object):
    def __init__(self, x, y, z, default_transform_matrix, block_type, color_hex="#ffffff", lego_unit_length=20, lego_unit_height=24):
        self.block_type = block_type
        self.block_size = decode_file_block(block_type)
        self.color_hex = color_hex
        self.unit_length = lego_unit_length
        self.unit_height = lego_unit_height
        self.tm = default_transform_matrix
        self.transformation_list = []

        # these coordinates are at the center of the brick
        self.init_x = x
        self.init_y = y
        self.init_z = z

        # initialize numpy array of vertices
        self.vertices = build_vertices(0, 0, 0, self.block_size, lego_unit_length=self.unit_length, lego_unit_height=self.unit_height)
        self.apply_transformation(self.tm)

    def rotate(self, angle):
        # translate the center to origin, pivot point
        self.translate(-self.init_x, -self.init_y, -self.init_z)

        #rotate
        tm = np.array([[np.cos(angle*np.pi/180), 0, -np.sin(angle*np.pi/180), 0],
                     [0, 1, 0, 0],
                     [np.sin(angle*np.pi/180), 0, np.cos(angle*np.pi/180), 0],
                     [0, 0, 0, 1]])
        self.apply_transformation(tm)

        # translate the center back, pivot point
        self.translate(self.init_x, self.init_y, self.init_z)

    def translate(self, x, y, z):
        tm = np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])
        self.apply_transformation(tm)

    def unit_translate(self, x, y, z):
        # this is the same as translate() but the unit is scaled to lego dimension
        # x, z (width, depth) unit increment of 20
        # -y (height) unit increment of 24
        self.translate(x*self.unit_length, y*self.unit_height, z*self.unit_length)

    def apply_transformation(self, tm):
        vertices = np.hstack([self.vertices, np.ones((8,1))])
        vertices = np.matmul(tm, vertices.T)
        self.vertices = vertices[:3,:].T
        self.transformation_list.append(tm)

    def get_transformation_history(self):
        return self.transformation_list

    def get_vertices(self):
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
    
    def build_plotly(self, opacity=0.5, marker_size=3, line_width=4):
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


def decode_file_block(f):
    # print(f)
    if f.lower() == "3001.dat":
        return [2,4]
    if f.lower() == "3002.dat":
        return [2,3]
    if f.lower() == "3003.dat":
        return [2,2]
    if f.lower() == "3004.dat":
        return [1,2]
    if f.lower() == "3006.dat":
        return [2,10]
    if f.lower() == "3008.dat":
        return [1,8]
    return [-1,-1]

def get_translation_matrix(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])

def get_rotation_matrix_yaxis(angle):
    # rotate around y axis
    return np.array([[np.cos(angle*np.pi/180), 0, -np.sin(angle*np.pi/180), 0],
                     [0, 1, 0, 0],
                     [np.sin(angle*np.pi/180), 0, np.cos(angle*np.pi/180), 0],
                     [0, 0, 0, 1]])

def build_translation_matrix(x, y, z, a, b, c, d, e, f, g, h, i):
    return np.array([[a, b, c, x],
                     [d, e, f, y],
                     [g, h, i, z],
                     [0, 0, 0, 1]])

def build_vertices(x, y, z, block_size, lego_unit_length=20, lego_unit_height=24):
    # ldr use x, z, -y coordinates (-y is positive up, z is depth)
    v = [[x-block_size[1]*lego_unit_length/2, y, z-block_size[0]*lego_unit_length/2],
         [x+ block_size[1]*lego_unit_length/2 , y, z-block_size[0]*lego_unit_length/2 ],
         [x+ block_size[1]*lego_unit_length/2, y, z+ block_size[0]*lego_unit_length/2],
         [x-block_size[1]*lego_unit_length/2, y, z+ block_size[0]*lego_unit_length/2], # end upper surface
         
         [x-block_size[1]*lego_unit_length/2, y+lego_unit_height, z-block_size[0]*lego_unit_length/2],
         [x+ block_size[1]*lego_unit_length/2 , y+lego_unit_height, z-block_size[0]*lego_unit_length/2 ],
         [x+ block_size[1]*lego_unit_length/2, y+lego_unit_height, z+ block_size[0]*lego_unit_length/2],
         [x-block_size[1]*lego_unit_length/2, y+lego_unit_height, z+ block_size[0]*lego_unit_length/2]] # end lower surface
    return np.array(v).reshape((8,3))

def build_plotly_data(x, y, z, opacity, line_color, marker_size=2, line_width=4):
    d = go.Scatter3d(x=x,
                    y=y,
                    z=z,
                    opacity=opacity,
                    mode='markers+lines',
                    line={"color":line_color, "width":line_width},
                    marker={"size":marker_size}
                    )
    return d


def build_plotly(vertices, opacity, color_dict, color_code, marker_size, line_width):
    plotly_data = []
    # build plotly data
    x_vertices_upper = [vertices[0][0], vertices[1][0], vertices[2][0], vertices[3][0], vertices[0][0]]
    y_vertices_upper = [vertices[0][1], vertices[1][1], vertices[2][1], vertices[3][1], vertices[0][1]]
    z_vertices_upper = [vertices[0][2], vertices[1][2], vertices[2][2], vertices[3][2], vertices[0][2]]
    plotly_data.append(build_plotly_data(x_vertices_upper, 
                                         y_vertices_upper, 
                                         z_vertices_upper, 
                                         opacity, color_dict[str(color_code)], marker_size, line_width))
    
    x_vertices_lower = [vertices[4][0], vertices[5][0], vertices[6][0], vertices[7][0], vertices[4][0]]
    y_vertices_lower = [vertices[4][1], vertices[5][1], vertices[6][1], vertices[7][1], vertices[4][1]]
    z_vertices_lower = [vertices[4][2], vertices[5][2], vertices[6][2], vertices[7][2], vertices[4][2]]
    plotly_data.append(build_plotly_data(x_vertices_lower, 
                                         y_vertices_lower, 
                                         z_vertices_lower, 
                                         opacity, color_dict[str(color_code)], marker_size, line_width))
    
    x_vertices_col1 = [vertices[0][0], vertices[4][0]]
    y_vertices_col1 = [vertices[0][1], vertices[4][1]]
    z_vertices_col1 = [vertices[0][2], vertices[4][2]]
    plotly_data.append(build_plotly_data(x_vertices_col1, 
                                         y_vertices_col1, 
                                         z_vertices_col1, 
                                         opacity, color_dict[str(color_code)], marker_size, line_width))

    x_vertices_col2 = [vertices[1][0], vertices[5][0]]
    y_vertices_col2 = [vertices[1][1], vertices[5][1]]
    z_vertices_col2 = [vertices[1][2], vertices[5][2]]
    plotly_data.append(build_plotly_data(x_vertices_col2, 
                                         y_vertices_col2, 
                                         z_vertices_col2, 
                                         opacity, color_dict[str(color_code)], marker_size, line_width))

    x_vertices_col3 = [vertices[2][0], vertices[6][0]]
    y_vertices_col3 = [vertices[2][1], vertices[6][1]]
    z_vertices_col3 = [vertices[2][2], vertices[6][2]]
    plotly_data.append(build_plotly_data(x_vertices_col3, 
                                         y_vertices_col3, 
                                         z_vertices_col3, 
                                         opacity, color_dict[str(color_code)], marker_size, line_width))

    x_vertices_col4 = [vertices[3][0], vertices[7][0]]
    y_vertices_col4 = [vertices[3][1], vertices[7][1]]
    z_vertices_col4 = [vertices[3][2], vertices[7][2]]
    plotly_data.append(build_plotly_data(x_vertices_col4, 
                                         y_vertices_col4, 
                                         z_vertices_col4, 
                                         opacity, color_dict[str(color_code)], marker_size, line_width))
    
    return plotly_data
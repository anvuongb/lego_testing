import numpy as np
import plotly.graph_objects as go

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
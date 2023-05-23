import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def decode_file_block(f):
    if f.lower() == "3001.dat":
        return [2,4]
    return [-1,-1]

def build_vertices(x, y, z, block_size):
    # v = [[x, y, z],
    #      [x+ block_size[1]*20 , y, z ],
    #      [x+ block_size[1]*20, y, z+ block_size[0]*20],
    #      [x, y, z+ block_size[0]*20], # end upper surface
    #      [x, y+24, z],
    #      [x+ block_size[1]*20, y+24, z ],
    #      [x+ block_size[1]*20, y+24, z+ block_size[0]*20],
    #      [x, y+24, z+ block_size[0]*20]] # end lower surface
    v = [[x-block_size[1]*20/2, y, z-block_size[0]*20/2],
         [x+ block_size[1]*20/2 , y, z-block_size[0]*20/2 ],
         [x+ block_size[1]*20/2, y, z+ block_size[0]*20/2],
         [x-block_size[1]*20/2, y, z+ block_size[0]*20/2], # end upper surface
         
         [x-block_size[1]*20/2, y+24, z-block_size[0]*20/2],
         [x+ block_size[1]*20/2 , y+24, z-block_size[0]*20/2 ],
         [x+ block_size[1]*20/2, y+24, z+ block_size[0]*20/2],
         [x-block_size[1]*20/2, y+24, z+ block_size[0]*20/2]] # end lower surface
    return np.array(v).reshape((8,3))

# load color dict
with open("color_codes.json", "r") as f:
    color_dict = json.load(f)

# ldr_filename = "./ldr_files/dataset/2blocks-perpendicular_15.ldr"
# ldr_filename = "./ldr_files/dataset/wall_augmented270_18.ldr"
# ldr_filename = "./block_fake.ldr"
# ldr_filename = "./2bricks_cross.ldr"
ldr_filename = "./5bricks_rotate.ldr"
print(ldr_filename)

columns = ["line_type", "color_code", "x", "y", "z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "file_block"]
df = pd.read_csv(ldr_filename, sep=" ", names=columns)
df["block_size"] = df["file_block"].apply(lambda x: decode_file_block(x))
# print(df)

translation_matrices = []

# ldraw use x -z y coordinate

for idx, row in df.iterrows():
    tm = np.array([[row.a, row.b, row.c, row.x],
                   [row.d, row.e, row.f, row.y],
                   [row.g, row.h, row.i, row.z],
                   [0, 0, 0, 1]])
    translation_matrices.append(tm)

origin = np.array([0,0,0,1]).reshape(4, 1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(origin[0], origin[1], origin[2])

color_list = ['b', 'g', 'r', 'c', 'm', 'y']
# color_list = ['r']

all_vertices_list = []
for idx, row in df.iterrows():
    tm = np.array([[row.a, row.b, row.c, row.x],
                   [row.d, row.e, row.f, row.y],
                   [row.g, row.h, row.i, row.z],
                   [0, 0, 0, 1]])
    print(tm)
    vertices = build_vertices(0, 0, 0, df.block_size[idx])
    print(vertices.shape, vertices)
    vertices = np.hstack([vertices, np.ones((8,1))])
    vertices = np.matmul(tm, vertices.T)
    print(vertices.shape, vertices)
    vertices = vertices[:3,:].T
    all_vertices_list += list(vertices)
    print(vertices.shape, vertices)
    print('\n\n')
    for id, v in enumerate(vertices):
        ax.scatter(v[0], v[1], v[2], color=color_dict[str(row.color_code)])
        ax.plot([vertices[id][0], vertices[(id+1)%4+4*int(id>=4)][0]],
                [vertices[id][1], vertices[(id+1)%4+4*int(id>=4)][1]],
                [vertices[id][2], vertices[(id+1)%4+4*int(id>=4)][2]],
                color=color_dict[str(row.color_code)])
        if id < 4:
            ax.plot([vertices[id][0], vertices[(id+4)][0]],
                    [vertices[id][1], vertices[(id+4)][1]],
                    [vertices[id][2], vertices[(id+4)][2]],
                    color=color_dict[str(row.color_code)])
ax.legend()

# first pass clean vertices
all_vertices_count = []
for v in all_vertices_list:
    tmp_count = 0
    for k in all_vertices_list:
        if np.array_equal(v, k):
            tmp_count += 1
    all_vertices_count.append(tmp_count)
all_vertices_count = np.array(all_vertices_count)
print(all_vertices_count)
print(all_vertices_count < 2)
all_vertices_set = np.array(all_vertices_list)[all_vertices_count < 2]


# for id, v in enumerate(all_vertices_set):
#     ax.scatter(v[0], v[1], v[2], color=color_list[idx%len(color_list)])

xlim = ax.get_xlim3d()
ylim = ax.get_ylim3d()
zlim = ax.get_zlim3d()
ax.set_xlabel('x-axis', fontsize=20)
ax.set_ylabel('y-axis', fontsize=20)
ax.set_zlabel('z-axis', fontsize=20)
ax.set_box_aspect((xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]))

# while True:
#     for a in range(-50,50):
#         for angle in range(0, 360):
#             ax.view_init(a, angle)
#             plt.draw()
#             plt.pause(.001)

# # ax.view_init(-60, 90)
plt.show()

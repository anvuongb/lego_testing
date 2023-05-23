import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def decode_file_block(f):
    if f.lower() == "3001.dat":
        return [2,4]
    return [-1,-1]

def build_vertices(x, y, z, block_size):
    v = [[x, y, z],
         [x+ block_size[1]*20 , y, z ],
         [x+ block_size[1]*20, y, z+ block_size[0]*20],
         [x, y, z+ block_size[0]*20],
         [x, y+24, z],
         [x+ block_size[1]*20, y+24, z ],
         [x+ block_size[1]*20, y+24, z+ block_size[0]*20],
         [x, y+24, z+ block_size[0]*20]]
    # v = [[x, y, z],
    #      [x, y + block_size[1]*20, z],
    #      [x+ block_size[0]*20, y + block_size[1]*20, z],
    #      [x+ block_size[0]*20, y, z],
    #      [x, y, z+24],
    #      [x, y+ block_size[1]*20, z+24],
    #      [x+ block_size[0]*20, y+ block_size[1]*20, z+24],
    #      [x+ block_size[0]*20, y, z+24]]
    return np.array(v).reshape((8,3))

# ldr_filename = "./ldr_files/dataset/2blocks-perpendicular_13.ldr"
ldr_filename = "./ldr_files/dataset/wall_augmented270_18.ldr"
# ldr_filename = "./block_fake.ldr"
print(ldr_filename)

columns = ["line_type", "color", "x", "y", "z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "file_block"]
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
    # tm = np.array([[row.a, row.d, row.g, 0],
    #                [row.b, row.e, row.h, 0],
    #                [row.c, row.f, row.i, 0],
    #                [row.x, row.y, row.z, 1]])
    translation_matrices.append(tm)

origin = np.array([0,0,0,1]).reshape(4, 1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(origin[0], origin[1], origin[2])

# color_list = ['b', 'g', 'r', 'c', 'm', 'y']
color_list = ['r']
for idx, row in df.iterrows():
    # print(tmp)
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
    print(vertices.shape, vertices)
    print('\n\n')
    for id, v in enumerate(vertices):
        ax.scatter(v[0], v[1], v[2], color=color_list[idx%len(color_list)])
        ax.plot([vertices[id][0], vertices[(id+1)%4+4*int(id>=4)][0]],
                [vertices[id][1], vertices[(id+1)%4+4*int(id>=4)][1]],
                [vertices[id][2], vertices[(id+1)%4+4*int(id>=4)][2]],
                color=color_list[idx%len(color_list)])
        if id < 4:
            ax.plot([vertices[id][0], vertices[(id+4)][0]],
                    [vertices[id][1], vertices[(id+4)][1]],
                    [vertices[id][2], vertices[(id+4)][2]],
                    color=color_list[idx%len(color_list)])
ax.legend()

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

import bpy
import os.path
import imageio
import numpy as np

renders_output_path = "/home/anvuong/Desktop/lego_testing/renders"

blender_filename = "pyramid.blend"

# render from blender
bpy.ops.wm.open_mainfile(filepath=blender_filename)

C = bpy.context
scn = C.scene

bricks_part_tmp = []
for ob in bpy.context.scene.objects:
    print(ob.name)
    if ob.name.endswith(".dat"):
        bricks_part_tmp.append(ob.name)

# get correct order
bricks_part_idx = [int(b.split("_")[0]) for b in bricks_part_tmp]
bricks_part_idx = np.argsort(bricks_part_idx)
bricks_part = [bricks_part_tmp[idx] for idx in bricks_part_idx]

# bricks_part = bricks_part[::-1]
# [print(b) for b in bricks_part]
scn.display.shading.light = 'FLAT'
scn.render.resolution_x = 1000
scn.render.resolution_y = 1000
scn.render.resolution_percentage = 50

for n in bricks_part:
        bpy.context.scene.objects[n].hide_render = True

for n in bricks_part:
    bpy.context.scene.objects[n].hide_render = False
    scn.render.filepath = os.path.join(renders_output_path, "tmp", "{}.png".format(n.split(".")[0]))
    bpy.ops.render.render(write_still=True)

# generate gif
filenames = [os.path.join(renders_output_path, "tmp", "{}.png".format(n.split(".")[0])) for n in bricks_part]
images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave(os.path.join(renders_output_path, blender_filename.split(".")[0]+".gif"), images)
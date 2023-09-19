import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("-n", "--model-name", default="", type=str, help = "model name")
parser.add_argument("-e", "--episode", default=0, type=int, help = "which episode to load from saved models")
parser.add_argument("-l", "--level", default=None, type=int, help = "which level to build, if None, build from 3->10")

# Read arguments from command line
args = parser.parse_args()

model_name = args.model_name
episode = args.episode
level = args.level

print(f"[GENERATE IMAGES] Loading {model_name} from episode {episode}, build with level set to {level}")

if level is not None:
    min_level = level
    max_level = level + 1
else:
    min_level = 3
    max_level = 11

for l in range(min_level,max_level):
    run_str = f"leocad -i /home/anvuong/works/lego_testing/renders/tmp_ldr_render/{model_name}_ep_{episode}_{l}_levels.png -w 500 -h 500 --camera-angles 30 30 /home/anvuong/works/lego_testing/renders/tmp_ldr_files/{model_name}_ep_{episode}_{l}_levels.ldr"
    subprocess.run(run_str, shell=True)

import cv2
import numpy as np

# print("level", level)
if level is None:
    imgs_array = []
    for level in range(min_level,max_level):
        img = cv2.imread(f"/home/anvuong/works/lego_testing/renders/tmp_ldr_render/{model_name}_ep_{episode}_{level}_levels.png")
        imgs_array.append(img)


    vis1 = np.concatenate((imgs_array[0], imgs_array[1], imgs_array[2], imgs_array[3]), axis=1)
    vis2 = np.concatenate((imgs_array[4], imgs_array[5], imgs_array[6], imgs_array[7]), axis=1)
    vis = np.concatenate((vis1, vis2), axis=0)

    # add some text
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (255, 255, 255)
    thickness = 2
    image = cv2.putText(vis, f'Model {model_name} - Episode {episode}', org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    
    cv2.imwrite(f'/home/anvuong/works/lego_testing/renders/{model_name}_ep_{episode}_combined.png', vis)

else:
    vis = cv2.imread(f"/home/anvuong/works/lego_testing/renders/tmp_ldr_render/{model_name}_ep_{episode}_{level}_levels.png")
    # add some text
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (30, 30)
    fontScale = 0.4
    color = (255, 255, 255)
    thickness = 1
    image = cv2.putText(vis, f'Model {model_name} - Episode {episode}', org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    
    cv2.imwrite(f'/home/anvuong/works/lego_testing/renders/{model_name}_ep_{episode}_combined.png', vis)
import os
import subprocess

for level in range(3,11):
    run_str = f"leocad -i /home/anvuong/Desktop/lego_testing/training_images/training_v2_{level}_levels.png -w 500 -h 500 --camera-angles 30 30 test_v2_new_{level}_levels.ldr"
    subprocess.run(run_str, shell=True)

import cv2
import numpy as np

imgs_array = []
for level in range(3,11):
    img = cv2.imread(f"/home/anvuong/Desktop/lego_testing/training_images/training_v2_{level}_levels.png")
    imgs_array.append(img)

vis1 = np.concatenate((imgs_array[0], imgs_array[1], imgs_array[2], imgs_array[3]), axis=1)
vis2 = np.concatenate((imgs_array[4], imgs_array[5], imgs_array[6], imgs_array[7]), axis=1)
vis = np.concatenate((vis1, vis2), axis=0)

cv2.imwrite('/home/anvuong/Desktop/lego_testing/training_images/combined.png', vis)
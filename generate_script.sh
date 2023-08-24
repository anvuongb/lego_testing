#!/bin/bash
for i in {1..100}
do
   /home/anvuong/miniconda3/envs/python310/bin/python rand_tower_pyramid.py pyramids $i 5
   /home/anvuong/miniconda3/envs/python310/bin/python rand_tower_pyramid.py pyramids $i 10
   /home/anvuong/miniconda3/envs/python310/bin/python rand_tower_pyramid.py pyramids $i 15
done
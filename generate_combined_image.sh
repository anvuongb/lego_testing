#!/bin/bash

python generate_ldr_replay_v2.py --model-name $1 --episode $2
python generate_images.py --model-name $1 --episode $2
from simple_env_v2 import SimpleLegoEnv, mask_fn
from sb3_contrib.ppo_mask import MaskablePPO
from lego import LegoModel
import torch as th
from stud_control import get_all_possible_placements
from simple_env import ACTIONS_MAP, ALL_ACTIONS

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import gymnasium as gym
import numpy as np

model = MaskablePPO.load("./models/1693378319/700000.zip")

print(model.policy)

env = SimpleLegoEnv()
obs, _ = env.reset()

for i in range(1000):
    valid_action_array = mask_fn(env)
    action, _states = model.predict(obs, action_masks=valid_action_array, deterministic=True)
    action = action.ravel()[0]
    obs, rewards, terminated, _, _ = env.step(action)
    if terminated:
        break
    # else:
    #     s = f'''/usr/bin/leocad -i /home/anvuong/Desktop/lego_testing/renders/image_{i}.png -w 400 -h 400 --camera-angles 30 30 test.ldr'''
    #     subprocess.run(s, shell=True)

if len(env.bricks_list) > 1:
    model = LegoModel(brick=env.bricks_list[0])
    for brick in env.bricks_list[1:]:
        model.add_brick(brick)
    model.generate_ldr_file("test_replay_v2.ldr")
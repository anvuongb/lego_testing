from simple_env_v2 import SimpleLegoEnv
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

def mask_fn(env: gym.Env) -> np.ndarray:
    placements_list = get_all_possible_placements(env.occupancy_mat_list[env.current_layer_idx], env.base_brick_stud_mat, mode="valid", collide_type="brick")
    if env.current_layer_idx > 0:
        # allow placement over holes if possible
        placements_partial_list = get_all_possible_placements(1 - env.occupancy_mat_list[env.current_layer_idx-1], env.base_brick_stud_mat, mode="valid", collide_type="hole")
        placements_list = list(set(placements_partial_list).intersection(placements_list))
    masked_actions_dict = {}

    for i in range(len(ALL_ACTIONS)):
        masked_actions_dict[i] = False

    # check if moveup allowed
    if env.bricks_per_level[env.current_layer_idx] > 0:
        masked_actions_dict[len(ALL_ACTIONS)-1] = True

    # check if other actions allowed:
    for k, v in ACTIONS_MAP.items():
        if v in placements_list:
            masked_actions_dict[k] = True

    masked_actions_list = [masked_actions_dict[i] for i in range(len(ALL_ACTIONS))]
    return np.array(masked_actions_list)

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
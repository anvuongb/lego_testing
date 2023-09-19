from simple_env_v3 import SimpleLegoEnv, mask_fn
from sb3_contrib.ppo_mask import MaskablePPO
from lego import LegoModel

from sb3_contrib.ppo_mask import MaskablePPO

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

print(f"[GENERATE LDR] Loading {model_name} from episode {episode}, build with level set to {level}")

model = MaskablePPO.load(f"./models/{model_name}/episode_{episode}.zip")

if level is not None:
    min_level = level
    max_level = level + 1
else:
    min_level = 3
    max_level = 11

# print(model.policy)

env = SimpleLegoEnv()

for l in range(min_level, max_level):
    env.pyramid_levels = l
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
        lego_model = LegoModel(brick=env.bricks_list[0])
        for brick in env.bricks_list[1:]:
            lego_model.add_brick(brick)
        lego_model.generate_ldr_file(f"./renders/tmp_ldr_files/{model_name}_ep_{episode}_{l}_levels.ldr")
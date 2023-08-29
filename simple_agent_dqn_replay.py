from simple_env import SimpleLegoEnv
from stable_baselines3 import DQN
import subprocess
import os
import time
import torch as th

model = DQN.load("./models/1693292546/3730000.zip")

print(model.policy)

env = SimpleLegoEnv()
obs, _ = env.reset()

for i in range(1000):
    action, _states = model.predict(obs, deterministic=False)
    action=action.ravel()[0]
    obs, rewards, terminated, _, _ = env.step(action)
    print(action, rewards)
    if terminated:
        break
    # else:
    #     s = f'''/usr/bin/leocad -i /home/anvuong/Desktop/lego_testing/renders/image_{i}.png -w 400 -h 400 --camera-angles 30 30 test.ldr'''
    #     subprocess.run(s, shell=True)
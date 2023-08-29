from simple_env import SimpleLegoEnv
from stable_baselines3 import DQN
import os
import time
import torch as th


env = SimpleLegoEnv()
env.reset()

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[512, 512, 256, 256, 128, 128])
model = DQN('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=logdir)
print(model.policy)

TIMESTEPS = 10000
iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"DQN")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")
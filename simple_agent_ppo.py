from simple_env import SimpleLegoEnv
from stable_baselines3 import PPO
import os
import time
import torch as th

# Simple test
# env = SimpleLegoEnv()
# episodes = 50

# for episode in range(episodes):
# 	print("\n\n ep", episode)
# 	terminated = False
# 	obs = env.reset()
# 	while not terminated:
# 		random_action = env.action_space.sample()
# 		print("action",random_action)
# 		obs, reward, terminated, truncated, info = env.step(random_action)
# 		print('reward',reward)

env = SimpleLegoEnv()
env.reset()

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[512, 512, 256, 256, 128, 128])
model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=logdir)
print(model.policy)

TIMESTEPS = 10000
iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")
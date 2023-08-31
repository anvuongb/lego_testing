import gymnasium as gym
import numpy as np
import time
import torch as th
from simple_env_v2 import SimpleLegoEnv, mask_fn
from stud_control import get_all_possible_placements

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO


env = SimpleLegoEnv()
env.reset()
env = ActionMasker(env, mask_fn)  # Wrap to enable masking

# MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
# with ActionMasker. If the wrapper is detected, the masks are automatically
# retrieved and used when learning. Note that MaskablePPO does not accept
# a new action_mask_fn kwarg, as it did in an earlier draft.
models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[1024, 1024, 512, 512, 256, 256, 256, 128, 128, 128])

model = MaskablePPO(MaskableActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=logdir)
print(model.policy)

TIMESTEPS = 100000
iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"MaskedPPO")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")


# # Note that use of masks is manual and optional outside of learning,
# # so masking can be "removed" at testing time
# model.predict(observation, action_masks=valid_action_array)
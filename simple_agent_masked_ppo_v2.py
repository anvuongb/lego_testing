import gymnasium as gym
import numpy as np
import time
import torch as th
from simple_env_v2 import SimpleLegoEnv
from stud_control import get_all_possible_placements
from simple_env import ACTIONS_MAP, ALL_ACTIONS

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO


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
                     net_arch=[512, 512, 256, 256, 128, 128])

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
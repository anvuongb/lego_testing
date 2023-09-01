import gymnasium as gym
import numpy as np
import time
import torch as th
from simple_env_v2 import SimpleLegoEnv, mask_fn
from stud_control import get_all_possible_placements

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

import argparse
 
if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Adding optional argument
    parser.add_argument("-l", "--load", default=None, type=str, help = "load from a pretrained model")
    parser.add_argument("-n", "--name", default=None, type=str, help = "name the model, default to timestamp")
    parser.add_argument("-r", "--rand-levels", default=False, type=bool, help = "if True, pyramid levels changes every episode")
    parser.add_argument("-le", "--levels", default=4, type=int, help = "default high, ignore if rand_levels is true")
    
    # Read arguments from command line
    args = parser.parse_args()

    env = SimpleLegoEnv(pyramid_levels=args.levels, rand_levels=args.rand_levels)
    print(f"rand_levels is set to {env.rand_levels}, pyramid levels is set to {env.pyramid_levels}")
    env.reset()
    env = ActionMasker(env, mask_fn)  # Wrap to enable masking

    # MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
    # with ActionMasker. If the wrapper is detected, the masks are automatically
    # retrieved and used when learning. Note that MaskablePPO does not accept
    # a new action_mask_fn kwarg, as it did in an earlier draft.
    if args.name is not None:
        models_dir = f"models/{args.name}/"
        logdir = f"logs/{args.name}/"
    else:
        models_dir = f"models/{int(time.time())}/"
        logdir = f"logs/{int(time.time())}/"

    print(f"model output is set to {models_dir}, logs are set to {logdir}")

    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                        net_arch=[1024, 1024, 512, 512, 256, 256, 256, 128, 128, 128])

    model = MaskablePPO(MaskableActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=logdir)
    if args.load is not None:
        print(f"loading model from {args.load}")
        model.load(args.load)
    print(model.policy)

    TIMESTEPS = 10000 
    iters = 0
    # iters = 90
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"MaskedPPO")
        if iters % 10 == 0:
            model.save(f"{models_dir}/episode_{iters}")


    # # Note that use of masks is manual and optional outside of learning,
    # # so masking can be "removed" at testing time
    # model.predict(observation, action_masks=valid_action_array)
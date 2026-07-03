import gymnasium as gym
import numpy as np
from simple_env_v4 import SimpleLegoEnv, LegoSetConfig, mask_fn, _target_from_ldr, _generate_pyramid_target
from lego import LegoModel
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

import argparse
import os


def load_and_generate(model_path, levels=4, grid=10, output_dir=".",
                      target_ldr=None, free_form=False, deterministic=True):
    config = LegoSetConfig(
        grid_height=grid,
        grid_width=grid,
        max_levels=levels + 2,
        brick_type_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        structural_support_ratio=0.5,
        rand_levels=False,
        min_levels=levels,
        max_levels_config=levels,
        use_budget=False,
        target_path=target_ldr,
        free_form=free_form,
    )

    raw_env = SimpleLegoEnv(config)
    obs, _ = raw_env.reset()
    env = ActionMasker(raw_env, mask_fn)
    _ = env  # keep reference

    model = MaskablePPO.load(model_path)

    os.makedirs(output_dir, exist_ok=True)

    for step in range(1000):
        raw_env = env.unwrapped if hasattr(env, 'unwrapped') else env.env if hasattr(env, 'env') else env
        valid_action_array = mask_fn(raw_env)
        action, _states = model.predict(obs, action_masks=valid_action_array, deterministic=deterministic)
        action = action.ravel()[0]
        obs, rewards, terminated, _, info = env.step(action)
        if terminated:
            break

    bricks_list = raw_env.bricks_list if hasattr(raw_env, 'bricks_list') else env.unwrapped.bricks_list
    if len(bricks_list) > 1:
        lego_model = LegoModel(brick=bricks_list[0])
        for brick in bricks_list[1:]:
            lego_model.add_brick(brick)

        output_path = os.path.join(output_dir, f"gen_{levels}_levels.ldr")
        lego_model.generate_ldr_file(output_path)
        print(f"Generated: {output_path}")
        print(f"Bricks: {len(bricks_list)}")
        print(f"Total reward: {rewards:.2f}")
        return True
    else:
        print("No bricks generated")
        return False


def generate_all_levels(model_path, min_levels=3, max_levels=11, grid=20, output_dir="renders/tmp_ldr_files"):
    for l in range(min_levels, max_levels):
        print(f"\nGenerating {l}-level pyramid...")
        load_and_generate(model_path, levels=l, grid=grid, output_dir=output_dir)


def generate_from_target_ldr(model_path, target_ldr_path, grid=20, output_dir="renders/generated"):
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(target_ldr_path))[0]

    # Load target to determine levels
    target_voxels = _target_from_ldr(target_ldr_path, grid, grid, 20)
    levels = 0
    for l in range(20):
        if np.any(target_voxels[l] > 0):
            levels = l + 1

    success = load_and_generate(
        model_path, levels=levels, grid=grid,
        output_dir=output_dir, target_ldr=target_ldr_path
    )
    if success:
        print(f"Generated from target '{basename}' at {output_dir}")
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str, help="path to trained model zip")
    parser.add_argument("-l", "--levels", default=4, type=int, help="number of pyramid levels")
    parser.add_argument("-g", "--grid", default=10, type=int, help="grid size")
    parser.add_argument("-o", "--output-dir", default="renders/tmp_ldr_files", type=str)
    parser.add_argument("--all-levels", action="store_true", help="generate from 3 to 10 levels")
    parser.add_argument("--target-ldr", default=None, type=str, help="generate to match this .ldr target")
    parser.add_argument("--free-form", action="store_true", help="free-form generation (no target)")
    parser.add_argument("--stochastic", action="store_true", help="use stochastic actions")

    args = parser.parse_args()

    if args.all_levels:
        generate_all_levels(args.model, grid=args.grid, output_dir=args.output_dir)
    elif args.target_ldr:
        generate_from_target_ldr(args.model, args.target_ldr, grid=args.grid, output_dir=args.output_dir)
    else:
        load_and_generate(
            args.model, levels=args.levels, grid=args.grid,
            output_dir=args.output_dir, free_form=args.free_form,
            deterministic=not args.stochastic
        )

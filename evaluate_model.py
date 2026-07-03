import numpy as np
from simple_env_v4 import (
    SimpleLegoEnv, LegoSetConfig, mask_fn,
    _target_from_ldr, _generate_pyramid_target,
    _check_support, _brick_footprint,
    BRICK_TYPE_MAP
)
from lego import LegoModel
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

import argparse
import os
import json


def compute_metrics(env):
    occ = np.array(env.occupancy_mat_list)
    target = np.array(env.target_stud_mat_list)

    # Shape accuracy (IoU per layer)
    iou_per_layer = []
    for l in range(env.pyramid_levels):
        target_sum = np.sum(target[l])
        if target_sum == 0:
            continue
        intersection = np.sum(np.logical_and(occ[l] > 0, target[l] > 0))
        union = np.sum(np.logical_or(occ[l] > 0, target[l] > 0))
        iou = intersection / union if union > 0 else 0.0
        iou_per_layer.append(iou)
    mean_iou = np.mean(iou_per_layer) if iou_per_layer else 0.0

    # Structural score
    structural_scores = []
    for brick in env.bricks_list:
        bs = brick.block_size
        if hasattr(brick, 'center_y_origin'):
            layer = int(round(abs(brick.center_y_origin) / 24))
            cx, cz = brick.center_x, brick.center_z
            half_w = bs[1] * 20 / 2
            half_h = bs[0] * 20 / 2
            x = int(round((cx - half_w) / 20))
            z = int(round(env.grid_height - (cz + half_h) / 20))
            if _check_support(env.occ_grid_3d, layer, bs[0], bs[1], x, z, 0.5):
                structural_scores.append(1.0)
            else:
                structural_scores.append(0.0)
    structural_score = np.mean(structural_scores) if structural_scores else 0.0

    # Brick diversity
    brick_types_used = set()
    for brick in env.bricks_list:
        brick_types_used.add(brick.block_type)
    max_types = min(len(env.brick_type_indices), 10)
    diversity_score = len(brick_types_used) / max_types if max_types > 0 else 0.0

    # Efficiency (studs per brick)
    total_studs = sum(b[0] * b[1] for b in [BRICK_TYPE_MAP[idx][0] for idx in env.brick_type_indices])
    efficiency = env.total_studs_placed / max(1, len(env.bricks_list)) / 4.0

    # Layer fill rate
    fill_rates = []
    for l in range(env.pyramid_levels):
        target_sum = np.sum(target[l])
        if target_sum > 0:
            fill_rates.append(np.sum(occ[l]) / target_sum)
    avg_fill_rate = np.mean(fill_rates) if fill_rates else 0.0

    return {
        "mean_iou": float(mean_iou),
        "structural_score": float(structural_score),
        "diversity_score": float(diversity_score),
        "efficiency_ratio": float(efficiency),
        "avg_fill_rate": float(avg_fill_rate),
        "total_bricks": len(env.bricks_list),
        "brick_types_used": list(brick_types_used),
        "layers_completed": sum(
            1 for l in range(env.pyramid_levels)
            if np.sum(target[l]) > 0 and np.all(np.equal(occ[l], target[l]))
        ),
    }


def evaluate_model(model_path, num_episodes=10, levels=4, grid=10, verbose=True):
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
    )

    model = MaskablePPO.load(model_path)
    all_metrics = []

    for ep in range(num_episodes):
        raw_env = SimpleLegoEnv(config)
        obs, _ = raw_env.reset()
        env = ActionMasker(raw_env, mask_fn)

        terminated = False
        for _ in range(1000):
            raw_env = env.unwrapped if hasattr(env, 'unwrapped') else env.env
            valid_action_array = mask_fn(raw_env)
            action, _states = model.predict(obs, action_masks=valid_action_array, deterministic=True)
            action = action.ravel()[0]
            obs, reward, terminated, _, _ = env.step(action)
            if terminated:
                break

        raw_env = env.unwrapped if hasattr(env, 'unwrapped') else env.env
        metrics = compute_metrics(raw_env)
        metrics["episode_reward"] = float(reward)
        all_metrics.append(metrics)

        if verbose:
            print(f"Ep {ep + 1}: IoU={metrics['mean_iou']:.3f}, "
                  f"Structural={metrics['structural_score']:.3f}, "
                  f"Diversity={metrics['diversity_score']:.3f}, "
                  f"Bricks={metrics['total_bricks']}, "
                  f"Reward={metrics['episode_reward']:.1f}")

    avg_metrics = {}
    for key in all_metrics[0]:
        if isinstance(all_metrics[0][key], (int, float)):
            avg_metrics[f"avg_{key}"] = float(np.mean([m[key] for m in all_metrics]))
            avg_metrics[f"std_{key}"] = float(np.std([m[key] for m in all_metrics]))

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Evaluation Results ({num_episodes} episodes)")
        print(f"{'=' * 50}")
        for key, val in avg_metrics.items():
            print(f"  {key}: {val:.4f}")

    return avg_metrics, all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str, help="path to trained model zip")
    parser.add_argument("-e", "--episodes", default=10, type=int, help="number of evaluation episodes")
    parser.add_argument("-l", "--levels", default=4, type=int, help="pyramid levels")
    parser.add_argument("-g", "--grid", default=10, type=int, help="grid size")
    parser.add_argument("-o", "--output", default=None, type=str, help="save metrics to JSON file")

    args = parser.parse_args()

    avg_metrics, all_metrics = evaluate_model(
        args.model, num_episodes=args.episodes,
        levels=args.levels, grid=args.grid
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"average": avg_metrics, "per_episode": all_metrics}, f, indent=2)
        print(f"Metrics saved to {args.output}")

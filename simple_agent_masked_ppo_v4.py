import gymnasium as gym
import numpy as np
import time
import torch as th
import torch.nn as nn
from torch.optim import Adam
from simple_env_v4 import SimpleLegoEnv, LegoSetConfig, mask_fn, BRICK_TYPE_MAP

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import argparse
import os


CURRICULUM_STAGES = [
    {"levels": 3, "brick_types": [0, 7, 8], "grid": 8,  "total_timesteps": 200_000,  "budget": 10},
    {"levels": 4, "brick_types": [0, 1, 7, 8], "grid": 10, "total_timesteps": 300_000,  "budget": 12},
    {"levels": 5, "brick_types": [0, 1, 3, 7, 8], "grid": 12, "total_timesteps": 500_000,  "budget": 15},
    {"levels": 6, "brick_types": [0, 1, 2, 3, 4, 7, 8], "grid": 14, "total_timesteps": 1_000_000, "budget": 18},
    {"levels": 8, "brick_types": [0, 1, 2, 3, 4, 5, 6, 7, 8], "grid": 16, "total_timesteps": 2_000_000, "budget": 20},
    {"levels": 10,"brick_types": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "grid": 20, "total_timesteps": 5_000_000, "budget": 25},
]


class LegoCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int,
                 grid_h: int = 20, grid_w: int = 20, max_levels: int = 10, n_brick_types: int = 10):
        super().__init__(observation_space, features_dim)

        self.grid_h = grid_h
        self.grid_w = grid_w
        self.max_levels = max_levels
        self.n_brick_types = n_brick_types
        self.grid_channels = 2 * max_levels
        scalar_size = 1 + n_brick_types

        self.cnn = nn.Sequential(
            nn.Conv2d(self.grid_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.zeros(1, self.grid_channels, grid_h, grid_w)
            cnn_out = self.cnn(sample)
            cnn_dim = cnn_out.shape[1]

        self.linear = nn.Sequential(
            nn.Linear(cnn_dim + scalar_size, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        grid_flat_size = self.grid_channels * self.grid_h * self.grid_w
        grid = observations[:, :grid_flat_size]
        scalars = observations[:, grid_flat_size:]
        batch_size = observations.shape[0]
        grid = grid.reshape(batch_size, self.grid_channels, self.grid_h, self.grid_w)
        cnn_features = self.cnn(grid)
        combined = th.cat([cnn_features, scalars], dim=1)
        return self.linear(combined)


def make_env(config, seed=0):
    env = SimpleLegoEnv(config)
    env.reset(seed=seed)
    env = ActionMasker(env, mask_fn)
    return env


class CurriculumTrainer:
    def __init__(self, stages=None, models_dir=None, logdir=None, device="auto"):
        self.stages = stages or CURRICULUM_STAGES
        self.models_dir = models_dir or f"models/v4_{int(time.time())}/"
        self.logdir = logdir or f"logs/v4_{int(time.time())}/"
        self.device = device
        self.model = None
        self.current_stage = 0
        self.max_levels = max(s["levels"] for s in self.stages) + 2

    def _create_config(self, stage):
        grid = stage["grid"]
        return LegoSetConfig(
            grid_height=grid,
            grid_width=grid,
            max_levels=self.max_levels,
            brick_type_indices=stage["brick_types"],
            structural_support_ratio=0.5,
            rand_levels=False,
            min_levels=stage["levels"],
            max_levels_config=stage["levels"],
            use_budget=True,
            budget_per_type=stage.get("budget", 20),
            reward_weights={
                "shape": 1.0,
                "structural": 0.5,
                "efficiency": 0.1,
                "diversity": 0.2,
                "layer_complete": 5.0,
                "model_complete": 10.0,
                "invalid_placement": -2.0,
            },
            free_form=False,
            target_path=None,
        )

    def _create_policy_kwargs(self, config):
        max_levels = self.max_levels
        n_types = len(config.brick_type_indices)
        feat_dim = 256
        return {
            "features_extractor_class": LegoCNNFeatureExtractor,
            "features_extractor_kwargs": {
                "features_dim": feat_dim,
                "grid_h": config.grid_height,
                "grid_w": config.grid_width,
                "max_levels": max_levels,
                "n_brick_types": n_types,
            },
            "net_arch": [512, 512, feat_dim],
            "activation_fn": th.nn.ReLU,
        }

    def train(self):
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logdir, exist_ok=True)

        for stage_idx, stage in enumerate(self.stages):
            print(f"\n{'='*60}")
            print(f"Curriculum Stage {stage_idx + 1}/{len(self.stages)}")
            print(f"  Levels: {stage['levels']}, Grid: {stage['grid']}x{stage['grid']}")
            print(f"  Brick types: {stage['brick_types']}")
            print(f"  Timesteps: {stage['total_timesteps']:,}")
            print(f"{'='*60}\n")

            config = self._create_config(stage)
            env = make_env(config)

            if self.model is None:
                policy_kwargs = self._create_policy_kwargs(config)
                self.model = MaskablePPO(
                    MaskableActorCriticPolicy,
                    env,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    tensorboard_log=self.logdir,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.01,
                    max_grad_norm=0.5,
                )
            else:
                self.model.env = env

            self.model.learn(
                total_timesteps=stage["total_timesteps"],
                reset_num_timesteps=(stage_idx == 0),
                tb_log_name=f"stage_{stage_idx + 1}",
            )

            save_path = os.path.join(self.models_dir, f"stage_{stage_idx + 1}_levels_{stage['levels']}")
            self.model.save(save_path)
            print(f"Model saved to {save_path}")

        print("\nCurriculum training complete!")
        final_path = os.path.join(self.models_dir, "final_model")
        self.model.save(final_path)
        print(f"Final model saved to {final_path}")
        return self.model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", default=None, type=str, help="model name")
    parser.add_argument("-l", "--load", default=None, type=str, help="load existing model")
    parser.add_argument("--stages", default=None, type=str, help="comma-separated list of levels to train (e.g. 3,4,5,6,8,10)")
    parser.add_argument("--fast", action="store_true", help="run a fast test with minimal stages")

    args = parser.parse_args()

    if args.fast:
        stages = [
            {"levels": 3, "brick_types": [0, 7, 8], "grid": 8,  "total_timesteps": 10_000, "budget": 10},
            {"levels": 4, "brick_types": [0, 1, 7, 8], "grid": 10, "total_timesteps": 10_000, "budget": 12},
        ]
    elif args.stages:
        level_list = [int(x.strip()) for x in args.stages.split(",")]
        stages = []
        for i, l in enumerate(level_list):
            grid = l * 2
            types = list(range(min(10, 3 + i * 2)))
            stages.append({
                "levels": l,
                "brick_types": types,
                "grid": max(8, grid),
                "total_timesteps": 200_000 * (i + 1),
                "budget": min(25, 8 + i * 3),
            })
    else:
        stages = CURRICULUM_STAGES

    if args.name is not None:
        models_dir = f"models/{args.name}/"
        logdir = f"logs/{args.name}/"
    else:
        ts = int(time.time())
        models_dir = f"models/v4_{ts}/"
        logdir = f"logs/v4_{ts}/"

    trainer = CurriculumTrainer(stages=stages, models_dir=models_dir, logdir=logdir)

    if args.load:
        print(f"Loading existing model from {args.load}")
        from sb3_contrib.ppo_mask import MaskablePPO
        trainer.model = MaskablePPO.load(args.load)

    trainer.train()

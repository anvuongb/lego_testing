# Changes: RL-Based Lego Set Generation

## Overview

This document details the implementation of a reinforcement learning system for generating meaningful Lego sets. The work extends the existing codebase with a new environment (v4), a CNN-based PPO agent with curriculum learning, and evaluation infrastructure.

---

## New Files

### `simple_env_v4.py` — v4 Environment (Phase 1)

A completely rewritten Gymnasium environment addressing the limitations of v2/v3:

#### Key Improvements

| Feature | v2/v3 | v4 |
|---------|-------|-----|
| Target format | 2D per-layer matrices | 3D voxel grid (levels × H × W) |
| Target source | Pyramids only | `.ldr` files, programmatic pyramids, free-form |
| Collision detection | 2D (per-layer occupancy) | **3D occupancy grid** with per-voxel checks |
| Structural support | Not checked | **Support ratio check** (≥50% studs must be supported from below) |
| Reward | ±1 binary per brick | **Multi-objective**: shape + structural + efficiency + diversity + completion |
| Budget | Optional per-type cap | Same, but enforced in mask + checked in step |
| Action masking | Manual mask function | Same pattern, **enhanced with structural filter** |
| Brick types | 7 types | **10 types** (added 1×1, 1×2, 1×3 plates) |

#### Multi-Objective Reward Components

```
shape_reward       — Overlap between placed brick and target voxels
structural_reward  — Bonus for ≥75% support, penalty for <50%
efficiency_reward  — Larger bricks get bonus (area / 4)
diversity_reward   — Avoids overusing a single brick type
layer_complete     — +5 when layer matches target exactly
model_complete     — +10 when all layers match target
invalid_placement  — -2 for collision, no support, or out-of-budget
```

#### Target Loading

- `_target_from_ldr(filepath, ...)` — Converts any `.ldr` file into a 3D voxel target grid with automatic coordinate offset correction
- `_generate_pyramid_target(levels, ...)` — Generates pyramid targets (backward compatible with v2/v3)
- `free_form=True` mode — No target shape; agent builds freely with only structural constraints

#### Mask Function

The `mask_fn()` in v4 adds an additional filter beyond v3:
1. Checks available placements on current layer occupancy (no collision)
2. Checks placements are within action bounds
3. Checks placements overlap with support from layer below
4. Filters by **structural support** (≥50% studs supported)
5. Enables `moveup` only when at least one brick placed on current layer

---

### `simple_agent_masked_ppo_v4.py` — Masked PPO Agent with CNN and Curriculum (Phase 2)

#### Custom CNN Feature Extractor (`LegoCNNFeatureExtractor`)

```python
Input:  flat observation vector
  ├─ Grid portion → reshape to (2×max_levels, H, W) → 3× Conv2d → AdaptiveAvgPool2d(4,4) → flatten
  ├─ Scalar portion → (pyramid_levels, budget_per_type...)
  └─ Concatenate → Linear(1034, 256) → ReLU → features
```

Architecture:
- `Conv2d(2*max_levels, 32, 3)` → ReLU
- `Conv2d(32, 64, 3)` → ReLU
- `Conv2d(64, 64, 3)` → ReLU
- `AdaptiveAvgPool2d(4, 4)` → Flatten
- `Linear(64*16 + scalars, 256)` → ReLU

#### Curriculum Learning

Six stages that progressively increase difficulty:

| Stage | Levels | Grid | Brick Types | Timesteps | Budget/Type |
|-------|--------|------|-------------|-----------|-------------|
| 1 | 3 | 8×8 | 0, 7, 8 (2×2, 1×2, 1×1) | 200K | 10 |
| 2 | 4 | 10×10 | +1 (2×4) | 300K | 12 |
| 3 | 5 | 12×12 | +3 (2×3) | 500K | 15 |
| 4 | 6 | 14×14 | +2, 4 (2×4 rotated, 2×3 rotated) | 1M | 18 |
| 5 | 8 | 16×16 | +5, 6 (2×6, 2×6 rotated) | 2M | 20 |
| 6 | 10 | 20×20 | +9 (1×3) | 5M | 25 |

The curriculum can be customized with `--stages` flag.

#### Training Configuration

```python
MaskablePPO(
    policy_kwargs = {
        net_arch: [512, 512, 256],
        features_extractor_class: LegoCNNFeatureExtractor,
        activation_fn: ReLU,
    },
    learning_rate = 3e-4,
    n_steps = 2048,
    batch_size = 64,
    n_epochs = 10,
    gamma = 0.99,
    gae_lambda = 0.95,
    clip_range = 0.2,
    ent_coef = 0.01,
)
```

---

### `generate_ldr_replay_v4.py` — Replay and Generation (Phase 3)

Loads a trained model and generates `.ldr` files:

- **Pyramid generation**: `--all-levels` generates from 3 to 10 levels
- **Target reconstruction**: `--target-ldr path.ldr` loads a target Lego model and attempts to reconstruct it
- **Free-form generation**: `--free-form` generates without target constraints
- **Stochastic mode**: `--stochastic` uses non-deterministic actions for variety

---

### `evaluate_model.py` — Evaluation Pipeline (Phase 4)

Comprehensive metrics computed per episode:

| Metric | Description |
|--------|-------------|
| `mean_iou` | Intersection-over-Union averaged across layers |
| `structural_score` | % of bricks with ≥50% stud support from below |
| `diversity_score` | Unique brick types used / max available |
| `efficiency_ratio` | Avg studs per brick / 4 (baseline) |
| `avg_fill_rate` | Avg occupancy / target ratio across layers |
| `total_bricks` | Total bricks placed |
| `brick_types_used` | Set of brick type names used |
| `layers_completed` | Layers matching target exactly |

Outputs JSON with per-episode and averaged metrics via `--output`.

---

## Modified Files

### `pyproject.toml`

Added required dependencies: `gymnasium`, `tensorboard`, `torch`, `sb3-contrib`

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   CurriculumTrainer                      │
│  Stage 1 → Stage 2 → ... → Stage 6                      │
│         │                                                │
│         ▼                                                │
│  ┌──────────────────┐      ┌────────────────────────┐   │
│  │  SimpleLegoEnv    │─────▶│   ActionMasker         │   │
│  │  (v4)             │      │   (mask_fn filter)     │   │
│  └────────┬─────────┘      └───────────┬────────────┘   │
│           │                            │                │
│           │ observation                │ action mask    │
│           ▼                            ▼                │
│  ┌─────────────────────────────────────────────────┐   │
│  │          MaskablePPO + MaskableActorCriticPolicy │   │
│  │  ┌─────────────────┐  ┌──────────────────────┐  │   │
│  │  │ LegoCNNFeature   │  │     MlpExtractor     │  │   │
│  │  │ Extractor        │──▶│  [512, 512, 256]    │  │   │
│  │  │ (Conv2d ×3)      │  │  ┌──────┐ ┌───────┐ │  │   │
│  │  │ → AdaptivePool   │  │  │Actor │ │Critic │ │  │   │
│  │  │ → Linear(1034,   │  │  └──────┘ └───────┘ │  │   │
│  │  │         256)     │  └──────────────────────┘  │   │
│  │  └─────────────────┘                             │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## How to Use

```bash
# Install dependencies
.venv/bin/python -m pip install -e .

# Train with full curriculum
./train_v4.sh my_experiment

# Train from specific stage
.venv/bin/python simple_agent_masked_ppo_v4.py \
    --name my_model \
    --stages "3,4,5,6,8,10"

# Fast test (2 stages, minimal timesteps)
.venv/bin/python simple_agent_masked_ppo_v4.py --fast

# Generate .ldr from trained model
.venv/bin/python generate_ldr_replay_v4.py \
    -m models/my_model/final_model.zip \
    -l 4 \
    -o renders/my_generations

# Generate matching a target .ldr file
.venv/bin/python generate_ldr_replay_v4.py \
    -m models/my_model/final_model.zip \
    --target-ldr my_design.ldr

# Evaluate
.venv/bin/python evaluate_model.py \
    -m models/my_model/final_model.zip \
    -e 20 -l 4 -g 10 \
    -o metrics/my_results.json
```

## Known Limitations

1. **LDR target loading** — Works best with grid-aligned models. Rotated bricks may not map correctly to the voxel grid.
2. **3D collision edge cases** — The 3D grid check uses coarse voxel mapping; very small positioning errors near layer boundaries could theoretically pass the check.
3. **Color is not learned** — Colors are auto-assigned from the brick type's base color; the action space doesn't include color selection.
4. **`sb3-contrib` dependency** — `MaskablePPO` requires the `sb3-contrib` package, not just base `stable-baselines3`.

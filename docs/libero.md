# LIBERO Configuration

Overrides for `base.yaml`. LIBERO uses **delta EEF pose** actions (7-D) and joint-position proprioception (9-D).

## YAML overrides (libero.yaml)

```yaml
defaults:
  - base

benchmark: libero

# ── Dimensions ───────────────────────────────────────────────
model:
  D_a: 7
  D_q: 9      # 7 joint positions [rad] + 2 gripper finger offsets [cm]
  D_c: 201    # 128 + 9 + 64

# ── Action space ─────────────────────────────────────────────
action:
  type: delta_eef
  # dims 0-5: Δxyz + Δeuler (delta EEF pose), normalized with Z-score
  # dim  6:   binary gripper {-1=open, +1=close}, NO normalization
  dims_to_normalize: [0, 1, 2, 3, 4, 5]
  normalize_method: zscore        # compute mean/std from training split
  gripper_dim: 6
  gripper_passthrough: true       # keep raw {-1, +1}

# ── Proprioception ────────────────────────────────────────────
proprio:
  # dims 0-6: joint positions [rad] → Z-score normalize
  joint_dims: [0, 1, 2, 3, 4, 5, 6]
  joint_normalize: zscore
  # dims 7-8: gripper finger offsets [cm] → min-max → [-1, 1]
  finger_dims: [7, 8]
  finger_normalize: minmax
  finger_output_range: [-1.0, 1.0]

# ── Images ───────────────────────────────────────────────────
images:
  camera_keys:
    - agentview_rgb          # third-person, (256, 256, 3)
    - eye_in_hand_rgb        # wrist, (256, 256, 3)
  target_size: [224, 224]
  normalize: clip            # ÷255 → CLIP mean/std
  # CLIP ImageNet stats:
  clip_mean: [0.48145466, 0.4578275, 0.40821073]
  clip_std:  [0.26862954, 0.26130258, 0.27577711]

# ── Dataset ───────────────────────────────────────────────────
data:
  format: hdf5
  suites: [libero_spatial, libero_object, libero_goal, libero_long]
  # Total: 1,693 actual episodes (not 2,000 nominal; 307 filtered)
  total_episodes: 1693
  total_frames: 273465
  fps: 10
  stats_file: data/libero_stats.json   # precomputed, see preprocessing.md

  # Task-balanced sampler: batch_size / n_tasks = 128/40 ≈ 3 samples/task/batch
  sampler: task_balanced
  # Sliding window: zero-pad head if episode shorter than L=16
  padding: zero

# ── Evaluation ────────────────────────────────────────────────
evaluation:
  n_rollouts_per_task: 20
  max_steps: 600
  success_criterion: task_completion  # LIBERO goal predicate ≥10 timesteps
  suites_to_eval: [libero_spatial, libero_object, libero_goal, libero_long]

# ── Training budget ───────────────────────────────────────────
training:
  # ~4h per 200-epoch run on RTX 3090
  estimated_runtime_hours: 4.0
```

## HDF5 data structure

```
libero_spatial/
└── pick_up_the_alphabet_soup_and_place_it_in_the_basket.hdf5
    └── data/
        ├── demo_0/
        │   ├── obs/
        │   │   ├── agentview_rgb       (T, 256, 256, 3)  uint8
        │   │   ├── eye_in_hand_rgb     (T, 256, 256, 3)  uint8
        │   │   ├── robot0_joint_pos    (T, 7)            float32  [rad]
        │   │   └── robot0_gripper_qpos (T, 2)            float32  [cm]
        │   ├── actions                 (T, 7)            float32
        │   └── dones                  (T,)               bool
        ├── demo_1/
        │   └── ...
        └── demo_N/
```

## Normalization stats format (data/libero_stats.json)

```json
{
  "action": {
    "mean": [ax0, ax1, ax2, ax3, ax4, ax5],   // dims 0-5 only
    "std":  [sx0, sx1, sx2, sx3, sx4, sx5]
  },
  "proprio": {
    "joint_mean": [j0, j1, j2, j3, j4, j5, j6],
    "joint_std":  [s0, s1, s2, s3, s4, s5, s6],
    "finger_min": [f0_min, f1_min],
    "finger_max": [f0_max, f1_max]
  }
}
```

## Normalization functions

```python
def normalize_action(action, stats):
    """action: (7,) float32"""
    a_norm = action.copy()
    a_norm[:6] = (action[:6] - stats["action"]["mean"]) / (stats["action"]["std"] + 1e-6)
    # dim 6: gripper {-1, +1} → passthrough
    a_norm[6] = action[6]
    return a_norm  # (7,)

def denormalize_action(action_norm, stats):
    """Inverse for inference output"""
    action = action_norm.copy()
    action[:6] = action_norm[:6] * (stats["action"]["std"] + 1e-6) + stats["action"]["mean"]
    # Gripper: threshold to binary
    action[6] = 1.0 if action_norm[6] >= 0.0 else -1.0
    return action  # (7,)

def normalize_proprio(joint_pos, gripper_qpos, stats):
    """joint_pos: (7,) rad; gripper_qpos: (2,) cm"""
    j = (joint_pos - stats["proprio"]["joint_mean"]) / (stats["proprio"]["joint_std"] + 1e-6)
    g = 2.0 * (gripper_qpos - stats["proprio"]["finger_min"]) \
             / (stats["proprio"]["finger_max"] - stats["proprio"]["finger_min"] + 1e-6) - 1.0
    g = np.clip(g, -1.0, 1.0)
    return np.concatenate([j, g])  # (9,)
```

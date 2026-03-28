"""
Check 2: Is flow-matching variance discriminative between success and failure states?

Goal: Test whether V_t (inter-sample action variance from the policy) is meaningfully
different at known-success states vs known-failure states.

If the two distributions overlap heavily → variance gate in PACT-V Level-2 will NOT work.

Strategy:
  - Load LIBERO HDF5 demo data (or run policy rollouts to collect states)
  - For each observation, collect: (obs, is_success_state) where is_success_state = True
    if the task was completed within 5 steps from this obs
  - Sample M action chunks from frozen SmolVLA at each obs
  - Compute V_t = mean whitened variance over M samples
  - Plot and compare distributions

Usage:
    cd /home/ubuntu/antd/arm_robot/pact
    source .venv/bin/activate
    python check_issue/check2_variance_probe.py \
        --policy_path outputs/train/2026-03-20/12-33-51_smolvla_libero_3090/checkpoints/300000/pretrained_model \
        --suite libero_spatial \
        --n_episodes 20 \
        --M 10 \
        --short_prefix 5 \
        --out check_issue/results/check2_libero_spatial.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import os

from lerobot.envs.libero import get_task_init_states, TASK_SUITE_MAX_STEPS


# ── env helpers ──────────────────────────────────────────────────────────────

def build_raw_env(suite, task_id, init_state_id=0):
    task = suite.get_task(task_id)
    bddl_path = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path, camera_heights=256, camera_widths=256
    )
    env.reset()
    init_states = get_task_init_states(suite, task_id)
    env.set_init_state(init_states[init_state_id % len(init_states)])
    dummy = [0, 0, 0, 0, 0, 0, -1]
    for _ in range(10):
        env.step(dummy)
    return env, task.language


def obs_to_tensor_batch(obs, lang, processor, device):
    """Convert raw LIBERO obs dict to SmolVLA batch tensor."""
    from PIL import Image
    agentview = obs.get("agentview_image")
    wrist = obs.get("robot0_eye_in_hand_image")
    images = []
    if agentview is not None:
        images.append(Image.fromarray(agentview))
    if wrist is not None:
        images.append(Image.fromarray(wrist))

    processed = processor(text=[lang], images=images, return_tensors="pt")
    batch = {k: v.to(device) for k, v in processed.items()}

    eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
    eef_quat = obs.get("robot0_eef_quat", np.zeros(4))
    gripper = obs.get("robot0_gripper_qpos", np.zeros(2))
    state = np.concatenate([eef_pos, eef_quat, gripper])
    batch["observation.state"] = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    return batch


# ── variance computation ──────────────────────────────────────────────────────

def compute_variance(policy, batch, M: int, short_prefix: int, device: str):
    """
    Sample M action chunks from policy at same observation.
    Returns scalar whitened variance V_t.
    """
    chunks = []
    for _ in range(M):
        with torch.no_grad():
            # Each call samples a new noise vector → different chunk
            chunk = policy.predict_action_chunk(batch)  # (1, n_action_steps, action_dim)
        prefix = chunk[:, :short_prefix, :].cpu().numpy()  # (1, Hs, action_dim)
        chunks.append(prefix[0])  # (Hs, action_dim)

    chunks = np.stack(chunks, axis=0)  # (M, Hs, action_dim)
    mean_chunk = chunks.mean(axis=0, keepdims=True)   # (1, Hs, action_dim)
    diffs = chunks - mean_chunk                        # (M, Hs, action_dim)

    # Per-dimension std from all data (whitening)
    flat = chunks.reshape(-1, chunks.shape[-1])         # (M*Hs, action_dim)
    std = flat.std(axis=0) + 1e-8                       # (action_dim,)

    # Whitened variance
    whitened = diffs / std[None, None, :]               # (M, Hs, action_dim)
    V = (whitened ** 2).mean()
    return float(V)


# ── collect observations with labels ─────────────────────────────────────────

def collect_labeled_obs(suite, task_id, init_state_id, max_steps,
                        policy_fn_raw, success_horizon=10):
    """
    Run one episode. For each step record obs and label:
      - "pre_success": within `success_horizon` steps before task completion
      - "during_failure": last `success_horizon` steps of a failed episode
      - "mid_episode": all other steps
    Returns list of (obs_dict, label, lang)
    """
    env, lang = build_raw_env(suite, task_id, init_state_id)
    obs_buffer = []
    obs = env._get_observations()
    success = False
    success_step = None

    for step in range(max_steps):
        obs_buffer.append(obs.copy())
        action = policy_fn_raw(obs, lang)
        obs, reward, done, info = env.step(action)
        if done or env._check_success():
            success = True
            success_step = step
            obs_buffer.append(obs.copy())
            break

    env.close()

    labeled = []
    T = len(obs_buffer)
    for t, o in enumerate(obs_buffer):
        if success and success_step is not None and t >= success_step - success_horizon:
            label = "pre_success"
        elif not success and t >= T - success_horizon:
            label = "during_failure"
        else:
            label = "mid_episode"
        labeled.append((o, label, lang))

    return labeled, success


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=str, required=True)
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--task_ids", type=int, nargs="+", default=None)
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--M", type=int, default=10, help="Samples per observation")
    parser.add_argument("--short_prefix", type=int, default=5, help="H_s")
    parser.add_argument("--max_obs_per_label", type=int, default=50,
                        help="Cap observations per label to avoid OOM")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="check_issue/results/check2.json")
    args = parser.parse_args()

    # Load policy
    print(f"Loading SmolVLA from {args.policy_path} ...")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import SmolVLAProcessor
    policy = SmolVLAPolicy.from_pretrained(args.policy_path).to(args.device)
    policy.eval()
    processor = SmolVLAProcessor.from_pretrained(args.policy_path)

    # Simple policy_fn_raw for collection
    @torch.no_grad()
    def policy_fn_raw(obs, lang):
        batch = obs_to_tensor_batch(obs, lang, processor, args.device)
        policy.reset()
        action = policy.select_action(batch)
        return action.cpu().numpy().squeeze()

    # Load suite
    bench = benchmark.get_benchmark_dict()
    suite = bench[args.suite]()
    task_ids = args.task_ids or list(range(len(suite.tasks)))
    max_steps = TASK_SUITE_MAX_STEPS.get(args.suite, 400)

    # Collect labeled observations
    all_labeled = []
    for task_id in task_ids:
        for ep in range(args.n_episodes):
            labeled, success = collect_labeled_obs(
                suite, task_id, ep, max_steps, policy_fn_raw
            )
            all_labeled.extend(labeled)
            print(f"task {task_id} ep {ep}: {'SUCCESS' if success else 'fail'} | "
                  f"collected {len(labeled)} obs")

    # Separate by label
    by_label = {"pre_success": [], "during_failure": [], "mid_episode": []}
    for obs, label, lang in all_labeled:
        by_label[label].append((obs, lang))

    for lbl, items in by_label.items():
        print(f"Label '{lbl}': {len(items)} observations")

    # Cap per label
    for lbl in by_label:
        if len(by_label[lbl]) > args.max_obs_per_label:
            idx = np.random.choice(len(by_label[lbl]), args.max_obs_per_label, replace=False)
            by_label[lbl] = [by_label[lbl][i] for i in idx]

    # Compute variance for each observation
    variances = {lbl: [] for lbl in by_label}
    for lbl, items in by_label.items():
        print(f"\nComputing variance for '{lbl}' ({len(items)} obs)...")
        for i, (obs, lang) in enumerate(items):
            batch = obs_to_tensor_batch(obs, lang, processor, args.device)
            policy.reset()
            V = compute_variance(policy, batch, args.M, args.short_prefix, args.device)
            variances[lbl].append(V)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(items)} done, current V={V:.4f}")

    # Statistics
    print("\n" + "=" * 60)
    print("VARIANCE STATISTICS")
    print("=" * 60)
    stats = {}
    for lbl, vals in variances.items():
        if not vals:
            continue
        arr = np.array(vals)
        s = {
            "n": len(arr),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
        }
        stats[lbl] = s
        print(f"\n  {lbl}:")
        print(f"    n={s['n']}  mean={s['mean']:.4f}  std={s['std']:.4f}")
        print(f"    median={s['median']:.4f}  [p25={s['p25']:.4f}, p75={s['p75']:.4f}]")

    # Overlap analysis: are pre_success and during_failure separable?
    if "pre_success" in stats and "during_failure" in stats:
        v_succ = np.array(variances["pre_success"])
        v_fail = np.array(variances["during_failure"])
        if len(v_succ) > 0 and len(v_fail) > 0:
            # A simple threshold test: if we pick τ_V = median(failure), what fraction of success is below?
            tau = np.median(v_fail)
            recall_succ = (v_succ <= tau).mean()
            recall_fail = (v_fail > tau).mean()
            print(f"\n  Threshold τ_V = median(during_failure) = {tau:.4f}")
            print(f"  Fraction of pre_success with V <= τ_V : {recall_succ:.1%}")
            print(f"  Fraction of during_failure with V > τ_V: {recall_fail:.1%}")
            if recall_succ > 0.6 and recall_fail > 0.6:
                print("  => Variance IS discriminative — Level-2 gate is viable")
            else:
                print("  => Variance is NOT discriminative — Level-2 gate will FAIL")
                print("     Root cause: flow-matching noise-induced variance is not state-quality-correlated")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "suite": args.suite,
        "task_ids": task_ids,
        "M": args.M,
        "short_prefix": args.short_prefix,
        "stats": stats,
        "variances": {k: v for k, v in variances.items()},
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

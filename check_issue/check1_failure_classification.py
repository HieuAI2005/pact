"""
Check 1: Classify failure modes from baseline rollouts.

Goal: Verify whether False-Positive Transitions (FPT) are the dominant failure mode,
or whether motor execution failures (miss grasp, drop object) dominate.

For each failed episode we track per-predicate satisfaction at every timestep and
classify the failure into:
  - NEVER_GRASPED:  robot never satisfied the first predicate (pure motor exec fail)
  - FPT:            robot satisfied k predicates persistently then "skipped" to next stage
                    before actually completing it (transition error)
  - STALLED:        robot partially progressed but got stuck before the final predicate
  - UNKNOWN:        cannot determine from available state info

Usage:
    cd /home/ubuntu/antd/arm_robot/pact
    source .venv/bin/activate
    python check_issue/check1_failure_classification.py \
        --policy_path outputs/train/2026-03-20/12-33-51_smolvla_libero_3090/checkpoints/300000/pretrained_model \
        --suite libero_spatial \
        --n_episodes 20 \
        --task_ids 0 1 2 \
        --out check_issue/results/check1_libero_spatial.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# Add project src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path
import os

from lerobot.envs.libero import get_task_init_states, TASK_SUITE_MAX_STEPS


# ── helpers ──────────────────────────────────────────────────────────────────

def get_predicates(env):
    """Return individual goal predicates from the parsed BDDL problem."""
    return env.parsed_problem.get("goal_state", [])


def eval_predicate_safe(env, predicate):
    """Safely evaluate a single predicate; returns bool."""
    try:
        return bool(env._eval_predicate(predicate))
    except Exception:
        return False


def build_env(suite, task_id, init_state_id=0):
    """Build a raw OffScreenRenderEnv for a given task."""
    task = suite.get_task(task_id)
    bddl_path = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    env_args = {
        "bddl_file_name": bddl_path,
        "camera_heights": 256,
        "camera_widths": 256,
    }
    env = OffScreenRenderEnv(**env_args)
    env.reset()

    init_states = get_task_init_states(suite, task_id)
    env.set_init_state(init_states[init_state_id % len(init_states)])

    # settle simulation
    dummy = [0, 0, 0, 0, 0, 0, -1]
    for _ in range(10):
        env.step(dummy)

    return env, task.language


# ── classify single episode ───────────────────────────────────────────────────

PERSISTENCE = 3   # frames a predicate must stay True to count as "persistently satisfied"


def classify_failure(predicate_history: list[list[bool]]) -> str:
    """
    predicate_history[t][k] = True/False for predicate k at timestep t.
    Returns failure category.
    """
    if not predicate_history or not predicate_history[0]:
        return "UNKNOWN"

    n_predicates = len(predicate_history[0])
    T = len(predicate_history)

    # Persistent completion time for each predicate
    T_k = []
    for k in range(n_predicates):
        t_k = None
        for t in range(T - PERSISTENCE + 1):
            if all(predicate_history[t + d][k] for d in range(PERSISTENCE)):
                t_k = t
                break
        T_k.append(t_k)

    # How many predicates were persistently satisfied?
    n_completed = sum(1 for t in T_k if t is not None)

    if n_completed == 0:
        return "NEVER_GRASPED"
    elif n_completed == n_predicates:
        # All predicates satisfied — should not be a failure episode, log as UNKNOWN
        return "UNKNOWN_ALL_SATISFIED"
    else:
        # k predicates satisfied, then failed at k+1
        # Was the (k+1)-th predicate ever transiently True (but not persistently)?
        next_k = n_completed  # 0-indexed
        ever_transient = any(predicate_history[t][next_k] for t in range(T))
        if ever_transient:
            return "FPT"
        else:
            return "STALLED"


# ── rollout with a random policy (placeholder until real policy is integrated) ─

def rollout_episode(suite, task_id, init_state_id, max_steps, policy_fn):
    """
    Run one episode with policy_fn(obs) -> action.
    Returns (success, predicate_history, n_steps)
    """
    env, lang = build_env(suite, task_id, init_state_id)
    predicates = get_predicates(env)

    if not predicates:
        env.close()
        return None, [], 0

    predicate_history = []
    obs = env._get_observations()
    success = False

    for step in range(max_steps):
        # Evaluate each predicate at this step
        preds_now = [eval_predicate_safe(env, p) for p in predicates]
        predicate_history.append(preds_now)

        action = policy_fn(obs, lang)
        obs, reward, done, info = env.step(action)

        if done or env._check_success():
            success = True
            break

    # Final predicate state
    preds_now = [eval_predicate_safe(env, p) for p in predicates]
    predicate_history.append(preds_now)

    env.close()
    return success, predicate_history, len(predicate_history)


# ── integrate with SmolVLA policy ─────────────────────────────────────────────

def load_smolvla_policy(policy_path: str, device: str = "cuda"):
    """Load SmolVLA policy from checkpoint."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained(policy_path)
    policy = policy.to(device)
    policy.eval()
    return policy


def make_smolvla_policy_fn(policy, processor, device, rename_map=None):
    """
    Returns a closure that takes raw LIBERO obs dict + lang str
    and returns a numpy action array.

    rename_map: e.g. {"observation.images.image": "observation.images.camera1"}
    """
    rename_map = rename_map or {}

    @torch.no_grad()
    def policy_fn(obs, lang):
        # Build batch expected by SmolVLA processor
        import numpy as np
        from PIL import Image

        # Get images from LIBERO obs
        agentview = obs.get("agentview_image")    # (H, W, 3) uint8
        wrist = obs.get("robot0_eye_in_hand_image")

        if agentview is None:
            return np.zeros(7, dtype=np.float32)

        images_dict = {}
        if agentview is not None:
            images_dict["observation.images.image"] = Image.fromarray(agentview)
        if wrist is not None:
            images_dict["observation.images.image2"] = Image.fromarray(wrist)

        # Build robot state
        eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
        eef_quat = obs.get("robot0_eef_quat", np.zeros(4))
        gripper_qpos = obs.get("robot0_gripper_qpos", np.zeros(2))
        state = np.concatenate([eef_pos, eef_quat, gripper_qpos])

        # Process through SmolVLA processor
        processed = processor(
            text=[lang],
            images=list(images_dict.values()),
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in processed.items()}
        batch["observation.state"] = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        # Apply rename map
        for old_k, new_k in rename_map.items():
            if old_k in batch:
                batch[new_k] = batch.pop(old_k)

        action = policy.select_action(batch)
        return action.cpu().numpy().squeeze()

    return policy_fn


def make_random_policy_fn():
    """Fallback: random policy for testing without GPU."""
    def policy_fn(obs, lang):
        return np.random.uniform(-0.1, 0.1, size=7).astype(np.float32)
    return policy_fn


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=str, default=None,
                        help="Path to SmolVLA checkpoint. If None, uses random policy.")
    parser.add_argument("--suite", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--n_episodes", type=int, default=10,
                        help="Episodes per task")
    parser.add_argument("--task_ids", type=int, nargs="+", default=None,
                        help="Task IDs to evaluate (default: all)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="check_issue/results/check1.json")
    parser.add_argument("--rename_map", type=str, default=None,
                        help='JSON rename map e.g. \'{"observation.images.image":"observation.images.camera1"}\'')
    args = parser.parse_args()

    # Build rename map
    rename_map = json.loads(args.rename_map) if args.rename_map else {}

    # Load suite
    bench = benchmark.get_benchmark_dict()
    suite = bench[args.suite]()
    total_tasks = len(suite.tasks)
    task_ids = args.task_ids if args.task_ids is not None else list(range(total_tasks))
    max_steps = TASK_SUITE_MAX_STEPS.get(args.suite, 400)

    print(f"Suite: {args.suite} | Tasks: {task_ids} | Episodes/task: {args.n_episodes} | Max steps: {max_steps}")

    # Load policy
    if args.policy_path:
        print(f"Loading SmolVLA from {args.policy_path} ...")
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.smolvla.processor_smolvla import SmolVLAProcessor
        policy = SmolVLAPolicy.from_pretrained(args.policy_path).to(args.device)
        policy.eval()
        processor = SmolVLAProcessor.from_pretrained(args.policy_path)
        policy_fn = make_smolvla_policy_fn(policy, processor, args.device, rename_map)
    else:
        print("WARNING: No policy_path provided, using random policy for testing")
        policy_fn = make_random_policy_fn()

    # Run rollouts
    results = defaultdict(lambda: defaultdict(list))
    category_counts = defaultdict(int)

    for task_id in task_ids:
        task_name = suite.get_task(task_id).name
        print(f"\n--- Task {task_id}: {task_name} ---")

        for ep in range(args.n_episodes):
            success, pred_hist, n_steps = rollout_episode(
                suite, task_id, ep, max_steps, policy_fn
            )

            if success is None:
                print(f"  ep {ep}: SKIP (no predicates)")
                continue

            if success:
                category = "SUCCESS"
            else:
                category = classify_failure(pred_hist)

            category_counts[category] += 1

            # Count max stage reached
            n_predicates = len(pred_hist[0]) if pred_hist else 0
            max_stage = 0
            for k in range(n_predicates):
                ever_true = any(pred_hist[t][k] for t in range(len(pred_hist)))
                if ever_true:
                    max_stage = k + 1

            results[task_id]["episodes"].append({
                "ep": ep,
                "success": success,
                "category": category,
                "n_steps": n_steps,
                "n_predicates": n_predicates,
                "max_stage_reached": max_stage,
            })
            print(f"  ep {ep:2d}: {'SUCCESS' if success else 'FAIL'} | category={category} | "
                  f"max_stage={max_stage}/{n_predicates} | steps={n_steps}")

    # Summary
    total = sum(category_counts.values())
    print("\n" + "=" * 60)
    print("FAILURE CLASSIFICATION SUMMARY")
    print("=" * 60)
    for cat, count in sorted(category_counts.items()):
        pct = count / total * 100 if total > 0 else 0
        print(f"  {cat:30s}: {count:4d} ({pct:5.1f}%)")
    print(f"  {'TOTAL':30s}: {total:4d}")

    fpt_count = category_counts.get("FPT", 0)
    fail_total = total - category_counts.get("SUCCESS", 0)
    if fail_total > 0:
        fpt_rate = fpt_count / fail_total
        print(f"\n  FPT rate among failures: {fpt_rate:.1%}")
        if fpt_rate > 0.5:
            print("  => FPT is DOMINANT (>50% of failures) — PACT-V hypothesis supported")
        elif fpt_rate > 0.3:
            print("  => FPT is significant but not dominant — paper needs nuanced framing")
        else:
            print("  => FPT is MINOR (<30% of failures) — paper core assumption may be WRONG")

    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "suite": args.suite,
        "task_ids": task_ids,
        "n_episodes": args.n_episodes,
        "category_counts": dict(category_counts),
        "total_episodes": total,
        "fpt_rate_among_failures": (
            category_counts.get("FPT", 0) / max(1, total - category_counts.get("SUCCESS", 0))
        ),
        "per_task": {str(k): v for k, v in results.items()},
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

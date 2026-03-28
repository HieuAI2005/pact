"""
Check 3: Can SmolVLA backbone features predict task completion predicates?

Goal: If a linear probe trained on frozen SmolVLA backbone hidden states cannot
predict φ_k(s_t) with >80% accuracy, then [PROG] token also won't learn it —
because the visual information needed for completion detection is simply not encoded
in the compressed representation.

Strategy:
  1. Run policy rollouts (or use demos) recording at each step:
       - backbone hidden state h_t (last token or mean pooled)
       - per-predicate ground truth: φ_k(s_t) ∈ {0, 1}
  2. Train logistic regression on (h_t, φ_k) pairs
  3. Report accuracy, AUC per predicate

Usage:
    cd /home/ubuntu/antd/arm_robot/pact
    source .venv/bin/activate
    python check_issue/check3_linear_probe.py \
        --policy_path outputs/train/2026-03-20/12-33-51_smolvla_libero_3090/checkpoints/300000/pretrained_model \
        --suite libero_spatial \
        --task_ids 0 1 2 \
        --n_episodes 20 \
        --out check_issue/results/check3_libero_spatial.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import os

from lerobot.envs.libero import get_task_init_states, TASK_SUITE_MAX_STEPS


# ── backbone feature extraction ───────────────────────────────────────────────

class SmolVLAFeatureExtractor:
    """
    Extract the backbone (VLM text model) hidden states from SmolVLA.
    We hook into the last transformer layer to get a per-observation
    feature vector before the action expert sees it.
    """

    def __init__(self, policy, processor, device):
        self.policy = policy
        self.processor = processor
        self.device = device
        self._features = None
        self._hook = None
        self._register_hook()

    def _register_hook(self):
        """Hook into the last layer of the VLM text model."""
        vlm_model = self.policy.model.model.vlm.model
        last_layer = vlm_model.text_model.layers[-1]

        def hook_fn(module, input, output):
            # output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                self._features = output[0].detach()
            else:
                self._features = output.detach()

        self._hook = last_layer.register_forward_hook(hook_fn)

    def remove_hook(self):
        if self._hook is not None:
            self._hook.remove()

    def extract(self, obs, lang) -> np.ndarray:
        """
        Forward pass through backbone. Returns mean-pooled hidden state (D,).
        """
        from PIL import Image

        agentview = obs.get("agentview_image")
        wrist = obs.get("robot0_eye_in_hand_image")
        images = []
        if agentview is not None:
            images.append(Image.fromarray(agentview))
        if wrist is not None:
            images.append(Image.fromarray(wrist))

        processed = self.processor(text=[lang], images=images, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in processed.items()}

        eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
        eef_quat = obs.get("robot0_eef_quat", np.zeros(4))
        gripper = obs.get("robot0_gripper_qpos", np.zeros(2))
        state = np.concatenate([eef_pos, eef_quat, gripper])
        batch["observation.state"] = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        self._features = None
        with torch.no_grad():
            # Trigger backbone forward pass via select_action
            self.policy.reset()
            self.policy.select_action(batch)

        if self._features is None:
            return None

        # Mean pool over sequence dimension → (D,)
        feat = self._features[0].float().mean(dim=0).cpu().numpy()
        return feat


# ── env helpers ───────────────────────────────────────────────────────────────

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


def get_predicates(env):
    return env.parsed_problem.get("goal_state", [])


def eval_predicate_safe(env, predicate) -> int:
    try:
        return int(bool(env._eval_predicate(predicate)))
    except Exception:
        return 0


# ── data collection ───────────────────────────────────────────────────────────

def collect_features_and_labels(suite, task_id, init_state_id, max_steps,
                                 extractor, policy_fn_raw):
    """
    Run one episode. At each step extract backbone features and predicate labels.
    Returns:
        features: np.ndarray (T, D)
        labels:   np.ndarray (T, n_predicates)
        n_predicates: int
    """
    env, lang = build_raw_env(suite, task_id, init_state_id)
    predicates = get_predicates(env)
    if not predicates:
        env.close()
        return None, None, 0

    features = []
    labels = []
    obs = env._get_observations()

    for step in range(max_steps):
        # Extract features
        feat = extractor.extract(obs, lang)
        if feat is None:
            obs, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            continue

        # Evaluate predicates
        preds = [eval_predicate_safe(env, p) for p in predicates]

        features.append(feat)
        labels.append(preds)

        action = policy_fn_raw(obs, lang)
        obs, reward, done, info = env.step(action)

        if done or env._check_success():
            # Record final state
            feat = extractor.extract(obs, lang)
            if feat is not None:
                preds = [eval_predicate_safe(env, p) for p in predicates]
                features.append(feat)
                labels.append(preds)
            break

    env.close()

    if not features:
        return None, None, len(predicates)

    return np.array(features), np.array(labels), len(predicates)


# ── linear probe training ─────────────────────────────────────────────────────

def train_linear_probe(X_train, y_train, X_test, y_test):
    """
    Train sklearn LogisticRegression and return accuracy + AUC.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_tr, y_train)

    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_test, y_pred)

    try:
        y_prob = clf.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = float("nan")

    # Baseline: predict majority class
    majority = int(y_train.mean() >= 0.5)
    baseline_acc = max(y_test.mean(), 1 - y_test.mean())

    return {"accuracy": acc, "auc": auc, "baseline_accuracy": baseline_acc,
            "n_train": len(y_train), "n_test": len(y_test),
            "class_balance": float(y_test.mean())}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=str, required=True)
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--task_ids", type=int, nargs="+", default=None)
    parser.add_argument("--n_episodes", type=int, default=15,
                        help="Episodes per task (split 70/30 for train/test)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="check_issue/results/check3.json")
    args = parser.parse_args()

    print(f"Loading SmolVLA from {args.policy_path} ...")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import SmolVLAProcessor
    policy = SmolVLAPolicy.from_pretrained(args.policy_path).to(args.device)
    policy.eval()
    processor = SmolVLAProcessor.from_pretrained(args.policy_path)

    extractor = SmolVLAFeatureExtractor(policy, processor, args.device)

    @torch.no_grad()
    def policy_fn_raw(obs, lang):
        from PIL import Image
        agentview = obs.get("agentview_image")
        wrist = obs.get("robot0_eye_in_hand_image")
        images = []
        if agentview is not None:
            images.append(Image.fromarray(agentview))
        if wrist is not None:
            images.append(Image.fromarray(wrist))
        processed = processor(text=[lang], images=images, return_tensors="pt")
        batch = {k: v.to(args.device) for k, v in processed.items()}
        eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
        eef_quat = obs.get("robot0_eef_quat", np.zeros(4))
        gripper = obs.get("robot0_gripper_qpos", np.zeros(2))
        state = np.concatenate([eef_pos, eef_quat, gripper])
        batch["observation.state"] = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(args.device)
        policy.reset()
        action = policy.select_action(batch)
        return action.cpu().numpy().squeeze()

    bench = benchmark.get_benchmark_dict()
    suite = bench[args.suite]()
    task_ids = args.task_ids or list(range(len(suite.tasks)))
    max_steps = TASK_SUITE_MAX_STEPS.get(args.suite, 400)

    all_results = {}

    for task_id in task_ids:
        task_name = suite.get_task(task_id).name
        print(f"\n=== Task {task_id}: {task_name} ===")

        all_features = []
        all_labels = []
        n_preds = 0

        for ep in range(args.n_episodes):
            feats, labels, np_ = collect_features_and_labels(
                suite, task_id, ep, max_steps, extractor, policy_fn_raw
            )
            if feats is None:
                continue
            all_features.append(feats)
            all_labels.append(labels)
            n_preds = np_
            print(f"  ep {ep}: {len(feats)} steps, {np_} predicates")

        if not all_features or n_preds == 0:
            print(f"  Skipping task {task_id}: no data")
            continue

        X = np.concatenate(all_features, axis=0)   # (N, D)
        Y = np.concatenate(all_labels, axis=0)     # (N, n_preds)

        print(f"\n  Total samples: {len(X)}, feature dim: {X.shape[1]}, n_predicates: {n_preds}")

        # Train/test split (70/30)
        split = int(0.7 * len(X))
        X_train, X_test = X[:split], X[split:]
        Y_train, Y_test = Y[:split], Y[split:]

        task_probe_results = {}
        for k in range(n_preds):
            y_tr = Y_train[:, k]
            y_te = Y_test[:, k]

            # Skip if no variation
            if y_te.sum() == 0 or y_te.sum() == len(y_te):
                task_probe_results[f"predicate_{k}"] = {
                    "skipped": True, "reason": "no variation in test set"
                }
                continue

            result = train_linear_probe(X_train, y_tr, X_test, y_te)
            task_probe_results[f"predicate_{k}"] = result

            print(f"  Predicate {k}: acc={result['accuracy']:.3f}  "
                  f"auc={result['auc']:.3f}  baseline={result['baseline_accuracy']:.3f}  "
                  f"balance={result['class_balance']:.2f}")

        all_results[str(task_id)] = task_probe_results

    # Summary
    print("\n" + "=" * 60)
    print("LINEAR PROBE SUMMARY")
    print("=" * 60)
    all_accs = []
    all_aucs = []
    for tid, probes in all_results.items():
        for pred_name, r in probes.items():
            if r.get("skipped"):
                continue
            all_accs.append(r["accuracy"])
            if not np.isnan(r["auc"]):
                all_aucs.append(r["auc"])

    if all_accs:
        mean_acc = np.mean(all_accs)
        mean_auc = np.mean(all_aucs) if all_aucs else float("nan")
        print(f"  Mean accuracy across predicates: {mean_acc:.3f}")
        print(f"  Mean AUC across predicates:      {mean_auc:.3f}")
        if mean_acc >= 0.80:
            print("  => Backbone CAN encode predicate completion — [PROG] token is viable")
        elif mean_acc >= 0.65:
            print("  => Backbone partially encodes completion — [PROG] may learn weak signal")
        else:
            print("  => Backbone CANNOT encode completion (accuracy < 65%)")
            print("     Root cause confirmed: visual token compression removes the discriminative cues")
            print("     Both progress head AND variance gate will struggle for the same reason")

    # Save
    extractor.remove_hook()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "suite": args.suite,
        "task_ids": task_ids,
        "mean_accuracy": float(np.mean(all_accs)) if all_accs else None,
        "mean_auc": float(np.mean(all_aucs)) if all_aucs else None,
        "per_task": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

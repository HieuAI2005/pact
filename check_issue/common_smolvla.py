from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.envs.factory import make_env_config, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def parse_rename_map(rename_map_arg: str | None) -> dict[str, str]:
    if not rename_map_arg:
        return {}
    rename_map = json.loads(rename_map_arg)
    if not isinstance(rename_map, dict):
        raise ValueError("--rename_map must decode to a JSON object")
    return {str(k): str(v) for k, v in rename_map.items()}


def infer_rename_map(policy: SmolVLAPolicy, rename_map: dict[str, str] | None = None) -> dict[str, str]:
    effective = {}
    expected_visuals = set(policy.config.image_features.keys())
    if "observation.images.camera1" in expected_visuals and "observation.images.image" not in expected_visuals:
        effective["observation.images.image"] = "observation.images.camera1"
    if "observation.images.camera2" in expected_visuals and "observation.images.image2" not in expected_visuals:
        effective["observation.images.image2"] = "observation.images.camera2"
    if rename_map:
        effective.update(rename_map)
    return effective


def _batched_array(value: Any, default: np.ndarray) -> np.ndarray:
    array = np.asarray(default if value is None else value)
    return np.expand_dims(array, axis=0)


def build_single_libero_observation(obs: dict[str, Any]) -> dict[str, Any] | None:
    agentview = obs.get("agentview_image")
    wrist = obs.get("robot0_eye_in_hand_image")

    if agentview is None:
        return None

    observation = {
        "pixels": {
            "image": agentview,
        },
        "robot_state": {
            "eef": {
                "pos": _batched_array(obs.get("robot0_eef_pos"), np.zeros(3, dtype=np.float32)),
                "quat": _batched_array(obs.get("robot0_eef_quat"), np.array([0, 0, 0, 1], dtype=np.float32)),
            },
            "gripper": {
                "qpos": _batched_array(obs.get("robot0_gripper_qpos"), np.zeros(2, dtype=np.float32)),
            },
        },
    }

    if wrist is not None:
        observation["pixels"]["image2"] = wrist

    return observation


@dataclass
class SmolVLARuntime:
    policy: SmolVLAPolicy
    env_preprocessor: Any
    preprocessor: Any
    postprocessor: Any
    rename_map: dict[str, str]

    def reset(self) -> None:
        self.policy.reset()
        self.env_preprocessor.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def prepare_batch(self, obs: dict[str, Any], lang: str) -> dict[str, Any] | None:
        env_obs = build_single_libero_observation(obs)
        if env_obs is None:
            return None

        batch = preprocess_observation(env_obs)
        batch["task"] = lang
        batch = self.env_preprocessor(batch)
        return self.preprocessor(batch)

    @torch.no_grad()
    def select_action(self, obs: dict[str, Any], lang: str) -> np.ndarray:
        batch = self.prepare_batch(obs, lang)
        if batch is None:
            return np.zeros(7, dtype=np.float32)

        action = self.policy.select_action(batch)
        action = self.postprocessor(action)
        return action.detach().cpu().numpy().squeeze()


def load_smolvla_runtime(
    policy_path: str,
    suite: str,
    device: str = "cuda",
    rename_map: dict[str, str] | None = None,
) -> SmolVLARuntime:
    policy = SmolVLAPolicy.from_pretrained(policy_path).to(device)
    policy.eval()
    policy.config.pretrained_path = Path(policy_path)

    effective_rename_map = infer_rename_map(policy, rename_map)
    preprocessor_overrides = {
        "device_processor": {"device": device},
        "rename_observations_processor": {"rename_map": effective_rename_map},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=policy_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_cfg = make_env_config("libero", task=suite)
    env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    return SmolVLARuntime(
        policy=policy,
        env_preprocessor=env_preprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        rename_map=effective_rename_map,
    )

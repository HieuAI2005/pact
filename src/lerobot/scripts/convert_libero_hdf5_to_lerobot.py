#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert raw LIBERO HDF5 files into a local LeRobot dataset v3.

Example:
python -m lerobot.scripts.convert_libero_hdf5_to_lerobot \
    --raw-dir /data/LIBERO \
    --repo-id local/libero_full \
    --root /data/lerobot/libero_full \
    --state-mode auto \
    --rotate-images-180 true \
    --use-videos true \
    --streaming-encoding true \
    --vcodec auto

The converter recursively scans `--raw-dir` for `.hdf5` / `.h5` files. Each
`data/demo_*` group becomes one LeRobot episode.

State modes:
- auto: prefer EEF+gripper (8-D) when available, otherwise joint+gripper (9-D)
- eef_gripper: eef_pos(3) + axis_angle(3) + gripper_qpos(2) = 8-D
- joint_gripper: joint_pos(7) + gripper_qpos(2) = 9-D
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
from tqdm import tqdm

CAMERA_KEY_MAP = {
    "observation.images.image": ("agentview_rgb", "agentview_image"),
    "observation.images.image2": ("eye_in_hand_rgb", "robot0_eye_in_hand_image"),
}
JOINT_STATE_KEYS = ("robot0_joint_pos", "joint_states")
GRIPPER_STATE_KEYS = ("robot0_gripper_qpos", "gripper_states")
EEF_POS_KEYS = ("robot0_eef_pos", "ee_pos")
EEF_ORI_KEYS = ("robot0_eef_quat", "ee_ori")

STATE_MODE_AUTO = "auto"
STATE_MODE_EEF = "eef_gripper"
STATE_MODE_JOINT = "joint_gripper"


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def require_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise ImportError(
            "Missing optional dependency 'h5py'. Install it first, for example with `pip install h5py`."
        ) from exc
    return h5py


def find_hdf5_files(raw_dir: Path, suites: set[str] | None = None) -> list[Path]:
    files = sorted([*raw_dir.rglob("*.hdf5"), *raw_dir.rglob("*.h5")])
    if suites is None:
        return files

    filtered = []
    for path in files:
        rel_parts = path.relative_to(raw_dir).parts
        if any(part in suites for part in rel_parts[:-1]):
            filtered.append(path)
    return filtered


def sort_demo_names(data_group: h5py.Group) -> list[str]:
    def demo_key(name: str) -> tuple[int, str]:
        try:
            return int(name.split("_")[-1]), name
        except ValueError:
            return 10**9, name

    return sorted(data_group.keys(), key=demo_key)


def decode_attr(value: Any) -> str | None:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray) and value.shape == ():
        return decode_attr(value.item())
    return None


def infer_task_name(h5_file: h5py.File, hdf5_path: Path) -> str:
    attr_keys = ["task", "task_description", "language", "lang", "description", "problem_statement"]
    for container in (h5_file, h5_file.get("data")):
        if container is None:
            continue
        for key in attr_keys:
            if key in container.attrs:
                decoded = decode_attr(container.attrs[key])
                if decoded:
                    return decoded.strip()

    stem = hdf5_path.stem
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]
    stem = stem.replace("_", " ").strip()
    return " ".join(stem.split())


def get_obs_dataset(demo_obs: h5py.Group, candidates: tuple[str, ...], *, required: bool) -> h5py.Dataset | None:
    for key in candidates:
        if key in demo_obs:
            return demo_obs[key]
    if required:
        raise KeyError(f"Missing observation key. Tried: {', '.join(candidates)}")
    return None


def quat_to_axis_angle(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float32)
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion with last dim 4, got shape {quat.shape}")

    xyz = quat[..., :3]
    w = np.clip(quat[..., 3], -1.0, 1.0)
    norm_xyz = np.linalg.norm(xyz, axis=-1, keepdims=True)
    axis = np.divide(xyz, np.maximum(norm_xyz, 1e-10), out=np.zeros_like(xyz), where=norm_xyz > 1e-10)
    angle = 2.0 * np.arccos(w)[..., None]
    return (axis * angle).astype(np.float32)


def normalize_eef_orientation(orientation: np.ndarray) -> np.ndarray:
    orientation = np.asarray(orientation, dtype=np.float32)
    if orientation.shape[-1] == 4:
        return quat_to_axis_angle(orientation)
    if orientation.shape[-1] == 3:
        return orientation.astype(np.float32)
    raise ValueError(f"Unsupported EEF orientation shape: {orientation.shape}")


def maybe_rotate_image_180(image: np.ndarray, rotate: bool) -> np.ndarray:
    if not rotate:
        return image
    if image.ndim != 3:
        raise ValueError(f"Expected image with shape (H, W, C), got {image.shape}")
    return np.flip(image, axis=(0, 1)).copy()


def resolve_state_mode(demo_obs: h5py.Group, state_mode: str) -> str:
    has_eef = any(key in demo_obs for key in EEF_POS_KEYS) and any(key in demo_obs for key in EEF_ORI_KEYS)
    has_joint = any(key in demo_obs for key in JOINT_STATE_KEYS)

    if state_mode == STATE_MODE_AUTO:
        if has_eef:
            return STATE_MODE_EEF
        if has_joint:
            return STATE_MODE_JOINT
        raise KeyError("Could not infer state mode: neither EEF nor joint state keys are present.")

    if state_mode == STATE_MODE_EEF and not has_eef:
        raise KeyError("Requested state_mode=eef_gripper but compatible EEF keys are missing.")
    if state_mode == STATE_MODE_JOINT and not has_joint:
        raise KeyError("Requested state_mode=joint_gripper but compatible joint keys are missing.")
    return state_mode


def build_features(state_mode: str, use_videos: bool, image_shape: tuple[int, ...], action_dim: int) -> dict[str, dict]:
    visual_dtype = "video" if use_videos else "image"
    if len(image_shape) != 3:
        raise ValueError(f"Expected image shape (H, W, C), got {image_shape}")

    if state_mode == STATE_MODE_EEF:
        state_names = [
            "eef_pos_x",
            "eef_pos_y",
            "eef_pos_z",
            "eef_axisangle_x",
            "eef_axisangle_y",
            "eef_axisangle_z",
            "gripper_qpos_0",
            "gripper_qpos_1",
        ]
    elif state_mode == STATE_MODE_JOINT:
        state_names = [f"joint_pos_{i}" for i in range(7)] + ["gripper_qpos_0", "gripper_qpos_1"]
    else:
        raise ValueError(f"Unsupported state mode: {state_mode}")

    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": state_names,
        },
        "observation.images.image": {
            "dtype": visual_dtype,
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.images.image2": {
            "dtype": visual_dtype,
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [f"action_{i}" for i in range(action_dim)],
        },
    }


def compute_frame_count(
    agentview: h5py.Dataset,
    wrist: h5py.Dataset,
    gripper_qpos: h5py.Dataset,
    actions: h5py.Dataset,
    *,
    joint_pos: h5py.Dataset | None = None,
    eef_pos: h5py.Dataset | None = None,
    eef_ori: h5py.Dataset | None = None,
) -> int:
    lengths = [len(agentview), len(wrist), len(gripper_qpos), len(actions)]
    if joint_pos is not None:
        lengths.append(len(joint_pos))
    if eef_pos is not None:
        lengths.append(len(eef_pos))
    if eef_ori is not None:
        lengths.append(len(eef_ori))
    return min(lengths)


def convert_demo_to_episode(
    dataset: LeRobotDataset,
    demo_group: h5py.Group,
    task_name: str,
    state_mode: str,
    rotate_images_180: bool,
) -> int:
    if "obs" not in demo_group:
        raise KeyError("Demo group is missing required `obs` subgroup.")
    if "actions" not in demo_group:
        raise KeyError("Demo group is missing required `actions` dataset.")

    obs_group = demo_group["obs"]
    agentview = get_obs_dataset(obs_group, CAMERA_KEY_MAP["observation.images.image"], required=True)
    wrist = get_obs_dataset(obs_group, CAMERA_KEY_MAP["observation.images.image2"], required=True)
    gripper_qpos = get_obs_dataset(obs_group, GRIPPER_STATE_KEYS, required=True)
    actions = demo_group["actions"]

    resolved_state_mode = resolve_state_mode(obs_group, state_mode)
    joint_pos = get_obs_dataset(obs_group, JOINT_STATE_KEYS, required=False)
    eef_pos = get_obs_dataset(obs_group, EEF_POS_KEYS, required=False)
    eef_ori = get_obs_dataset(obs_group, EEF_ORI_KEYS, required=False)

    frame_count = compute_frame_count(
        agentview,
        wrist,
        gripper_qpos,
        actions,
        joint_pos=joint_pos,
        eef_pos=eef_pos,
        eef_ori=eef_ori,
    )
    if frame_count <= 0:
        raise ValueError("Episode has no valid frames.")

    for frame_idx in range(frame_count):
        if resolved_state_mode == STATE_MODE_EEF:
            assert eef_pos is not None and eef_ori is not None
            state = np.concatenate(
                [
                    np.asarray(eef_pos[frame_idx], dtype=np.float32),
                    normalize_eef_orientation(np.asarray(eef_ori[frame_idx], dtype=np.float32)),
                    np.asarray(gripper_qpos[frame_idx], dtype=np.float32),
                ],
                axis=-1,
            )
        else:
            assert joint_pos is not None
            state = np.concatenate(
                [
                    np.asarray(joint_pos[frame_idx], dtype=np.float32),
                    np.asarray(gripper_qpos[frame_idx], dtype=np.float32),
                ],
                axis=-1,
            )

        frame = {
            "observation.state": state.astype(np.float32),
            "observation.images.image": maybe_rotate_image_180(
                np.asarray(agentview[frame_idx], dtype=np.uint8),
                rotate_images_180,
            ),
            "observation.images.image2": maybe_rotate_image_180(
                np.asarray(wrist[frame_idx], dtype=np.uint8),
                rotate_images_180,
            ),
            "action": np.asarray(actions[frame_idx], dtype=np.float32),
            "task": task_name,
        }
        dataset.add_frame(frame)

    dataset.save_episode()
    return frame_count


def build_dataset(
    raw_dir: Path,
    repo_id: str,
    root: Path,
    fps: int,
    state_mode: str,
    use_videos: bool,
    streaming_encoding: bool,
    vcodec: str,
    encoder_threads: int | None,
    rotate_images_180: bool,
    suites: set[str] | None,
    max_files: int | None,
    max_episodes: int | None,
) -> LeRobotDataset:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    h5py = require_h5py()

    hdf5_files = find_hdf5_files(raw_dir, suites=suites)
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found under {raw_dir}")

    if max_files is not None:
        hdf5_files = hdf5_files[:max_files]

    with h5py.File(hdf5_files[0], "r") as first_file:
        first_demo_name = sort_demo_names(first_file["data"])[0]
        first_demo = first_file["data"][first_demo_name]
        first_obs = first_demo["obs"]
        resolved_state_mode = resolve_state_mode(first_obs, state_mode)
        first_image = np.asarray(
            get_obs_dataset(first_obs, CAMERA_KEY_MAP["observation.images.image"], required=True)[0]
        )
        first_actions = np.asarray(first_demo["actions"][0], dtype=np.float32)

    features = build_features(
        state_mode=resolved_state_mode,
        use_videos=use_videos,
        image_shape=tuple(first_image.shape),
        action_dim=int(first_actions.shape[-1]),
    )

    logging.info(
        "Creating LeRobot dataset at %s | repo_id=%s | state_mode=%s | use_videos=%s | streaming_encoding=%s | rotate_images_180=%s",
        root,
        repo_id,
        resolved_state_mode,
        use_videos,
        streaming_encoding and use_videos,
        rotate_images_180,
    )
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=fps,
        features=features,
        use_videos=use_videos,
        vcodec=vcodec,
        streaming_encoding=streaming_encoding and use_videos,
        encoder_threads=encoder_threads,
    )


def convert_dataset(args: argparse.Namespace) -> None:
    h5py = require_h5py()

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    root = Path(args.root).expanduser().resolve()
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"Raw LIBERO directory does not exist: {raw_dir}")
    if root.exists():
        raise FileExistsError(f"Output directory already exists: {root}")

    suites = set(args.suites.split(",")) if args.suites else None
    dataset = build_dataset(
        raw_dir=raw_dir,
        repo_id=args.repo_id,
        root=root,
        fps=args.fps,
        state_mode=args.state_mode,
        use_videos=args.use_videos,
        streaming_encoding=args.streaming_encoding,
        vcodec=args.vcodec,
        encoder_threads=args.encoder_threads,
        rotate_images_180=args.rotate_images_180,
        suites=suites,
        max_files=args.max_files,
        max_episodes=args.max_episodes,
    )

    total_files = 0
    total_episodes = 0
    total_frames = 0
    hdf5_files = find_hdf5_files(raw_dir, suites=suites)
    if args.max_files is not None:
        hdf5_files = hdf5_files[: args.max_files]

    try:
        for hdf5_path in tqdm(hdf5_files, desc="HDF5 files"):
            with h5py.File(hdf5_path, "r") as h5_file:
                if "data" not in h5_file:
                    raise KeyError(f"Missing `data` group in {hdf5_path}")

                task_name = infer_task_name(h5_file, hdf5_path)
                demo_names = sort_demo_names(h5_file["data"])
                logging.info("Processing %s | task=%s | demos=%d", hdf5_path, task_name, len(demo_names))
                total_files += 1

                for demo_name in tqdm(demo_names, desc=hdf5_path.name, leave=False):
                    frame_count = convert_demo_to_episode(
                        dataset=dataset,
                        demo_group=h5_file["data"][demo_name],
                        task_name=task_name,
                        state_mode=args.state_mode,
                        rotate_images_180=args.rotate_images_180,
                    )
                    total_episodes += 1
                    total_frames += frame_count

                    if args.max_episodes is not None and total_episodes >= args.max_episodes:
                        break

            if args.max_episodes is not None and total_episodes >= args.max_episodes:
                break
    finally:
        dataset.finalize()

    logging.info(
        "Finished conversion | files=%d | episodes=%d | frames=%d | output=%s",
        total_files,
        total_episodes,
        total_frames,
        root,
    )


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert raw LIBERO HDF5 files to a LeRobot dataset v3.")
    parser.add_argument("--raw-dir", required=True, help="Directory containing raw LIBERO HDF5 files.")
    parser.add_argument("--repo-id", required=True, help="Logical dataset repo id, e.g. local/libero_full.")
    parser.add_argument(
        "--root",
        required=True,
        help="Output directory for the converted LeRobot dataset. Must not already exist.",
    )
    parser.add_argument("--fps", type=int, default=10, help="Dataset FPS. LIBERO is typically 10.")
    parser.add_argument(
        "--state-mode",
        default=STATE_MODE_AUTO,
        choices=[STATE_MODE_AUTO, STATE_MODE_EEF, STATE_MODE_JOINT],
        help="How to build `observation.state` from the raw HDF5 observations.",
    )
    parser.add_argument(
        "--use-videos",
        type=parse_bool,
        default=True,
        help="Store image observations as videos instead of image files.",
    )
    parser.add_argument(
        "--rotate-images-180",
        type=parse_bool,
        default=True,
        help="Rotate raw LIBERO images by 180 degrees to match the official HuggingFaceVLA/libero convention.",
    )
    parser.add_argument(
        "--streaming-encoding",
        type=parse_bool,
        default=True,
        help="Encode videos on the fly while converting. Ignored when --use-videos=false.",
    )
    parser.add_argument(
        "--vcodec",
        default="auto",
        help="Video codec used by LeRobot when --use-videos=true. Examples: auto, h264, hevc, libsvtav1.",
    )
    parser.add_argument(
        "--encoder-threads",
        type=int,
        default=None,
        help="Optional per-encoder thread count when video encoding is enabled.",
    )
    parser.add_argument(
        "--suites",
        default=None,
        help="Optional comma-separated suite filter, e.g. libero_spatial,libero_object",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Optional limit for quick dry runs.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional limit for quick dry runs.")
    return parser


def main() -> None:
    from lerobot.utils.utils import init_logging

    init_logging()
    args = make_argparser().parse_args()
    convert_dataset(args)


if __name__ == "__main__":
    main()

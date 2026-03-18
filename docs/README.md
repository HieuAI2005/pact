# SmolVLA 500M × LIBERO: Hướng Dẫn Đầy Đủ

Hướng dẫn end-to-end: từ chuẩn bị dataset, cấu hình normalization, training SmolVLA 500M trên LIBERO, đến evaluation.

---

## Mục Lục

1. [Tổng Quan Kiến Trúc](#1-tổng-quan-kiến-trúc)
2. [Cài Đặt](#2-cài-đặt)
3. [Chuẩn Bị Dataset](#3-chuẩn-bị-dataset)
4. [Normalization — Hướng Tốt Nhất](#4-normalization--hướng-tốt-nhất)
5. [Training](#5-training)
6. [Evaluation](#6-evaluation)
7. [Tham Khảo Nhanh](#7-tham-khảo-nhanh)

---

## 1. Tổng Quan Kiến Trúc

SmolVLA là model VLA (Vision-Language-Action) sử dụng **Flow Matching** để sinh action chunk.

```
┌──────────────────────────────────┐
│              actions (7-D)       │
│                  ▲               │
│  ┌──────────┐   ┌──┴─────┐      │
│  │          │──►│        │      │
│  │          │kv │ Action │      │
│  │SmolVLM2  │──►│ Expert │      │
│  │  500M    │   │(cross- │      │
│  │(SigLIP+  │──►│ attn)  │      │
│  │  LLM)    │   │        │      │
│  └▲──▲──▲───┘   └───▲────┘      │
│   │  │  │           │           │
│   │  │ state     noise          │
│   │ language      (10 steps)    │
│  images                         │
└──────────────────────────────────┘
```

**Thông số chính:**

| Tham số | Giá trị |
|---------|---------|
| VLM backbone | `SmolVLM2-500M-Video-Instruct` |
| Vision encoder | SigLIP (frozen) |
| Action chunk size | 50 |
| Denoising steps | 10 |
| Image resize | 512×512 (pad giữ aspect ratio) |
| Image normalize | `[0,1] → [-1,1]` (SigLIP range) |
| Max state dim | 32 (auto-pad) |
| Max action dim | 32 (auto-pad) |
| Trainable | Action Expert only (`train_expert_only=True`) |
| Expert width | 0.75× VLM hidden size |

---

## 2. Cài Đặt

```bash
# Clone project
git clone <repo-url> && cd pact

# Install LeRobot + SmolVLA dependencies
pip install -e ".[smolvla]"

# Install LIBERO (nếu cần eval sim)
pip install libero
```

**Yêu cầu GPU**: ≥ 16GB VRAM (RTX 3090, A100, ...)

---

## 3. Chuẩn Bị Dataset

### 3.1 Cấu Trúc HDF5 Hiện Tại

Dataset LIBERO nằm tại `dataset/LIBERO/`:
```
dataset/LIBERO/
├── libero_spatial/
│   ├── pick_up_the_black_bowl_*.hdf5
│   └── ...
├── libero_object/
├── libero_goal/
├── libero_10/
└── libero_90/
```

Mỗi file HDF5:
```
data/
├── demo_0/
│   ├── obs/
│   │   ├── agentview_rgb       (T, 256, 256, 3) uint8
│   │   ├── eye_in_hand_rgb     (T, 256, 256, 3) uint8
│   │   ├── robot0_joint_pos    (T, 7)           float32  [rad]
│   │   └── robot0_gripper_qpos (T, 2)           float32  [cm]
│   ├── actions                 (T, 7)           float32
│   └── dones                   (T,)             bool
├── demo_1/ ...
└── demo_N/
```

### 3.2 Convert HDF5 → LeRobot Dataset Format

LeRobot sử dụng format riêng (Parquet + MP4). Cần convert:

```python
# scripts/convert_libero_to_lerobot.py
"""Convert LIBERO HDF5 → LeRobot dataset format."""
import h5py
import numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Xem doc chi tiết: https://huggingface.co/docs/lerobot/lerobot-dataset-v3

# Cách 1: Sử dụng lerobot built-in converter nếu dataset đã trên Hub
# Cách 2: Push HDF5 data lên Hub rồi dùng LeRobot API
# Cách 3: Tự convert bằng script (xem bên dưới)
```

**Features mapping khi convert:**

| HDF5 key | LeRobot key | Shape | Type |
|----------|-------------|-------|------|
| `obs/agentview_rgb` | `observation.images.image` | (256,256,3) | VISUAL |
| `obs/eye_in_hand_rgb` | `observation.images.image2` | (256,256,3) | VISUAL |
| `obs/robot0_joint_pos` | `observation.state` (dims 0-6) | (7,) | STATE |
| `obs/robot0_gripper_qpos` | `observation.state` (dims 7-8) | (2,) | STATE |
| `actions` | `action` | (7,) | ACTION |

> **Quan trọng:** State = concat(`joint_pos[7]`, `gripper_qpos[2]`) = **9-D vector**.

### 3.3 Verify Dataset

```bash
# Kiểm tra dataset sau khi convert
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('<YOUR_HF_USER>/libero_spatial')
print(f'Episodes: {ds.num_episodes}')
print(f'Frames: {ds.num_frames}')
print(f'Features: {list(ds.meta.features.keys())}')
print(f'Stats keys: {list(ds.meta.stats.keys())}')
"
```

---

## 4. Normalization — Hướng Tốt Nhất

### 4.1 So Sánh Các Phương Án

| Thành phần | SmolVLA mặc định | Cấu hình tùy chỉnh (docs/libero.md) | ✅ Khuyến nghị |
|------------|:----------------:|:------------------------------------:|:--------------:|
| **Action (Δpose, dims 0-5)** | MEAN_STD | Z-score | **MEAN_STD** ✅ |
| **Action (gripper, dim 6)** | MEAN_STD | Passthrough {-1,+1} | **MEAN_STD** ✅ |
| **State (joints, dims 0-6)** | MEAN_STD | Z-score | **MEAN_STD** ✅ |
| **State (fingers, dims 7-8)** | MEAN_STD | Min-Max [-1,1] | **MEAN_STD** ✅ |
| **Images** | IDENTITY* | CLIP mean/std | **IDENTITY** ✅ |

> \* SmolVLA tự xử lý images bên trong `prepare_images()`: resize → 512×512 → normalize `[0,1]→[-1,1]` cho SigLIP.

### 4.2 Tại Sao Dùng MEAN_STD Mặc Định?

**① Gripper binary {-1, +1} vẫn ổn với MEAN_STD:**
- LeRobot tính mean/std **per-dimension**
- Gripper dim 6: mean ≈ 0 (nếu open/close cân bằng), std ≈ 1
- Normalized ≈ {-1, +1} — không bị phóng đại

**② Finger offsets vẫn ổn với MEAN_STD:**
- Dù dải giá trị nhỏ (0.01–0.04 cm), LeRobot tính std per-dim
- Normalized sẽ center quanh 0 với std ≈ 1

**③ Image IDENTITY — SmolVLA tự xử lý:**
- SmolVLA dùng SigLIP (không phải CLIP)
- Cấu hình CLIP mean/std trong `docs/libero.md` dành cho model khác (ACT, Diffusion)
- **KHÔNG áp dụng CLIP normalize cho SmolVLA**

### 4.3 Cấu Hình Normalization (tự động)

SmolVLA đã cài sẵn trong `configuration_smolvla.py`:
```python
normalization_mapping = {
    "VISUAL": NormalizationMode.IDENTITY,   # ảnh: tự xử lý bên trong
    "STATE":  NormalizationMode.MEAN_STD,   # Z-score per-dim
    "ACTION": NormalizationMode.MEAN_STD,   # Z-score per-dim
}
```

**Không cần thay đổi gì** — chỉ cần đảm bảo dataset có statistics đúng.

---

## 5. Training

### 5.1 Train Từ Scratch

```bash
lerobot-train \
  --policy.type=smolvla \
  --dataset.repo_id=<YOUR_HF_USER>/libero_spatial \
  --batch_size=64 \
  --steps=200000 \
  --output_dir=outputs/train/smolvla_libero_spatial \
  --policy.device=cuda \
  --save_freq=10000 \
  --log_freq=100 \
  --seed=42
```

### 5.2 Finetune Từ Pretrained

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=<YOUR_HF_USER>/libero_spatial \
  --batch_size=64 \
  --steps=100000 \
  --output_dir=outputs/train/smolvla_libero_finetuned \
  --policy.device=cuda \
  --save_freq=10000 \
  --log_freq=100 \
  --wandb.enable=true \
  --job_name=smolvla_libero_finetune
```

### 5.3 Train Multi-Suite (tất cả LIBERO)

```bash
# Train trên nhiều suite cùng lúc
lerobot-train \
  --policy.type=smolvla \
  --dataset.repo_id=<YOUR_HF_USER>/libero_all \
  --batch_size=64 \
  --steps=300000 \
  --output_dir=outputs/train/smolvla_libero_all \
  --policy.device=cuda
```

### 5.4 Hyperparameters

| Tham số | Giá trị mặc định | Ghi chú |
|---------|:-----------------:|---------|
| Optimizer | AdamW | |
| Learning rate | 1e-4 | |
| Betas | (0.9, 0.95) | |
| Weight decay | 1e-10 | |
| Grad clip norm | 10 | |
| Warmup steps | 1,000 | |
| Decay steps | 30,000 | |
| Final LR | 2.5e-6 | |
| Scheduler | Cosine + Warmup | |
| Batch size | 64 | Giảm nếu OOM |

**Override hyperparams:**
```bash
lerobot-train \
  --policy.type=smolvla \
  --dataset.repo_id=<DATASET> \
  --policy.optimizer_lr=5e-5 \
  --policy.scheduler_warmup_steps=2000 \
  --policy.scheduler_decay_steps=50000 \
  --batch_size=32 \
  --steps=200000
```

### 5.5 Luồng Dữ Liệu Khi Training

```
HDF5/Parquet Data
      │
      ▼
┌─ Preprocessor Pipeline ────────────────────────┐
│ 1. RenameObservationsProcessorStep             │
│ 2. AddBatchDimensionProcessorStep              │
│ 3. SmolVLANewLineProcessor (thêm \n vào task)  │
│ 4. TokenizerProcessorStep (max_len=48)         │
│ 5. DeviceProcessorStep (→ GPU)                 │
│ 6. NormalizerProcessorStep (MEAN_STD)          │
└────────────────────────────────────────────────┘
      │
      ▼
┌─ Model Forward ────────────────────────────────┐
│ • prepare_images(): resize→512×512, [0,1]→[-1,1]│
│ • prepare_state(): pad 9-D → 32-D              │
│ • prepare_action(): pad 7-D → 32-D             │
│                                                 │
│ Flow Matching Loss:                             │
│   x_t = t·noise + (1-t)·action                 │
│   u_t = noise - action                          │
│   loss = MSE(u_t, v_t)                          │
└─────────────────────────────────────────────────┘
```

---

## 6. Evaluation

### 6.1 Eval trên LIBERO Simulation

```bash
lerobot-eval \
  --policy.path=outputs/train/smolvla_libero_spatial/checkpoints/200000/pretrained_model \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.batch_size=10 \
  --eval.n_episodes=20 \
  --policy.device=cuda \
  --seed=42
```

### 6.2 Eval Multi-Suite

```bash
# Eval trên từng suite riêng
for SUITE in libero_spatial libero_object libero_goal libero_10; do
  lerobot-eval \
    --policy.path=outputs/train/smolvla_libero_all/checkpoints/BEST/pretrained_model \
    --env.type=libero \
    --env.task=$SUITE \
    --eval.batch_size=10 \
    --eval.n_episodes=20 \
    --policy.device=cuda \
    --output_dir=outputs/eval/smolvla_${SUITE}
done
```

### 6.3 Eval Với Video Recording

```bash
lerobot-eval \
  --policy.path=<CHECKPOINT_PATH> \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.batch_size=5 \
  --eval.n_episodes=10 \
  --policy.device=cuda
# Video tự động lưu tại output_dir/videos/
```

### 6.4 LIBERO Environment Config

| Suite | Tasks | Max Steps/Episode |
|-------|:-----:|:-----------------:|
| `libero_spatial` | 10 | 280 |
| `libero_object` | 10 | 280 |
| `libero_goal` | 10 | 300 |
| `libero_10` | 10 | 520 |
| `libero_90` | 90 | 400 |

**Camera Mapping (eval env → policy):**
```
agentview_image         → observation.images.image
robot0_eye_in_hand_image → observation.images.image2
```

**Action Space:** 7-D delta EEF pose (relative mode), gripper dim 6 = {-1=open, +1=close}

---

## 7. Tham Khảo Nhanh

### Cấu Trúc File Quan Trọng

```
src/lerobot/
├── policies/smolvla/
│   ├── configuration_smolvla.py   # Config: normalization, hyperparams
│   ├── modeling_smolvla.py        # Model: VLA + Flow Matching
│   └── processor_smolvla.py       # Pre/post processing pipeline
├── envs/
│   ├── libero.py                  # LIBERO environment wrapper
│   ├── configs.py                 # LiberoEnv config (features, cameras)
│   └── factory.py                 # Env creation factory
├── processor/
│   └── normalize_processor.py     # MEAN_STD, MIN_MAX, QUANTILES logic
└── scripts/
    ├── lerobot_train.py           # Training entrypoint
    └── lerobot_eval.py            # Evaluation entrypoint
```

### Lệnh Thường Dùng

```bash
# Xem help
lerobot-train --help
lerobot-eval --help

# Train SmolVLA
lerobot-train --policy.type=smolvla --dataset.repo_id=<DATASET> --steps=200000

# Finetune từ pretrained
lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=<DATASET>

# Eval
lerobot-eval --policy.path=<CHECKPOINT> --env.type=libero --env.task=libero_spatial

# Xem dataset info
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; ds = LeRobotDataset('<DATASET>'); print(ds)"
```

### Troubleshooting

| Vấn đề | Giải pháp |
|--------|-----------|
| OOM khi train | Giảm `--batch_size` (32 → 16 → 8) |
| EGL error khi eval | `export MUJOCO_GL=egl` hoặc `export MUJOCO_GL=osmesa` |
| LIBERO not found | `pip install libero` và set `LIBERO_ROOT` env var |
| Images sai size | SmolVLA tự resize → 512×512, không cần lo |
| Gripper luôn sai | Kiểm tra dataset stats, đảm bảo gripper dim có mean≈0, std≈1 |

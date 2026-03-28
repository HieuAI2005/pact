# PACT-V: Progress-Aware Chunking and Transition Verification

**Full name:** Progress-Aware Chunking and Transition Verification for Long-Horizon Lightweight Vision-Language-Action Policies
**Target model:** SmolVLA (500M)
**Benchmark:** LIBERO (libero_10 / LIBERO-LONG là primary, libero_spatial và libero_object là secondary)

---

## Vấn đề cần giải quyết

VLA model như SmolVLA sinh ra action theo từng **chunk** (đoạn hành động liên tiếp). Trong các task dài (long-horizon), robot phải thực hiện nhiều **subtask** tuần tự (ví dụ: với tới vật → cầm chặt → di chuyển → đặt xuống).

**Lỗi mục tiêu: False-Positive Transition (FPT)**
FPT xảy ra khi controller quyết định "subtask hiện tại đã xong" và chuyển sang subtask tiếp theo — nhưng thực ra **trạng thái môi trường chưa thỏa mãn** điều kiện hoàn thành (ví dụ: cầm vật chưa chắc nhưng đã chuyển sang phase di chuyển). Một FPT sẽ gây lỗi compounding toàn bộ phần sau.

---

## PACT-V gồm 3 phần chính

### Phần 1: State-Grounded Boundary Supervision (Offline)

Thay vì dùng heuristic (như "gripper đóng lại = grasp xong"), PACT-V dùng **simulator predicates** — các hàm kiểm tra trực tiếp trên trạng thái simulator — để xác định khi nào một subtask **thực sự hoàn thành**.

**Persistent completion time** — lọc bỏ các tiếp xúc thoáng qua (transient contacts):

```
T_k = thời điểm sớm nhất t mà φ_k(s_u) = 1 với mọi u trong [t, t+Δ-1]
```

Từ đó, với mỗi timestep trong trajectory, tính 2 nhãn:
- `s̃_t` (active-stage label): subtask hiện tại đang ở giai đoạn nào
- `b̃_t` (boundary label): có đang trong vùng "sắp hoàn thành subtask" không (trong cửa sổ `w` bước trước `T_k`)

**Failure augmentation:** Ngoài demo thành công, còn thu thập rollout thất bại từ base policy bị đóng băng (frozen). Các rollout thất bại cung cấp **negative example** cho boundary head (dạy model nhận ra "trông có vẻ gần xong nhưng thực ra chưa"). Quan trọng: chỉ dùng observation+state từ rollout thất bại, **không** dùng action của chúng để train imitation.

**Stage-loss mask:** Trong rollout thất bại, sau khi subtask đầu tiên thất bại, stage label trở nên vô nghĩa (robot drift). Mask này loại bỏ stage loss từ timestep đó trở đi, chỉ giữ boundary loss.

---

### Phần 2: Progress-Aware PEFT (Training)

Thêm một **learned token `[PROG]`** vào chuỗi token của backbone. Hidden state tại vị trí token này (`h_t^(p)`) được dùng làm "progress representation".

Một MLP head nhỏ (`D_ψ`) dự đoán 3 giá trị:

| Output | Ý nghĩa |
|--------|---------|
| `ŝ_t` | Phân phối xác suất trên các stages (softmax) |
| `b̂_t` | Xác suất đang gần boundary (sigmoid) |
| `û_t` | Entropy của `ŝ_t` — đo độ không chắc chắn về stage |

**Training loss tổng hợp:**
```
L = L_act (chỉ trên demo thành công)
  + λ_s * CE(ŝ_t, s̃_t) * m_t  (trên cả demo thành công + rollout thất bại)
  + λ_b * BCE(b̂_t, b̃_t)       (trên cả demo thành công + rollout thất bại)
```

Training dùng **LoRA** — chỉ fine-tune một phần nhỏ tham số, không thay đổi kiến trúc action generator.

---

### Phần 3: Verified Replanning by Prefix Execution (Inference)

Base policy luôn sinh chunk có độ dài cố định `L`. PACT-V kiểm soát **bao nhiêu action trong chunk đó được thực thi** trước khi re-observe:

- **Long mode:** thực thi `H_l` action đầu tiên → hiệu quả trong vùng smooth
- **Short mode:** chỉ thực thi `H_s < H_l` action → re-observe thường xuyên hơn gần boundary

**Level 1 — Coarse trigger** (quyết định vào short mode):
```
r_t = [b̂_t > δ_b]  OR  [û_t > η]  OR  [argmax(ŝ_t) thay đổi so với bước trước]
```
Nếu `r_t = 0` và không đang accumulate confirmations → giữ long mode.

**Level 2 — Variance-based acceptance** (quyết định chấp nhận boundary):

Khi ở short mode, sample `M` action chunks độc lập từ cùng một observation:

```
V_t = mean variance của short prefix qua M samples (whitened bởi demo covariance)
```

- `V_t` **cao** → policy không chắc chắn, có thể là boundary giả hoặc state chưa ổn → **giữ short mode**
- `V_t` **thấp** → policy consistent → có thể là genuine boundary

Chỉ **chấp nhận boundary event** sau `C` lần consecutive confirm (cả `r_t=1` và `V_t ≤ τ_V`):

```
q_t = q_{t-1} + 1  nếu r_t=1 và V_t ≤ τ_V
q_t = 0            ngược lại

Khi q_t = C → emit verified boundary event → quay lại long mode
```

**Lưu ý quan trọng:** Verified boundary event chỉ thay đổi **controller state** (quay lại long mode), không inject stage index vào policy. Policy không được conditioned trên stage — nó chỉ được query lại từ observation mới.

---

## Tóm tắt Flow

```
[OFFLINE]
Demo thành công + Rollout thất bại
        ↓
Tính T_k bằng persistent predicates
        ↓
Gán nhãn (s̃_t, b̃_t, m_t) cho từng timestep
        ↓
Train: LoRA + [PROG] token + progress head
       L_act (demo) + L_stage (masked) + L_boundary

[ONLINE INFERENCE]
Observation o_t
        ↓
Backbone + [PROG] → h_t^(p)
        ↓
Progress head → ŝ_t, b̂_t, û_t
        ↓
Trigger? r_t
  NO  → Long mode: thực thi H_l actions
  YES → Short mode:
          Sample M chunks → tính V_t
          V_t ≤ τ_V và r_t=1? → q++
          q = C? → Verified boundary → Long mode
          Còn lại → tiếp tục Short mode
```

---

## Metrics đánh giá

| Metric | Ý nghĩa |
|--------|---------|
| **SR** (Task Success Rate) | Tỷ lệ episode thành công |
| **FPT** (False-Positive Transition Rate) | Tỷ lệ boundary event được chấp nhận sai (chưa thực sự xong subtask) |
| **BR** (Boundary Recall) | Tỷ lệ ground-truth boundary được detect đúng (trong cửa sổ ±ω) |
| **BTR** (Blocked-Trigger Rate) | Tỷ lệ trigger Level-1 bị Level-2 chặn lại — đo độ conservative của controller |
| **Normalized compute** | Số chunk samples so với baseline fixed long-prefix |

---

## Hyperparameters gợi ý

| Param | Phạm vi | Ý nghĩa |
|-------|---------|---------|
| `H_s` | {4, 5, 6} | Short prefix length |
| `H_l` | {16, 20, 24} | Long prefix length |
| `M` | {4, 8, 12} | Số samples để tính variance |
| `C` | {2, 3} | Số lần confirm liên tiếp cần thiết |
| `δ_b`, `η`, `τ_V` | tuned trên validation set | Ngưỡng trigger và variance |

`τ_V` là param nhạy cảm nhất vì phụ thuộc vào action scaling và stochasticity của policy.

---

## So sánh với các phương pháp liên quan

| Phương pháp | Cách tiếp cận | Compute |
|-------------|---------------|---------|
| **PALM** | Progress cues trong subtask | Trung bình |
| **CycleVLA** | VLM verifier + subtask backtracking + MBR decoding | Nặng |
| **PACT-V** | Lightweight progress head + variance-based verification | Nhẹ (không cần external VLM) |

---

## Hạn chế

- **Low variance không đảm bảo state đúng** — model có thể confident nhưng sai. Đó là lý do chỉ dùng variance như rejection signal, kết hợp với re-observation nhiều lần.
- Cần **explicit stage predicates** — dễ trong simulation, khó ngoài thực tế.
- Không có **backtracking** — nếu fail nặng, short-mode replanning không đủ để recover.
- Không inject stage index vào policy → không thể switch sang action distribution chuyên biệt cho từng stage.

---

## Liên quan đến code hiện tại

- [libero.py](../src/lerobot/envs/libero.py): Wrapper môi trường LIBERO, cung cấp `OffScreenRenderEnv`, quản lý init states, camera observations
- [report/main.tex](../report/main.tex): Paper gốc mô tả đầy đủ thuật toán
- Training hiện tại (pact-v1): SmolVLA 500M fine-tuned trên toàn bộ 100 LIBERO tasks (libero_90 + libero_10), chưa implement progress head — đây là baseline cần thiết trước khi add PACT-V

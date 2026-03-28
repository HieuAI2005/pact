# PACT-V Assumption Validation Scripts

Ba script này verify các giả thuyết cốt lõi của PACT-V **trước khi** train bất cứ thứ gì.
Chạy theo thứ tự 1 → 3. Nếu Check 1 fail, paper cần rethink.

---

## Check 1 — FPT có dominant không?

**Câu hỏi:** Trong các episode thất bại, False-Positive Transition (chuyển sai subtask)
chiếm bao nhiêu %? Nếu < 30% thì core assumption của PACT-V có vấn đề.

```bash
cd /home/hiwe/project/pact
source .venv/bin/activate

python check_issue/check1_failure_classification.py \
    --policy_path outputs/train/2026-03-20/12-33-51_smolvla_libero_3090/checkpoints/300000/pretrained_model \
    --suite libero_spatial \
    --n_episodes 20 \
    --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
    --out check_issue/results/check1_libero_spatial.json
```

Nếu checkpoint của bạn đã dùng sẵn key `observation.images.image` / `observation.images.image2`
thì có thể bỏ `--rename_map`.

**Failure categories:**
- `NEVER_GRASPED`: robot không bao giờ satisfy predicate đầu tiên → pure motor execution fail
- `FPT`: robot satisfy k predicates rồi transient satisfy predicate k+1 nhưng không persistent → FPT
- `STALLED`: robot satisfy k predicates rồi stuck, không có transient k+1 → task too hard
- `SUCCESS`: episode thành công

**Interpret:**
- FPT rate > 50% → PACT-V hypothesis supported
- FPT rate < 30% → paper needs rethink

---

## Check 2 — Variance có discriminative không?

**Câu hỏi:** V_t (inter-sample action variance từ M flow-matching samples) có phân biệt
được success state vs failure state không? Nếu hai distribution overlap nhiều → Level-2
gate của PACT-V sẽ không work.

```bash
python check_issue/check2_variance_probe.py \
    --policy_path outputs/train/2026-03-20/12-33-51_smolvla_libero_3090/checkpoints/300000/pretrained_model \
    --suite libero_spatial \
    --n_episodes 20 \
    --M 10 \
    --short_prefix 5 \
    --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
    --out check_issue/results/check2_libero_spatial.json
```

**Interpret:**
- pre_success variance thấp hơn during_failure rõ ràng → variance gate viable
- Hai distribution overlap > 60% → variance gate FAIL, root cause xác nhận

---

## Check 3 — Backbone có encode completion predicates không?

**Câu hỏi:** Nếu train linear probe từ frozen SmolVLA backbone features để predict φ_k(s_t),
accuracy có đạt > 80% không? Nếu không → [PROG] token cũng sẽ không học được,
và đây là root cause chung cho cả hai vấn đề trên.

```bash
python check_issue/check3_linear_probe.py \
    --policy_path outputs/train/2026-03-20/12-33-51_smolvla_libero_3090/checkpoints/300000/pretrained_model \
    --suite libero_spatial \
    --task_ids 0 1 2 \
    --n_episodes 15 \
    --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
    --out check_issue/results/check3_libero_spatial.json
```

**Interpret:**
- Mean accuracy >= 80% → backbone encodes completion, [PROG] token viable
- Mean accuracy 65-80% → weak signal, marginal
- Mean accuracy < 65% → visual compression loại bỏ discriminative cues, cả progress head
  lẫn variance gate đều bị ảnh hưởng bởi cùng root cause

---

## Dependencies

```bash
pip install scikit-learn  # cho check3
```

Các package còn lại đã có trong .venv của project.

---

## Decision tree

```
Check 1: FPT rate?
├── < 30%  → paper core assumption wrong → rethink target failure mode
├── 30-50% → FPT significant but not dominant → scope paper more narrowly
└── > 50%  → FPT is dominant → proceed to Check 2 & 3

Check 3: Linear probe accuracy?
├── < 65%  → backbone cannot encode completion
│            → both progress head AND variance gate will fail
│            → need higher-res visual tokens or different approach
├── 65-80% → partial encoding → proceed but expect modest gains
└── > 80%  → backbone encodes completion → Check 2

Check 2: Variance discriminative?
├── overlap > 60% → variance gate not viable → replace with confidence-based gate
└── overlap < 40% → variance gate viable → proceed with full PACT-V
```

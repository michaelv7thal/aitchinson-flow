# Getting Started: Path A, Path B, and Path C

This guide shows how to run the three supported workflows in this repository:

- Path A: char-level text8 OOD auditor (`K=27`, `L=30` by default)
- Path B: pretrained-LLM top-K OOD auditor (raw text → GPT-2/Qwen → per-position top-K softmax → ILR)
- Path C: byte-level Q+A hallucination auditor (`K=256`, `L=128` by default)

It focuses on practical commands plus the config fields that matter most.

## 1) Install and Sanity Check

Choose one setup style.

### Option A: `uv`

```bash
uv sync
uv sync --extra dev --extra benchmarks
uv run pytest -q
```

### Option B: `venv` + `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,benchmarks]"
pytest -q
```

## 2) Which Path Should I Run?

| Path | Use case | Core scripts |
|---|---|---|
| Path A | text8 corruption / OOD detection on char vocab | `scripts/two_stage_train.py`, `benchmarks/runner.py` |
| Path B | OOD detection on pretrained-LLM top-K distributions | `scripts/two_stage_train.py` (same driver; flip `cfg.training_data.source="llm_topk"`) |
| Path C | Q+A hallucination auditing with byte tokens | `scripts/phase1_train_bytes.py`, `scripts/phase2_train_qa.py`, `scripts/phase2_eval_hallucination.py` |

## 3) Path A (text8 OOD) Quickstart

### Train Stage 1 -> Stage 2 -> fused model

```bash
python scripts/two_stage_train.py \
  --out-dir checkpoints/two_stage/baseline \
  --stage1-epochs 10 \
  --stage2-epochs 5
```

Outputs:

- `checkpoints/two_stage/baseline/stage1.pt`
- `checkpoints/two_stage/baseline/stage2.pt`
- `checkpoints/two_stage/baseline/fused.pt`
- `checkpoints/two_stage/baseline/orchestration.json`

### Optional smoke run

```bash
python scripts/two_stage_train.py --smoke
```

### Optional benchmark sweep (train + evaluate)

```bash
python benchmarks/runner.py
```

Default benchmark output:

- `results/benchmark/benchmark_latest.json`

### Path A config knobs (most relevant)

```python
from dataclasses import replace
from aitchinson_flow.config import Config

cfg = Config()

# Core path identity
cfg.dataset = replace(cfg.dataset, K=27, L=30)
cfg.training_data = replace(cfg.training_data, source="raw_text", raw_dataset="text8")

# Two-stage model selection (used by scripts internally)
cfg.training = replace(cfg.training, model_name="bayesian_auditor_stage1")

# Stage 1 objective family
cfg.training = replace(cfg.training, velocity_loss="soft_hilbert")

# text8 corruption controls
cfg.text8_dataset = replace(
    cfg.text8_dataset,
    train_corrupt_rate=0.15,
    eval_corrupt_rate=0.30,
    max_train_windows=10_000,
    max_eval_windows=5_000,
)

# Optional benchmark controls
cfg.benchmark = replace(
    cfg.benchmark,
    task_name="text_audit",
    data_source="text8",
    train_before_eval=True,
    train_epochs=20,
)
```

## 4) Path B (LLM top-K OOD) Quickstart

Path B reuses Path A's two-stage driver. The only change is the data source: instead of one-hot char rows, each position is the top-K softmax probabilities from a pretrained causal LM, projected through the ILR transform. Invalid batches are produced by Path A's char-level corruption applied *before* LLM inference.

### Train Stage 1 -> Stage 2 -> fused model

```bash
python scripts/two_stage_train.py \
  --out-dir checkpoints/two_stage/path_b_baseline \
  --stage1-epochs 5 \
  --stage2-epochs 3 \
  --training-data-source llm_topk
```

(Add the `--training-data-source` flag or override `cfg.training_data.source="llm_topk"` directly in a wrapper.)

### Path B config knobs (most relevant)

```python
from dataclasses import replace
from aitchinson_flow.config import Config, LLMTopKDatasetConfig

cfg = Config()

# Core path identity: top-K slots drive ILR dim; L is LLM-token length
cfg.dataset = replace(cfg.dataset, K=64, L=32)
cfg.training_data = replace(cfg.training_data, source="llm_topk")

# Pretrained LLM (default: GPT-2 small; fits in <1 GB)
cfg.teacher = replace(
    cfg.teacher,
    model_id="gpt2",          # or "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"
    dtype="float32",
    device="auto",
)

# Path B pipeline knobs
cfg.llm_topk_dataset = LLMTopKDatasetConfig(
    lm_key="hf_causal",
    raw_text_backend="text8",
    char_window_length=256,   # raw chars fed into the LLM tokenizer before truncation
    corrupt_rate=0.15,        # None -> falls back to cfg.text8_dataset.train_corrupt_rate
    generation_seed=0,
)
```

## 5) Path C (byte-level Q+A hallucination) Quickstart

Path C is a 3-step flow and writes to separate checkpoint trees. File names inherit the "phase1_bytes / phase2_qa" naming from when this pipeline was labelled Path B.

### Step 1: train byte-level Stage 1 backbone (`K=256`)

```bash
python scripts/phase1_train_bytes.py \
  --out-dir checkpoints/phase1_bytes/baseline \
  --epochs 10 \
  --L 256
```

### Step 2: train Stage 2 Q+A head on top of that backbone

```bash
python scripts/phase2_train_qa.py \
  --out-dir checkpoints/phase2_qa/baseline \
  --stage1-backbone-ckpt checkpoints/phase1_bytes/baseline/stage1.pt \
  --epochs 10 \
  --L 128
```

### Step 3: evaluate against LLM-generated answers

```bash
python scripts/phase2_eval_hallucination.py \
  --stage2-ckpt checkpoints/phase2_qa/baseline/stage2.pt \
  --answer-model-id gpt2 \
  --max-val-samples 200 \
  --out results/benchmark/hallucination_eval.json
```

### Optional smoke/plumbing mode

Skip LLM instantiation and use placeholder negatives:

```bash
python scripts/phase2_train_qa.py --smoke --skip-llm-eval
python scripts/phase2_eval_hallucination.py \
  --stage2-ckpt checkpoints/phase2_qa/baseline/stage2.pt \
  --skip-llm
```

### Path C config knobs (most relevant)

```python
from dataclasses import replace
from aitchinson_flow.config import Config

cfg = Config()

# Core path identity
cfg.dataset = replace(cfg.dataset, K=256, L=128)
cfg.training_data = replace(cfg.training_data, source="qa_pairs")

# Q+A dataset source and parsing
cfg.qa_dataset = replace(
    cfg.qa_dataset,
    hf_path="trivia_qa",
    name="rc.nocontext",
    split_train="train",
    split_val="validation",
    question_col="question",
    answer_col="answer.value",
    aliases_col="answer.aliases",
    max_question_bytes=96,
    max_answer_bytes=28,
)

# Eval-time answer generator LM
cfg.answer_generator = replace(
    cfg.answer_generator,
    model_id="gpt2",
    prompt_template="Q: {q}\nA:",
    max_new_tokens=24,
    temperature=0.7,
    top_p=0.9,
)

# Restrict GP loss/scoring to answer tokens (recommended for Path C)
cfg.gp = replace(cfg.gp, score_answer_tokens_only=True)

# Useful for smoke tests only
cfg.training_data = replace(cfg.training_data, qa_skip_llm_eval=True)
```

## 6) Common Training Knobs for All Paths

These apply across all workflows:

- `cfg.training.B`, `cfg.training.epochs`, `cfg.training.lr`
- `cfg.training.lr_scheduler`, warmup/cosine settings
- `cfg.training.weight_decay`, `cfg.training.grad_clip_norm`
- `cfg.training.device`
- `cfg.training.wandb_*` fields
- `cfg.gp.num_inducing`, `cfg.gp.lambda_kl`, `cfg.gp.margin_E`

## 7) Typical Artifacts by Path

### Path A / Path B

- `checkpoints/two_stage/<run>/stage1.pt`
- `checkpoints/two_stage/<run>/stage2.pt`
- `checkpoints/two_stage/<run>/fused.pt`
- `checkpoints/two_stage/<run>/orchestration.json`

### Path C

- `checkpoints/phase1_bytes/<run>/stage1.pt`
- `checkpoints/phase2_qa/<run>/stage2.pt`
- `checkpoints/phase2_qa/<run>/fused.pt`
- `checkpoints/phase1_bytes/<run>/phase1_bytes_manifest.json`
- `checkpoints/phase2_qa/<run>/phase2_qa_manifest.json`

## 8) Minimal End-to-End Command Sets

### Path A (single command)

```bash
python scripts/two_stage_train.py --out-dir checkpoints/two_stage/quick --stage1-epochs 5 --stage2-epochs 3
```

### Path B (single command, same driver)

```bash
python scripts/two_stage_train.py \
  --out-dir checkpoints/two_stage/path_b_quick \
  --stage1-epochs 5 --stage2-epochs 3 \
  --training-data-source llm_topk
```

### Path C (three commands)

```bash
python scripts/phase1_train_bytes.py --out-dir checkpoints/phase1_bytes/quick --epochs 5 --L 256
python scripts/phase2_train_qa.py --out-dir checkpoints/phase2_qa/quick --stage1-backbone-ckpt checkpoints/phase1_bytes/quick/stage1.pt --epochs 5 --L 128
python scripts/phase2_eval_hallucination.py --stage2-ckpt checkpoints/phase2_qa/quick/stage2.pt --answer-model-id gpt2 --max-val-samples 100
```

---

If you need architecture context while running these commands, see:

- `README.md`
- `ARCHITECTURE.md`

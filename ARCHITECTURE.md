# Architecture Overview

This document describes how the major components of the `aitchinson_flow` codebase fit together.

---

## High-level picture

```
Config
  │
  ├─► LM (HFCausalLMInference)
  │       │
  │       ▼
  │   CausalLMTeacher  ──► CausalLMTeacherDataModule
  │                                │
  │                                ▼ batches: {log_x, token_ids?, logits?}
  │
  ├─► build_model(cfg)  ──► fit(cfg, datamodule, model)  ──► trained model
  │                                                               │
  └─► BenchmarkTask.run(model, datamodule, cfg)  ◄───────────────┘
            │
            ▼
      results: dict[scale_tag, dict[metric, float]]
```

---

## 1. Config

**File:** [`src/aitchinson_flow/config.py`](src/aitchinson_flow/config.py)

All configuration lives in a tree of plain `@dataclass` objects rooted at `Config`:

| Sub-config | Purpose |
|---|---|
| `DatasetConfig` | Vocabulary size `K`, sequence length `L`, dataset name |
| `TransformerConfig` | `d_model`, `nhead`, `num_layers`, `d_latent`, dropout |
| `TrainingConfig` | `model_name`, batch size `B`, epochs, LR, scheduler, device, `use_tqdm` |
| `TrainingDataConfig` | Data-source selection: `raw_text`, `llm_generated`, or `qa_pairs` (Path B) |
| `BenchmarkConfig` | Scale grid, task name, LM key, `n_batches`, corruption, `train_before_eval`, `train_epochs` |
| `GPConfig` | Inducing point count, KL/variance margins; `score_answer_tokens_only` (Path B) |
| `TeacherConfig` | HF model ID, dtype, device, caching for the teacher LM |
| `HFDatasetConfig` | Hub paths, splits, streaming settings |
| `QADatasetConfig` | **Path B** Q+A HF source (default `trivia_qa/rc.nocontext`), column paths, byte caps |
| `AnswerGeneratorConfig` | **Path B** eval-time LLM (model id, prompt template, sampling knobs, answer cache) |
| `EquilibriumFlowConfig` | Interpolation endpoints, ODE generation steps |

`Config()` with no arguments gives working defaults for every field. The default `K=27`/`L=30` targets the Path A text8 task; Path B drivers override `K=256` and `L=128` at construction time.
Scale sweeps call `apply_transformer_scale(cfg, ...)` which `deepcopy`s and `replace`s the relevant fields without mutating the original.

---

## 2. Registries

Three independent registries follow the same pattern — a dict populated by `@register("key")` decorators on factory functions, and a `build_X(key, cfg)` lookup function.

| Registry | File | Key type |
|---|---|---|
| **Models** | [`src/aitchinson_flow/models/factory.py`](src/aitchinson_flow/models/factory.py) | `cfg.training.model_name` |
| **Tasks** | [`benchmarks/tasks/registry.py`](benchmarks/tasks/registry.py) | `cfg.benchmark.task_name` |
| **LMs** | [`src/aitchinson_flow/llms/registry.py`](src/aitchinson_flow/llms/registry.py) | `cfg.benchmark.lm_key` |

Registries are populated as a side-effect of importing the relevant module (e.g. `import aitchinson_flow.models` at the top of the benchmark runner). Any new model, task, or LM only needs to call `@register("name")` on its factory function to be discoverable.

---

## 3. Data pipeline

**Single entrypoint:** [`src/aitchinson_flow/training/data_sources.py::build_training_datamodule`](src/aitchinson_flow/training/data_sources.py). Training scripts read `cfg.training_data.source` and dispatch here — they do not branch on source kind themselves.

| `cfg.training_data.source` | Path | DataModule |
|---|---|---|
| `"raw_text"` | A | `Text8DataModule` (K=27) or `HFDataModule` |
| `"llm_generated"` | A | `CausalLMTeacherDataModule` (teacher LM streams batches) |
| `"qa_pairs"` | B | `QAPairsDataModule` (byte-encoded `[Q][A]` pairs) |

### Path A — text corruption / LLM teacher

**Files:**
- [`src/aitchinson_flow/data/teachers/causal_lm.py`](src/aitchinson_flow/data/teachers/causal_lm.py)
- [`src/aitchinson_flow/training/datamodule.py`](src/aitchinson_flow/training/datamodule.py)

```
HFCausalLMInference
  │  .generate_ids(batch_size, max_new_tokens, prompt_ids?)
  │  .forward_logits(input_ids)
  │  .decode(ids, skip_special_tokens=True)   ← used by Path B eval
  ▼
CausalLMTeacher
  │  .sample_log_x()           → {log_x: (B,L,K)}
  │  .sample_with_logits()     → {log_x, token_ids, logits}
  ▼
CausalLMTeacherDataset   (IterableDataset, streams n_batches batches)
  ▼
CausalLMTeacherDataModule  implements DataModule protocol
  │  .train_dataloader()       → DataLoader (batch_size=None)
  │  .val_dataloader()         → None
  └  .test_dataloader()        → None
```

Each `__iter__` of the dataset calls the teacher to generate a fresh batch on the fly. `emit_logits=True` (set when `compute_spilled_energy=True`) switches from `sample_log_x` to `sample_with_logits`, adding `token_ids` and raw `logits` to every batch. Those extra fields are what the spilled-energy computation requires.

**Text8 prompts:** when `use_text8_prompts=True`, [`benchmarks/text8_prompts.py`](benchmarks/text8_prompts.py) loads real text8 chunks, tokenizes them, and passes them to the teacher as conditioning prompts so generated sequences are grounded in natural-language context rather than BOS tokens.

### Path B — byte-level Q+A

**Files:**
- [`src/aitchinson_flow/data/byte_vocab.py`](src/aitchinson_flow/data/byte_vocab.py) — UTF-8 codec + role markers (`STX=0x02`, `ETX=0x03`, `EOT=0x04`, `PAD=0x00`).
- [`src/aitchinson_flow/data/bytes_datamodule.py`](src/aitchinson_flow/data/bytes_datamodule.py) — byte windows for Phase 1 backbone retraining at `K=256`.
- [`src/aitchinson_flow/data/qa_datamodule.py`](src/aitchinson_flow/data/qa_datamodule.py) — `[STX] question [ETX] answer [EOT]` padded to `cfg.dataset.L`.
- [`src/aitchinson_flow/data/qa_negatives.py`](src/aitchinson_flow/data/qa_negatives.py) — cross-question-swap in-batch negatives.

Train batches emit `{log_x, token_ids, answer_mask, log_x_invalid}`; eval batches interleave correct (`label=0`) and LLM-generated (`label=1`) answers for the same question. LLM answers are pre-generated at `setup()` and cached under `cfg.answer_generator.cache_dir` so repeat eval runs skip regeneration.

---

## 4. Models

**Base protocols:** [`src/aitchinson_flow/models/base.py`](src/aitchinson_flow/models/base.py)

```
GenerativeTrainingModel (Protocol)
  training_step(batch, step) → LossDict   # must include key "loss"
  eval_step(batch)           → LossDict

AuditorModel(GenerativeTrainingModel)
  audit(batch)               → LossDict   # inference-time scoring
```

Registered concrete models:

| Key | File | What it does |
|---|---|---|
| `flow_matching` | [`models/flow_matching.py`](src/aitchinson_flow/models/flow_matching.py) | Linear interpolation flow on the log-simplex. `training_step` supervises a velocity field; `audit` returns MSE velocity error. |
| `bayesian_generator` | [`models/bayesian_generator.py`](src/aitchinson_flow/models/bayesian_generator.py) | TransformerBackbone → latent `z` → SparseGP energy. Velocity = −∇E. Adds KL and volume penalties. |
| `bayesian_auditor` | [`models/bayesian_auditor.py`](src/aitchinson_flow/models/bayesian_auditor.py) | Extends `bayesian_generator` with contrastive valid/invalid hinge losses. Uses GP variance as OOD score. |
| `bayesian_auditor_stage1` | [`models/bayesian_auditor_stage1.py`](src/aitchinson_flow/models/bayesian_auditor_stage1.py) | Stage 1 EqM + Hilbert-family training on valid-only data. Learns the backbone that Stage 2 freezes. |
| `bayesian_auditor_stage2` | [`models/bayesian_auditor_stage2.py`](src/aitchinson_flow/models/bayesian_auditor_stage2.py) | Frozen backbone + trainable `latent_head` + trainable per-token GP. Reads optional `log_x_invalid` / `answer_mask` (Path B) and falls back to `randn_like` negatives if absent (Path A). |
| `per_token_bayesian_auditor` | [`models/per_token_bayesian_auditor.py`](src/aitchinson_flow/models/per_token_bayesian_auditor.py) | Maintains a separate GP per sequence position; optionally conditions on teacher hidden states. |

All models consume `batch["log_x"]` as the primary input. Bayesian auditor variants also expect `batch["log_x_invalid"]` (corrupted samples) during training. For Path B, `BayesianAuditorStage2` additionally reads `batch["answer_mask"]`: when `cfg.gp.score_answer_tokens_only=True`, the per-token NLL and contrastive loss are gathered only at answer-span positions. Missing keys fall back to the Path A code path unchanged.

---

## 5. Training runner

**Files:**
- [`src/aitchinson_flow/training/runner.py`](src/aitchinson_flow/training/runner.py)
- [`src/aitchinson_flow/training/loops.py`](src/aitchinson_flow/training/loops.py)

```
fit(cfg, datamodule, model=None, resume_from=None) → nn.Module
  │
  ├─ build_model(cfg)  if model not provided
  ├─ build_optimizer(model, cfg)
  ├─ build_scheduler(optimizer, cfg)
  ├─ optional: load_checkpoint(resume_from, ...)
  │
  └─ epoch loop  [tqdm: "epochs", leave=True]
       │
       ├─ train_epoch(model, train_loader, optimizer, ...)
       │    └─ batch loop  [tqdm: "train epoch N", leave=False]
       │         model.training_step(batch, step) → loss → backward → step
       │         running_average(agg, counts, out)
       │
       ├─ evaluate(model, val_loader, ...)   every eval_every epochs
       │    └─ batch loop  [tqdm: "validation", leave=False]
       │         model.eval_step(batch) → metrics
       │
       ├─ epoch_pbar.set_postfix(train=..., val=..., lr=...)
       ├─ scheduler.step()
       └─ save_checkpoint(...)   every checkpoint_every epochs
```

**tqdm nesting:** during a run you see three live bars — epochs (persistent) above train-batch (transient) and optionally val-batch (transient). The epoch bar's postfix updates with the epoch-average train loss, val loss, and current LR.

Metric accumulation uses [`src/aitchinson_flow/training/metrics.py`](src/aitchinson_flow/training/metrics.py): `running_average` keeps a running sum + count dict; `finalize_averages` divides at the end of each loop.

---

## 6. Benchmark runner

**Files:**
- [`benchmarks/runner.py`](benchmarks/runner.py)
- [`benchmarks/tasks/text_audit.py`](benchmarks/tasks/text_audit.py)
- [`benchmarks/corruption.py`](benchmarks/corruption.py)

```
run_benchmark(cfg) → dict[scale_tag, dict[metric, float]]
  │
  ├─ build_lm(cfg.benchmark.lm_key, cfg.teacher)
  ├─ CausalLMTeacher(lm, cfg)
  ├─ optional: load_text8_prompts(lm, n_prompts, prompt_length)
  ├─ CausalLMTeacherDataModule(teacher, cfg, emit_logits=compute_spilled_energy)
  ├─ build_task(cfg.benchmark.task_name)
  │
  └─ scale loop  [tqdm: "benchmark scales"]
       │
       ├─ apply_transformer_scale(cfg, d_model, num_layers, nhead)
       ├─ override epochs if benchmark.train_epochs is set
       ├─ build_model(scfg)
       ├─ fit(scfg, datamodule, model)   if train_before_eval=True
       └─ task.run(model, datamodule, scfg) → dict[metric, float]
```

`TextAuditTask.run` iterates the dataloader once in `model.eval()` mode:
1. Calls `build_invalid_batch` on each batch to generate corrupted counterparts.
2. Calls `model.audit(batch)` — the trained model's detection metrics.
3. If `compute_spilled_energy`, calls `_spilled_metrics` for the training-free baseline.
4. All metrics are accumulated via `running_average` and returned as floats.

The result dict has two non-overlapping metric namespaces, so trained-model and spilled-energy scores can be directly compared:

```
results["d_model128_nhead8_num_layers4"] = {
    # model metrics (from audit())
    "loss":            0.31,
    "velocity_loss":   0.31,

    # spilled-energy baseline (training-free)
    "spilled_mean_valid":    -4.21,
    "spilled_mean_invalid":  -3.87,
    "spilled_separation":     0.34,
    "spilled_anomaly_valid":  4.21,
    "spilled_anomaly_invalid": 3.87,
    "spilled_token_mean":    -4.21,
    "spilled_token_std":      0.62,
}
```

---

## 7. Spilled energy

**File:** [`src/aitchinson_flow/metrics/spilled_energy.py`](src/aitchinson_flow/metrics/spilled_energy.py)

**Reference:** Minut, Dewidar & Masi, "Spilled Energy in Large Language Models", ICLR 2026.

The per-token spilled energy measures how much probability mass the LM "wasted" on tokens it did not actually predict:

```
ΔE(x_i) = −logsumexp(logits[i−1])   ← marginal energy at step i−1
         + logits[i−1, token_id[i]]  ← logit of the token that actually followed
```

A low (more negative) value means the LM confidently placed mass on the correct token — normal behaviour. A high (less negative) value means the actual token was assigned low probability — anomalous.

The sequence-level anomaly score is `−mean(ΔE)`: higher = more anomalous.

In the benchmark, spilled energy is compared against the trained auditor's own anomaly scores to evaluate whether learned detection adds value beyond this training-free baseline.

---

## 8. Entry points

| Script | Path | Purpose |
|---|---|---|
| `benchmarks/runner.py` | A | Scale sweep: train each model variant and run the audit benchmark. Results saved to `cfg.benchmark.results_dir/benchmark_latest.json`. |
| `scripts/scale_sweep.py` | A | Lighter training-only sweep (calls `fit` without a benchmark task). |
| `scripts/two_stage_train.py` | A | Stage 1 → Stage 2 → compose for text8 (`K=27`). Produces `stage1.pt`/`stage2.pt`/`fused.pt` under `--out-dir`. |
| `scripts/phase1_train_bytes.py` | B | Train the byte-level Stage 1 backbone at `K=256` on a raw-text corpus. Writes a separate checkpoint tree (default `checkpoints/phase1_bytes/`) so Path A artifacts are untouched. |
| `scripts/phase2_train_qa.py` | B | Load the byte Stage 1 backbone, train Stage 2 on `QAPairsDataModule` with cross-question-swap negatives and answer-span scoring. |
| `scripts/phase2_eval_hallucination.py` | B | Instantiate the configured answer-generator LM, score `[Q][A_llm]` pairs, report AUROC via `HallucinationAuditTask`. |

All accept a `Config()` built with defaults. Override fields directly or pass `--config-out path.json` to the benchmark runner to dump the resolved config for inspection. The Path B drivers import Path A helpers (`_load_backbone_into_stage2`, `_init_inducing_from_data`) from `scripts/two_stage_train.py` rather than duplicating them.

---

## 9. Path A vs Path B at a glance

Two trained models, one shared stack. Selection is config-driven — nothing branches on a global task flag.

| Axis | Path A (text8 OOD) | Path B (Q+A hallucination) |
|---|---|---|
| Vocabulary `K` | 27 (char-level a–z + space) | 256 (UTF-8 bytes) |
| Sequence length `L` | 30 (default) | 128 (default) |
| Training data | Text corpus windows + corruption, or teacher LM streams | `[STX] question [ETX] answer [EOT]` from an HF Q+A dataset |
| Train negatives | `build_invalid_batch` corruption / order-mix | In-batch cross-question-swap |
| Eval negatives | Corrupted variants | LLM-generated answers, decoded → UTF-8 → re-encoded as bytes |
| GP scoring span | All tokens | Answer span only (when `gp.score_answer_tokens_only=True`) |
| Phase 1 checkpoint | `checkpoints/two_stage/.../stage1.pt` | `checkpoints/phase1_bytes/.../stage1.pt` |
| Phase 2 checkpoint | `checkpoints/two_stage/.../stage2.pt` | `checkpoints/phase2_qa/.../stage2.pt` |
| Benchmark task | `text_audit`, `trivia_audit` | `hallucination_audit` |
| Driver | `scripts/two_stage_train.py` | `scripts/phase1_train_bytes.py` + `scripts/phase2_train_qa.py` + `scripts/phase2_eval_hallucination.py` |

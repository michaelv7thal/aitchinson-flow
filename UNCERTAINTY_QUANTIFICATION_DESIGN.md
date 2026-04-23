# Uncertainty Quantification Design & Implementation Plan

## Overview

This document specifies the three-component uncertainty quantification (UQ) system for discrete token sequences, the benchmarking strategy, and the OOD healing roadmap. It is intended as both a design reference and an implementation guide.

---

## System Goals

| Goal | Description |
|------|-------------|
| **Per-token UQ** | Assign energy + variance to each token position; low = confident, high = uncertain |
| **Two sources of uncertainty** | Structural (geometric validity) and contextual (semantic plausibility) |
| **Lightweight consistency check** | Spilled energy as a training-free internal-coherence signal |
| **Calibrated benchmarks** | Evaluate all three signals on text8, trivia QA, and domain-specific corpora |
| **OOD healing** | Use UQ signals to guide post-hoc repair of out-of-distribution sequences |

---

## Component 1 — Structural Geometry UQ

### What it measures

Epistemic uncertainty of the raw token geometry: does this sequence look structurally valid at the character/token level, independent of meaning? Examples:
- Structural validity of a DNA nucleotide sequence
- Correct spelling of a medication name
- Valid syntax of a code token stream

### Pipeline

```
Raw text (token ids)
        │
        ▼
  One-hot encode                      shape: (B, L, K)
        │
        ▼ token_ids_to_features()
  ILR transform                       shape: (B, L, K−1)   [Aitchison space]
        │
        ▼
  Stage 1 — Equilibrium Flow Matching
  · Learns velocity field: uniform noise → valid-only data manifold
  · Hilbert / soft-Hilbert distance loss on log-simplex
  · Output: per-token latent h ∈ R^{d_latent}
        │
        ▼
  Stage 2 — Sparse GP (SVGP, Matérn 5/2)
  · Inputs: per-token latents from frozen Stage 1 backbone
  · Outputs per token:
      mean   → energy score  (low = in-distribution)
      variance → epistemic uncertainty
  · Trained contrastively: valid sequences vs structurally corrupted sequences
```

### Signals produced

| Signal | Interpretation |
|--------|---------------|
| Low energy, low variance | Structurally certain token |
| Low energy, high variance | Structurally plausible but the model is unsure |
| High energy, low variance | Confidently out-of-distribution |
| High energy, high variance | Both uncertain and anomalous |

### Existing implementation

| File | Role |
|------|------|
| [src/aitchinson_flow/geometry.py](src/aitchinson_flow/geometry.py) | `ilr()`, `ilr_inv()`, Hilbert distances |
| [src/aitchinson_flow/data/transforms/discrete.py](src/aitchinson_flow/data/transforms/discrete.py) | One-hot → label-smooth → ILR |
| [src/aitchinson_flow/models/bayesian_auditor_stage1.py](src/aitchinson_flow/models/bayesian_auditor_stage1.py) | Equilibrium flow + backbone; `energy_score()`, `ood_score()` |
| [src/aitchinson_flow/models/bayesian_auditor_stage2.py](src/aitchinson_flow/models/bayesian_auditor_stage2.py) | Frozen backbone + GP contrastive training |
| [src/aitchinson_flow/gp/gp.py](src/aitchinson_flow/gp/gp.py) | `SparseGP`: Matérn 5/2, inducing variables, KL |
| [src/aitchinson_flow/data/text8_datamodule.py](src/aitchinson_flow/data/text8_datamodule.py) | Path A data (char-level, K=27) |
| [src/aitchinson_flow/data/corruption.py](src/aitchinson_flow/data/corruption.py) | Structural negatives (swap, drop, insert) |

### Configuration knobs

```python
DatasetConfig(K=27, L=30, dataset="text8")            # char-level
TrainingConfig(velocity_loss="soft_hilbert")            # Stage 1 geometry
GPConfig(num_inducing=64, score_answer_tokens_only=False)
EquilibriumFlowConfig(ode_steps=20)
```

---

## Component 2 — Contextual Semantic UQ

### What it measures

Epistemic uncertainty of the autoregressive next-token probability distribution: does the LLM treat this context as plausible? Examples:
- High confidence (low energy): common English trigrams, expected factual continuations
- Low confidence (high energy): non-sequitur continuations, factually implausible answer tokens

### Key insight

Valid/expected token sequences produce an exponential-decay-like top-K distribution (one token dominates). Invalid or surprising sequences produce a flatter, more uniform top-K distribution. The ILR transform makes this geometric difference learnable.

### Pipeline

```
Raw text → LLM tokenizer
        │
        ▼ CausalLMTeacher (frozen)
  Top-K softmax probabilities           shape: (B, L, K)   K = vocab slots retained
        │
        ▼ re-normalize within K slots
  Probability simplex                   shape: (B, L, K)   sums to 1 per position
        │
        ▼ ILR transform
  Aitchison coordinates                 shape: (B, L, K−1)
        │
        ▼
  Stage 1 — Equilibrium Flow Matching
  · Learns manifold of "plausible" top-K distributions
  · Hilbert loss on ILR-projected probability simplex
        │
        ▼
  Stage 2 — Sparse GP (SVGP, Matérn 5/2)
  · Contrastive: plausible contexts vs corrupted/invalid contexts
  · Outputs per token:
      mean   → semantic energy  (low = plausible, high = surprising)
      variance → epistemic uncertainty about plausibility
```

### Signals produced

| Pattern | Interpretation |
|---------|---------------|
| Exponential-decay top-K, low energy | LLM finds this token predictable/expected |
| Flat/uniform top-K, high energy | LLM is uncertain — surprising or invalid context |
| High variance | Model uncertain whether this distribution is plausible |

### Existing implementation

| File | Role |
|------|------|
| [src/aitchinson_flow/data/teachers/causal_lm.py](src/aitchinson_flow/data/teachers/causal_lm.py) | Frozen LLM, on-the-fly top-K logits |
| [src/aitchinson_flow/data/llm_embedding_datamodule.py](src/aitchinson_flow/data/llm_embedding_datamodule.py) | Tokenize → frozen embedding lookup (Path B) |
| [src/aitchinson_flow/models/llm_projection.py](src/aitchinson_flow/models/llm_projection.py) | `TokenEmbeddingToSimplex`: LLM embedding → K-dim simplex |

### Gap: top-K probability data path

The existing Path B pipeline uses frozen LLM **embeddings** (hidden states). Component 2 requires extracting the **softmax probability distribution** over vocabulary and selecting the top-K entries. The required additions are:

1. **`data/teachers/causal_lm.py`** — add `top_k_probs(input_ids, K)` method that returns re-normalized top-K probabilities (shape `B×L×K`) with their corresponding token indices
2. **`data/llm_topk_datamodule.py`** — new datamodule that calls `top_k_probs()`, re-normalizes, and applies ILR transform before batching
3. **`config.py`** — new `LLMTopKProbsConfig` (distinct from existing `LLMTopKDatasetConfig` which operates on embeddings)
4. **`models/`** — Stage 1 and Stage 2 already work for any `(B, L, K−1)` input; no model changes needed

### Configuration knobs

```python
LLMTopKProbsConfig(
    model_id="gpt2",        # or any HF causal LM
    K=50,                   # top-K vocabulary slots to retain
    renormalize=True,       # re-normalize after top-K truncation
)
DatasetConfig(K=50, L=64)
```

---

## Component 3 — Spilled Energy (Autoregressive Consistency)

### What it measures

Internal consistency of the autoregressive token distribution. Theoretically equals zero for a perfectly calibrated LM on in-distribution text. Positive deviation indicates unexpected tokens in the sequence.

**Definition (per-token):**
```
ΔE(x_i) = −logsumexp(logits[i]) + logits[i, token_id[i+1]]
```

**Sequence anomaly score:**
```
anomaly(x) = −mean_i(ΔE(x_i))
```

Low (negative) spilled energy = expected continuation; high (positive) = surprising/anomalous.

### Properties

- **Training-free**: requires only a frozen LLM; no GP or flow matching
- **No uncertainty estimate**: produces a scalar anomaly score per token/sequence only
- **Complementary**: catches distributional drift that energy/variance miss; vice versa

### Existing implementation

| File | Role |
|------|------|
| [src/aitchinson_flow/metrics/spilled_energy.py](src/aitchinson_flow/metrics/spilled_energy.py) | `marginal_energy()`, `spilled_energy()`, per-token and sequence-level scores |

### Reference

Spilled energy formulation from ICLR 2026 (internal consistency of autoregressive sequence models).

---

## Joint Signal Summary

At inference, each token position receives up to five signals:

| Signal | Source | Has UQ? |
|--------|--------|---------|
| Structural energy | Component 1 GP mean | Yes (GP variance) |
| Structural variance | Component 1 GP variance | — |
| Contextual energy | Component 2 GP mean | Yes (GP variance) |
| Contextual variance | Component 2 GP variance | — |
| Spilled energy | Component 3 (training-free) | No |

These can be combined as a weighted score or used individually per-task.

---

## Architecture Diagram

```
                     ┌──────────────────────────────────────┐
                     │         Input Token Sequence          │
                     │         (raw text / token IDs)        │
                     └────────────┬─────────────────────────┘
                                  │
              ┌───────────────────┼──────────────────────┐
              │                   │                      │
              ▼                   ▼                      ▼
   ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │  Component 1:   │  │  Component 2:    │  │  Component 3:    │
   │  Structural UQ  │  │  Contextual UQ   │  │  Spilled Energy  │
   │                 │  │                  │  │  (training-free) │
   │ one-hot → ILR   │  │ Top-K probs →    │  │                  │
   │ → EqM Flow      │  │ re-norm → ILR    │  │ ΔE = -logsumexp  │
   │ → Sparse GP     │  │ → EqM Flow       │  │     + logit[t+1] │
   │                 │  │ → Sparse GP      │  │                  │
   │ Outputs:        │  │ Outputs:         │  │ Output:          │
   │  energy (L,)    │  │  energy (L,)     │  │  ΔE (L,)         │
   │  variance (L,)  │  │  variance (L,)   │  │  anomaly (1,)    │
   └─────────────────┘  └──────────────────┘  └──────────────────┘
              │                   │                      │
              └───────────────────┴──────────────────────┘
                                  │
                     ┌────────────▼────────────┐
                     │   Unified UQ Report     │
                     │   per token position    │
                     │   (5 signals total)     │
                     └─────────────────────────┘
```

---

## Benchmarking Strategy

### Benchmark Tasks

#### Task 1 — Basic English Language (text8)

| Property | Value |
|----------|-------|
| Vocabulary | K=27 (a–z + space) |
| Sequence length | L=30 |
| Valid | Contiguous windows from text8 corpus |
| Invalid | Character-level corruptions (swap, drop, insert, replace) |
| Goal | Validate Component 1; establish energy/variance separation |

**Expected behavior:**
- Valid windows → low energy, low variance
- Corrupted windows → high energy, rising variance at corruption sites
- Spilled energy: invalid windows deviate positively

#### Task 2 — Trivia QA with Ground Truth / Incorrect Answers

| Property | Value |
|----------|-------|
| Format | `[STX] Question [ETX] Answer [EOT]` byte-encoded |
| Valid | Correct answers from `trivia_qa/rc.nocontext` |
| Invalid | Cross-question-swap (factually plausible but wrong) or LLM-generated hallucinations |
| Goal | Validate Component 2; detect hallucinated answer spans |

**Expected behavior:**
- Correct `[Q, A]` → low contextual energy on answer span
- Wrong/swapped answers → high contextual energy and high variance on answer tokens
- Spilled energy: detects surprising token choices in answer

#### Task 3 — Domain-Specific: DNA Sequences

| Property | Value |
|----------|-------|
| Vocabulary | K=4 (A, C, G, T) or K=5 with padding |
| Valid | Real genomic sequences (e.g., RefSeq, Ensembl) |
| Invalid | Random shuffled sequences, point mutations, frameshifts |
| Goal | Validate Component 1 on biologically-constrained vocabularies |

**Expected behavior:**
- Valid sequences → low structural energy (CpG patterns, codon structure captured)
- Random shuffles → high energy, high variance
- Point mutations → elevated energy at mutated position only

#### Task 4 — Domain-Specific: Medical / Clinical Text

| Property | Value |
|----------|-------|
| Vocabulary | Char-level or BPE subword |
| Valid | Medical literature, drug names, clinical notes (e.g., MIMIC-III notes) |
| Invalid | Misspelled drug names, invalid dosage patterns, non-medical insertions |
| Goal | Validate Components 1 + 2 jointly on high-stakes domain |

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **AUROC (energy)** | Separability of valid/invalid using GP energy score |
| **AUROC (variance)** | Separability using GP variance alone |
| **AUROC (combined)** | Energy + variance joint score |
| **AUROC (spilled)** | Spilled energy separability as training-free baseline |
| **Energy gap** | `mean_energy(invalid) − mean_energy(valid)` |
| **Variance ratio** | `mean_var(invalid) / mean_var(valid)` |
| **Per-token ROC** | Token-level AUROC at corruption sites vs clean positions |

### Benchmark Configuration Grid

```python
BenchmarkConfig(
    scale_grid=[
        # (d_model, nhead, num_layers)
        (64, 4, 2),
        (128, 8, 4),
        (256, 8, 6),
    ],
    tasks=["text8_audit", "trivia_audit", "dna_audit", "medical_audit"],
    train_epochs=20,
    train_before_eval=True,
)
```

### Benchmark Output Schema

```json
{
  "component": "structural | contextual | spilled",
  "task": "text8 | trivia | dna | medical",
  "scale": "d_model=128_nhead=8_num_layers=4",
  "auroc_energy": 0.92,
  "auroc_variance": 0.87,
  "auroc_combined": 0.94,
  "auroc_spilled": 0.81,
  "energy_gap": 3.41,
  "variance_ratio": 4.2,
  "per_token_auroc_at_corruption": 0.89
}
```

---

## OOD Healing (Post-Processing)

Healing is enabled once the UQ system is validated. It uses the energy and variance maps to identify and repair high-uncertainty token positions.

### Healing Strategy 1 — Beam Search Re-ranking

Use the joint UQ signal as a reranking score for beam search candidates:

```
score(candidate) = log_p(candidate | context) − λ · mean_energy(candidate)
```

Candidates with high energy are penalized; low-energy completions are preferred.

### Healing Strategy 2 — Targeted Re-sampling

1. Run forward pass → identify token positions where `energy[t] > threshold`
2. Mask those positions
3. Re-sample masked positions conditioned on surrounding context (using the LLM or a masked LM)
4. Re-evaluate UQ → iterate until energy drops below threshold or max iterations reached

### Healing Strategy 3 — Simplex Projection Repair

For Component 1 (structural), use the inverse ILR + flow matching to project the anomalous token's latent representation back onto the learned valid manifold. Recover the nearest valid token via argmax of the projected simplex.

### Evaluation of Healing

| Metric | Description |
|--------|-------------|
| **Heal AUROC** | AUROC of energy before vs after healing (should increase) |
| **Token accuracy** | For trivia QA: fraction of healed tokens that match ground truth |
| **Energy reduction** | `mean_energy(before) − mean_energy(after)` |
| **Variance reduction** | `mean_var(before) − mean_var(after)` |

Healing is tracked in [benchmarks/tasks/healing_audit.py](benchmarks/tasks/healing_audit.py).

---

## Implementation Plan

### Phase 0 — Existing Foundation (Complete)

- [x] ILR/CLR geometry (`geometry.py`)
- [x] Equilibrium flow matching (Stage 1 backbone)
- [x] Sparse GP Stage 2 contrastive training
- [x] Spilled energy metric
- [x] text8 and trivia QA data pipelines
- [x] Two-stage training script
- [x] Benchmark runner with AUROC metrics

### Phase 1 — Component 2: Top-K Probability Path

**Goal:** Extract and pipeline top-K softmax probabilities from a frozen LLM into ILR space.

**Tasks:**

| Task | File | Description |
|------|------|-------------|
| 1.1 | `data/teachers/causal_lm.py` | Add `top_k_probs(input_ids, K)` → returns `(probs, indices)` shape `B×L×K`; apply softmax then take top-K and re-normalize |
| 1.2 | `data/llm_topk_probs_datamodule.py` | New datamodule: tokenize raw text → call `top_k_probs()` → apply ILR → return `log_x` tensor |
| 1.3 | `config.py` | Add `LLMTopKProbsConfig(model_id, K, renormalize)` |
| 1.4 | `training/data_sources.py` | Register `"llm_topk_probs"` source in `build_training_datamodule()` |
| 1.5 | `scripts/two_stage_train.py` | Add CLI flag `--training-data-source llm_topk_probs` |
| 1.6 | `tests/` | `test_llm_topk_probs_datamodule.py` smoke test |

**Acceptance criteria:** Running `two_stage_train.py --training-data-source llm_topk_probs` produces trained Stage 1 + Stage 2 checkpoints; GP energy separates plausible from corrupted top-K distributions with AUROC > 0.80 on text8.

### Phase 2 — Domain-Specific Data Pipelines

**Goal:** Add DNA and medical text datasets with appropriate vocabularies and invalid sequence generation.

**Tasks:**

| Task | File | Description |
|------|------|-------------|
| 2.1 | `data/dna_datamodule.py` | FASTA-format nucleotide sequences; K=4 (ACGT) or K=5 with N; point mutation + frameshift invalid generation |
| 2.2 | `data/medical_datamodule.py` | Character or BPE subword tokenization of clinical text; drug-name misspelling + unit corruption negatives |
| 2.3 | `config.py` | `DNADatasetConfig`, `MedicalDatasetConfig` |
| 2.4 | `benchmarks/tasks/dna_audit.py` | DNA benchmark task (AUROC on mutation types) |
| 2.5 | `benchmarks/tasks/medical_audit.py` | Medical text benchmark |
| 2.6 | `benchmarks/tasks/registry.py` | Register new tasks |

### Phase 3 — Unified Benchmark Runner

**Goal:** Single entry point that exercises all three components across all tasks and produces a consolidated results table.

**Tasks:**

| Task | File | Description |
|------|------|-------------|
| 3.1 | `benchmarks/runner.py` | Extend to iterate over `(component, task)` pairs; log all five signals per row |
| 3.2 | `benchmarks/results_schema.py` | Typed result dataclass; JSON serialization with component/task/scale keys |
| 3.3 | `scripts/run_full_benchmark.py` | Top-level script: trains all components → runs all tasks → saves `results/full_benchmark.json` |
| 3.4 | `plots/plots.py` | Add `plot_benchmark_table()`: heatmap of AUROC by (component, task) |

### Phase 4 — OOD Healing

**Goal:** Implement and evaluate targeted re-sampling and simplex projection healing.

**Tasks:**

| Task | File | Description |
|------|------|-------------|
| 4.1 | `healing/targeted_resample.py` | Mask high-energy tokens → resample via LLM → re-evaluate; configurable threshold + max iterations |
| 4.2 | `healing/simplex_project.py` | ILR inverse + flow ODE backward pass → project to nearest valid token |
| 4.3 | `healing/beam_rerank.py` | Beam search scorer with energy penalty |
| 4.4 | `benchmarks/tasks/healing_audit.py` | Extend existing stub with Phase 4 methods |
| 4.5 | `config.py` | `HealingConfig(strategy, threshold, max_iter, lambda_energy)` |
| 4.6 | `scripts/run_healing_eval.py` | Evaluate healing strategies; report energy/variance/accuracy before vs after |

### Phase 5 — Calibration & Analysis

**Goal:** Verify calibration of GP variance estimates and analyze failure modes.

**Tasks:**

| Task | File | Description |
|------|------|-------------|
| 5.1 | `metrics/calibration.py` | Expected Calibration Error (ECE) and reliability diagrams for GP variance |
| 5.2 | `analysis/token_heatmaps.py` | Extend existing plot: overlay all three UQ signals on same token axis |
| 5.3 | `analysis/component_correlation.py` | Pearson/Spearman correlation between Component 1 energy, Component 2 energy, and spilled energy |
| 5.4 | `analysis/failure_modes.py` | Identify sequences where components disagree; manual inspection tooling |

---

## Key Design Decisions & Trade-offs

### ILR vs CLR for probability simplex

Both ILR and CLR are valid Aitchison-space transforms. ILR produces an orthonormal basis (K→K−1 dims, no redundancy) and is preferred for the GP kernel (distances are Euclidean in ILR space). CLR retains interpretability (one coordinate per token) but adds a sum-to-zero constraint. **Decision: use ILR throughout** for cleaner GP geometry.

### Top-K re-normalization

When extracting top-K from a full-vocabulary distribution, the discarded probability mass must be re-distributed. Simple re-normalization (divide by sum of top-K) is chosen over temperature scaling or tail redistribution because:
- Preserves the shape of the top-K distribution
- The discarded mass is typically small for well-calibrated LMs
- Consistent with the Aitchison closed-operation on the simplex

### Shared backbone vs separate backbones

Components 1 and 2 operate on different input spaces (character one-hot vs LLM probability simplex). They require **separate Stage 1 + Stage 2 models** because their learned manifolds reflect different notions of validity. However, they share all architecture code, loss functions, and the GP implementation.

### Spilled energy as a baseline, not a component

Spilled energy has no free parameters and no UQ. It serves as a calibration check and a training-free lower bound for AUROC, not a primary UQ signal.

### Two-stage architecture rationale

Stage 1 learns the geometry of valid sequences without seeing invalid sequences. This prevents the backbone from collapsing to a trivial discriminative feature. Stage 2 then trains the GP contrastively with frozen backbone features. This separation ensures that the learned manifold reflects true structural/contextual regularity, not just binary classification.

---

## File Inventory Summary

### Existing files (relevant)

| File | Component(s) |
|------|-------------|
| [src/aitchinson_flow/geometry.py](src/aitchinson_flow/geometry.py) | 1, 2 |
| [src/aitchinson_flow/models/bayesian_auditor_stage1.py](src/aitchinson_flow/models/bayesian_auditor_stage1.py) | 1, 2 |
| [src/aitchinson_flow/models/bayesian_auditor_stage2.py](src/aitchinson_flow/models/bayesian_auditor_stage2.py) | 1, 2 |
| [src/aitchinson_flow/gp/gp.py](src/aitchinson_flow/gp/gp.py) | 1, 2 |
| [src/aitchinson_flow/metrics/spilled_energy.py](src/aitchinson_flow/metrics/spilled_energy.py) | 3 |
| [src/aitchinson_flow/data/text8_datamodule.py](src/aitchinson_flow/data/text8_datamodule.py) | 1 (text8) |
| [src/aitchinson_flow/data/qa_datamodule.py](src/aitchinson_flow/data/qa_datamodule.py) | 2 (trivia) |
| [src/aitchinson_flow/data/teachers/causal_lm.py](src/aitchinson_flow/data/teachers/causal_lm.py) | 2, 3 |
| [src/aitchinson_flow/data/corruption.py](src/aitchinson_flow/data/corruption.py) | 1, 2 |
| [benchmarks/tasks/text_audit.py](benchmarks/tasks/text_audit.py) | 1 eval |
| [benchmarks/tasks/trivia_audit.py](benchmarks/tasks/trivia_audit.py) | 2 eval |
| [benchmarks/tasks/healing_audit.py](benchmarks/tasks/healing_audit.py) | Healing |

### Files to create (new)

| File | Phase | Purpose |
|------|-------|---------|
| `src/aitchinson_flow/data/llm_topk_probs_datamodule.py` | 1 | Top-K probability pipeline |
| `src/aitchinson_flow/data/dna_datamodule.py` | 2 | DNA sequence data |
| `src/aitchinson_flow/data/medical_datamodule.py` | 2 | Medical text data |
| `benchmarks/tasks/dna_audit.py` | 2 | DNA benchmark task |
| `benchmarks/tasks/medical_audit.py` | 2 | Medical benchmark task |
| `benchmarks/results_schema.py` | 3 | Typed result schema |
| `scripts/run_full_benchmark.py` | 3 | Full benchmark entry point |
| `src/aitchinson_flow/healing/targeted_resample.py` | 4 | Targeted re-sampling healer |
| `src/aitchinson_flow/healing/simplex_project.py` | 4 | Simplex projection healer |
| `src/aitchinson_flow/healing/beam_rerank.py` | 4 | Beam search reranking |
| `scripts/run_healing_eval.py` | 4 | Healing evaluation script |
| `src/aitchinson_flow/metrics/calibration.py` | 5 | ECE + reliability diagram |
| `src/aitchinson_flow/analysis/component_correlation.py` | 5 | Inter-component correlation |

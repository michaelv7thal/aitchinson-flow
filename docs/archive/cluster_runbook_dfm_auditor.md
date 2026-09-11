# Cluster runbook — DirichletFM auditor at LM scale


> **ARCHIVED (2026-08-20). Retired research line; none of the commands below can run.** Operational runbook for the GPT-2 / HaluEval "DFM auditor" line of 2026-05. The line was retired and contributed nothing to the capstone paper, and every script invoked below was deleted by commit `7474f16`. Kept for the LM memory and disk-sizing tables (§1, §5) and the troubleshooting notes (§7), which are recorded nowhere else. Companion index: `docs/archive/dfm_auditor_README.md`.

End-to-end guide for caching HaluEval-QA features under a chosen LM
(GPT-2, Llama-3.2-1B, Llama-2-7B, Mistral-7B, …) and running the DFM
auditor's three modes (Architecture A baseline, the unsupervised EBM,
and Architecture B's joint dual-head supervised model).

## 0. Setup (once per cluster node)

```bash
# 0a. Pull the repo
git pull origin capstone-project    # or whatever branch you're on

# 0b. Make sure uv is on PATH and deps are installed.
# The lock file already pins everything; uv sync will pick up
# torch + transformers + datasets + scipy + gpytorch + bitsandbytes.
uv sync

# 0c. HuggingFace auth — needed for Llama gated repos. Skip for GPT-2 / Mistral.
export HF_TOKEN="hf_..."
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# 0d. Decide where caches live. They can be large (Llama-7B fp16
# hidden_states for 20000 rows × L=256 × H=4096 ≈ 40 GB).
mkdir -p data
```

## 1. Build the LM-feature cache (one-time per LM)

The new script `scripts/cache_hallueval_llama.py` produces **both** caches
that the DFM auditor consumes (hidden + topk) in a single LM pass.
Pick a precision based on GPU memory:

| LM | parameters | bf16 weights | + activations (b=4, L=256) | total est. | precision flag |
|---|---:|---:|---:|---:|---|
| GPT-2 small | 124 M | 0.25 GB | <1 GB | ~1 GB | `--precision fp32` |
| Llama-3.2-1B | 1 B | 2 GB | ~1 GB | ~3 GB | `--precision bf16` |
| Llama-3.2-3B | 3 B | 6 GB | ~2 GB | ~8 GB | `--precision bf16` |
| Llama-2-7B / Llama-3.1-8B | 7-8 B | 14-16 GB | 2-3 GB | 16-19 GB | `--precision bf16` (tight) |
| Llama-2-7B at 4-bit | 7 B | ~4 GB | ~3 GB | ~7 GB | `--precision 4bit` |
| Mistral-7B | 7 B | 14 GB | ~2 GB | ~16 GB | `--precision bf16` |

### 1a. Llama-3.2-1B (recommended starting point)

```bash
python scripts/cache_hallueval_llama.py \
    --lm meta-llama/Llama-3.2-1B \
    --precision bf16 \
    --max-n 10000 --L 256 --K 32 --lm-batch-size 8 \
    --hidden-out data/hallueval_cache_llama1b.pt \
    --topk-out   data/hallueval_topk_llama1b.pt
```

Wall-clock: ~30-60 min on a single H100 / A100 / 3090 (less on a 4090).
Outputs:
* `data/hallueval_cache_llama1b.pt` — ~26 GB (fp16 hidden states for
  20000 rows × 256 tokens × 2048 dims).
* `data/hallueval_topk_llama1b.pt` — ~1 GB.

### 1b. Llama-2-7B at 4-bit (the SOTA-territory backbone)

```bash
python scripts/cache_hallueval_llama.py \
    --lm meta-llama/Llama-2-7b-hf \
    --precision 4bit \
    --max-n 10000 --L 256 --K 32 --lm-batch-size 4 \
    --hidden-out data/hallueval_cache_llama7b.pt \
    --topk-out   data/hallueval_topk_llama7b.pt
```

Wall-clock: ~2-3 hours. Outputs ~52 GB hidden / ~1 GB topk.

To shrink the hidden cache to ~13 GB use `--no-fp16-hidden` *off* (default
is fp16) and verify `H` in the cache metadata.

### 1c. GPT-2 small (the existing baseline; for parity-compute checks)

The existing 4000-row cache `data/hallueval_topk_gpt2.pt` works as-is.
To rebuild on the full 10000 pairs:

```bash
python scripts/cache_hallueval_llama.py \
    --lm gpt2 \
    --precision fp32 \
    --max-n 10000 --L 256 --K 32 --lm-batch-size 16 \
    --hidden-out data/hallueval_cache_gpt2_full.pt \
    --topk-out   data/hallueval_topk_gpt2_full.pt
```

## 2. Train the DFM auditor (Architecture B — primary)

Architecture B trains the encoder jointly with two heads:

* `slot_head`   — CE over top-K slots on **clean** rows (preserves the EBM)
* `halluc_head` — BCE on row label propagated to answer-mask tokens
                  (supervised UQ, all rows)

Total loss: `λ_slot * CE_slot + λ_halluc * BCE_halluc`. The auto-pos-weight
in the runner balances the BCE for HaluEval-QA's ~6:1 halluc:clean class
imbalance among answer-mask positions.

### 2a. Llama-1B backbone, full 10000 pairs, d=512/L=6, 15 epochs

```bash
python scripts/run_dirichlet_fm_auditor_archB.py \
    --topk-cache   data/hallueval_topk_llama1b.pt \
    --hidden-cache data/hallueval_cache_llama1b.pt \
    --ctx-hidden 2048 \
    --epochs 15 --batch-size 8 --max-rows 20000 \
    --d-model 512 --num-layers 6 --nhead 8 \
    --mode product_concat \
    --energy-t 4.0 --t-max 8.0 \
    --lambda-slot 1.0 --lambda-halluc 1.0 \
    --warmup-steps 400 --save-best \
    --out-dir runs/dfm_archB_llama1b
```

Wall-clock: ~1-2 hours on an H100 / A100.

### 2b. Llama-7B backbone, same setup, ctx-hidden=4096

```bash
python scripts/run_dirichlet_fm_auditor_archB.py \
    --topk-cache   data/hallueval_topk_llama7b.pt \
    --hidden-cache data/hallueval_cache_llama7b.pt \
    --ctx-hidden 4096 \
    --epochs 15 --batch-size 4 --max-rows 20000 \
    --d-model 512 --num-layers 6 --nhead 8 \
    --mode product_concat \
    --warmup-steps 400 --save-best \
    --out-dir runs/dfm_archB_llama7b
```

The smaller batch (4 vs 8) is to fit the 4096-d projection at high d_model.

### 2b'. Same as 2a but with the per-position MLP backbone (cascade-clean)

The `--backbone mlp` flag swaps the transformer encoder for a weight-shared
per-position MLP with **no cross-positional attention**. This makes
`tok(non-answer-span) AUROC ≈ 0.5` *by construction* — the cascade-
contamination problem in Architecture B's transformer encoder doesn't
exist in this architecture.

```bash
python scripts/run_dirichlet_fm_auditor_archB.py \
    --topk-cache   data/hallueval_topk_llama1b.pt \
    --hidden-cache data/hallueval_cache_llama1b.pt \
    --ctx-hidden 2048 \
    --epochs 15 --batch-size 8 --max-rows 20000 \
    --d-model 512 --num-layers 6 --nhead 8 \
    --mode product_concat \
    --backbone mlp \
    --warmup-steps 400 --save-best \
    --out-dir runs/dfm_archB_llama1b_mlp
```

Expected (from GPT-2 sanity): row AUROC slightly lower than transformer
(0.92-0.96 vs 0.985), but `tok(non-ans)` ≈ 0.5 instead of 0.98 — clean
locality + strong row-level. This is the architecturally-principled
choice when *per-position localisation matters* (i.e. when you want to
say "*which* token is anomalous", not just "is this row anomalous").

### 2c. The three-way mode ablation (`off` / `hidden_only` / `product_concat`)

To replicate the GPT-2 ablation result on Llama-1B (showed `off` is the
best generator, `hidden_only` the best UQ, `product_concat` middle):

```bash
for mode in off hidden_only product_concat; do
    python scripts/run_dirichlet_fm_auditor_archB.py \
        --topk-cache   data/hallueval_topk_llama1b.pt \
        --hidden-cache data/hallueval_cache_llama1b.pt \
        --ctx-hidden 2048 \
        --epochs 15 --batch-size 8 --max-rows 20000 \
        --d-model 512 --num-layers 6 --nhead 8 \
        --mode "$mode" \
        --warmup-steps 400 --save-best \
        --out-dir "runs/dfm_archB_llama1b_${mode}"
done
```

### 2d. Slot-only (Architecture-A baseline) — the no-supervision control

To compare against the unsupervised EBM, train the slot head only (no
joint BCE). This recovers our existing GPT-2 numbers (~0.81 row AUROC):

```bash
python scripts/run_dirichlet_fm_auditor.py \
    --topk-cache   data/hallueval_topk_llama1b.pt \
    --hidden-cache data/hallueval_cache_llama1b.pt \
    --ctx-hidden 2048 \
    --epochs 15 --batch-size 8 --max-rows 20000 \
    --d-model 512 --num-layers 6 --nhead 8 \
    --mode hidden_only \
    --warmup-steps 400 --save-best \
    --out-dir runs/dfm_slot_llama1b
```

Then run the SVGP-on-frozen-encoder Architecture A:

```bash
python scripts/run_dfm_auditor_archA.py \
    --ckpt runs/dfm_slot_llama1b/best.pt \
    --out  runs/dfm_archA_llama1b \
    --inducing 128 --iters 500
```

## 3. Phase-H-style generation eval

This tests the user's "is the text actually coherent?" diagnostic. The
EqM Phase H reference is NLL = 8.89 (FAIL); DFM at GPT-2 scale already
hits NLL ≈ 5.7 on the simplest mode. Llama-scale should be substantially
better.

```bash
python scripts/eval_dfm_auditor_generation.py \
    --ckpt runs/dfm_archB_llama1b/best.pt \
    --n-samples 64 --nfe 100
```

Outputs `genH.json` with mean NLL (generated, random-slot, clean), vocab
match, slot-diversity, and a small text grid printed to stdout.

## 4. Compile results into a comparison table

```bash
python scripts/compile_dfm_auditor_results.py \
    runs/dfm_archB_llama1b_off \
    runs/dfm_archB_llama1b_hidden_only \
    runs/dfm_archB_llama1b_product_concat \
    --out runs/dfm_archB_llama1b_table.md
```

## 5. Disk usage estimates

| artifact | GPT-2 (4000) | Llama-1B (20000) | Llama-7B (20000) |
|---|---:|---:|---:|
| hidden cache | 0.25 GB | 26 GB | 52 GB |
| topk cache | 0.25 GB | 0.5 GB | 0.5 GB |
| training checkpoints | 0.1 GB | 0.1 GB | 0.1 GB |

If disk is tight, you can post-process the hidden cache to keep only
**answer-span positions** (saves ~12× since most positions are the
shared prompt prefix). Will provide a `scripts/trim_hidden_cache.py`
helper if requested.

## 6. SOTA reference numbers (HaluEval-QA, EigenTrack 2025)

To put DFM auditor numbers in context after a Llama run:

| method | LLaMa-1B | Llama-3B | Llama-7B | Mistral-7B |
|---|---:|---:|---:|---:|
| EigenTrack (2025 SOTA) | 0.842 | 0.861 | 0.894 | 0.888 |
| HaloScope | 0.820 | 0.827 | 0.861 | — |
| LapEigvals | 0.785 | 0.819 | 0.871 | — |
| INSIDE | 0.753 | 0.831 | 0.810 | — |
| SelfCheckGPT | 0.739 | 0.804 | 0.809 | — |

Our DFM auditor at GPT-2 scale: 0.81 (unsupervised EBM), 0.85
(Architecture A SVGP). Targets at Llama-1B/7B scale: beat the
respective row.

## 7. Troubleshooting

* **`bitsandbytes` import error in `--precision 4bit`**: install with
  `pip install bitsandbytes` or `uv add bitsandbytes`. Needs a CUDA
  GPU and a recent driver.
* **OOM on cache step**: drop `--lm-batch-size`. For Llama-7B fp16
  on 20 GB, batch-size=2 is safe.
* **OOM on training step (Llama-7B context)**: drop `--batch-size`
  or set `--ctx-proj-dim 32` (default 64) to halve the projection
  width.
* **`--ctx-hidden` does not match cache H**: the runner asserts
  `cache.H == --ctx-hidden`; bump the flag to the LM's `model.config.hidden_size`
  (Llama-1B = 2048, Llama-3B = 3072, Llama-7B = 4096, Mistral-7B = 4096).
* **Auto pos_weight is 0**: the auto-compute requires the train loader
  to be iterable; if the cache is empty the runner aborts. Verify with
  `python -c "import torch; c=torch.load('data/hallueval_topk_llama1b.pt'); print(c['N'])"`.
* **HF Llama 401 / gated repo**: accept the model's license at
  `huggingface.co/meta-llama/Llama-3.2-1B` then `huggingface-cli login`.

## 8. What to commit back

After a full run, the cluster typically commits:

* `runs/dfm_archB_<lm>/{summary.json, best.pt, epoch_final.pt, genH.json}`
* `runs/dfm_archB_<lm>_<mode>/...` for each mode in the ablation
* `runs/dfm_archA_<lm>/archA_summary.json`
* `runs/dfm_archB_<lm>_table.md` — compiled comparison

Cache files (`data/hallueval_*_<lm>.pt`) are too large for git; keep
them on the cluster's storage and reference by path.

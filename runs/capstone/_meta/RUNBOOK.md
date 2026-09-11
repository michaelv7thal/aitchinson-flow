# Capstone protocol runbook

Pickle-friendly recipe for the A100 session. Everything below is *idempotent*:
re-running a step that has already produced its output is a no-op (eval.json /
ckpt existence check).

All commands run from `/workspace`.

## 0. Sanity check

```bash
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
ls runs/capstone/   # R S T U V _meta checkpoints README.md
```

## 1. Build base checkpoints (~14 min × 4 ≈ 1 hr on A100 MIG)

The protocol assumes runs/{eqm,dfm,fmclr,lkflow}_data50k_ep5_v2/epoch_final.pt.
Re-train them under `runs/capstone/checkpoints/` so the original eval.jsons
are preserved.

```bash
python scripts/run_sweep.py --sweep sweeps/capstone_base.yaml \
    --runs-root runs/capstone/checkpoints
```

Sanity baselines (compare against the protocol's reference numbers):

```bash
for ck in eqm_data50k_ep5_v2 dfm_data50k_ep5_v2 fmclr_data50k_ep5_v2 lkflow_data50k_ep5; do
    python -c "
import json, sys
d = json.load(open(f'runs/capstone/checkpoints/$ck/eval.json'))
print(f'{$ck:>20}  KL_uni={d[\"unigram_kl\"]:.4f}  KL_bi={d[\"bigram_kl\"]:.4f}')
"
done
```

Expect: DFM KL_bi ≈ 0.148, EqM ≈ 1.38, FMonCLR ≈ 1.56, LKFlow ≈ 1.43.

## 2. Phase R — SDE sampling (eval-only, ~30 min)

```bash
python scripts/run_sweep.py --sweep sweeps/phaseR_sde.yaml \
    --runs-root runs/capstone/R \
    --n 256 --steps 128
python scripts/plot_phaseR.py --root runs/capstone/R \
    --out runs/capstone/R/phaseR_sde.png \
    --summary runs/capstone/R/phaseR_sde.json
```

Read off best-α KL_bi per family from `runs/capstone/R/phaseR_sde.json` and
apply the §4 PASS/PARTIAL/FAIL criteria. Append a Phase R block to
`runs/DECISION_LOG.md`.

## 3. Phase S — K=2 binary collapse (~80 min)

```bash
python scripts/run_sweep.py --sweep sweeps/phaseS_K2.yaml \
    --runs-root runs/capstone/S
```

Compare EqM-best vs DFM-best KL_bi at K=2 per the §5 EQUAL / GAP / AMBIG
criteria. The K=2 entropy is much smaller in absolute terms; report
Δ KL_bi alongside the absolute numbers.

## 4. Phase T — Conservative-gradient regression (~35 min)

```bash
python scripts/run_sweep.py --sweep sweeps/phaseT_consgrad.yaml \
    --runs-root runs/capstone/T
python scripts/eval_w1_compare.py \
    --eqm-ckpt      runs/capstone/checkpoints/eqm_data50k_ep5_v2/epoch_final.pt \
    --consgrad-ckpt runs/capstone/T/eqm_consgrad_data50k_ep5/epoch_final.pt \
    --out           runs/capstone/T/phaseT_w1_compare.json
```

Phase T PASS criterion: Euler-on-output(consgrad) KL_bi ≤ 1.45 (within 0.05
of NAG-on-eqm 1.382). See §6.

## 5. Phase U — Cascade audit (~30 min total)

```bash
# 5.1 Build the wiki cache (one-time, ~5 min on A100).
python scripts/cache_wiki.py --lm gpt2 --n 300 --L 64 --K 64 \
    --out data/wiki_cache_gpt2.pt

# 5.2 Train the auditor (~25 min, follows the original aud_gpt2_ctx recipe).
python scripts/run_sweep.py --sweep sweeps/capstone_auditor.yaml \
    --runs-root runs/capstone/checkpoints

# 5.3 Run the cascade audit.
python scripts/cascade_audit.py \
    --cache data/wiki_cache_gpt2.pt \
    --include SE topk_entropy linear_probe blr_laplace svgp eqm_auditor \
    --auditor-ckpt runs/capstone/checkpoints/aud_gpt2_ctx/epoch_final.pt \
    --out runs/capstone/U/phaseU_cascade_audit.json

python scripts/plot_phaseU.py runs/capstone/U/phaseU_cascade_audit.json
```

Phase U decision: see §7 — pattern-match locality-clean vs cascade-contaminated.

## 6. Phase V — SAPLMA replication (~18 min)

```bash
python scripts/train_saplma.py \
    --cache data/wiki_cache_gpt2.pt \
    --out runs/capstone/checkpoints/saplma_wiki/probe.pt \
    --epochs 25

python scripts/cascade_audit.py \
    --cache data/wiki_cache_gpt2.pt \
    --include SE topk_entropy linear_probe blr_laplace svgp eqm_auditor saplma \
    --auditor-ckpt runs/capstone/checkpoints/aud_gpt2_ctx/epoch_final.pt \
    --saplma-ckpt  runs/capstone/checkpoints/saplma_wiki/probe.pt \
    --out runs/capstone/U/phaseU_with_saplma.json
python scripts/plot_phaseU.py runs/capstone/U/phaseU_with_saplma.json
```

Phase V decision criteria in §8: DIAGNOSTIC FIRES if AUROC₁>0.95 AND
AUROC₂>0.85 AND AUROC₃<0.70.

## 7. Update REPORT.md and DECISION_LOG.md

After each phase, append a `Hypothesis / Result / Decision / Next` block
to `runs/DECISION_LOG.md`. After Phase R/T/U, also extend the relevant
section of `REPORT.md` per §11.

## Total wall-clock budget (parity compute on A100 MIG)

| step | time |
|---|---:|
| §1 base                | 1 hr   |
| §2 Phase R             | 30 min |
| §3 Phase S             | 80 min |
| §4 Phase T             | 35 min |
| §5 Phase U             | 30 min |
| §6 Phase V             | 18 min |
| **total**              | **~3 hr 30 min** |

# Compositional EqM — MSE vs Hilbert test runbook

You are a Claude session running on a compute cluster. Your job is to
execute the Compositional EqM ablation defined in
`sweeps/compositional_eqm_test.yaml` and report results back. The sweep
trains 8 cells: a smoke test, 3 seeds × {Compositional MSE, Compositional
Hilbert}, and 2 deterministic-CLR reference cells.

The proposal lives in `PROPOSAL_COMPOSITIONAL_EQM.md`. Read § 1-3 if you
need context for what the recipe is and why it should help.

## 0. Prerequisites — verify before starting

Run these in order. Each must succeed before the next.

```bash
# Repo at expected SHA?
git rev-parse HEAD
git status --short
```

Expect a clean tree. If `recovery.json` files in `runs/` show as modified,
that's leftover from prior eval runs and is OK — leave them.

```bash
# Python + GPU?
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
```

Expect `True 1` (or more) and a GPU with ≥ 16 GB free. If `False`, stop —
the sweep needs a GPU.

```bash
# Dirichlet calibration sanity (≈ 5 s)
uv run python scripts/check_dirichlet_data.py --alpha-peak 10 --alpha-base 0.1 --seeds 42,43,44 || true
```

Expect ≥ 99.5 % argmax recovery for `α_peak=10`. If lower, flag it — the
calibration assumption in the proposal is broken.

## 1. Smoke test — fail fast

```bash
uv run python scripts/run_sweep.py \
    --sweep sweeps/compositional_eqm_test.yaml \
    --only comp_smoke \
    --runs-root runs
```

Expect:
* runs in 1-3 minutes
* `runs/comp_smoke/{config.json,history.jsonl,eval.json,epoch_final.pt}` present
* `flow_loss` strictly decreasing across the epoch (read `history.jsonl`)
* `eval.json` `H_ratio` finite (any value — model is not trained, just
  validating the wiring)

If any of those fail, stop and surface the error to the user instead of
proceeding. Common failures:
* CUDA OOM — drop `text8_dataset.batch_size` in the smoke cell.
* `_standard_gamma` not found — torch < 2.0; environment is wrong.
* `loss.hilbert_alpha` missing — sweep file edited; re-pull.

## 2. Full sweep — 6 main cells + 2 reference cells

Launch in the **background** so you can poll progress without blocking.
Each cell is ~30-50 min on an A100; total wall ≈ 4-6 h.

```bash
nohup uv run python scripts/run_sweep.py \
    --sweep sweeps/compositional_eqm_test.yaml \
    --runs-root runs \
    > runs/compositional_eqm_test.log 2>&1 &
echo $! > runs/compositional_eqm_test.pid
```

The orchestrator is idempotent: a cell with an existing `eval.json` is
skipped, so re-running the same command after an interruption resumes
where it left off (the smoke cell's eval.json from § 1 means the smoke
cell will be skipped — that is the intended behaviour).

### 2.1 Monitor

Poll progress with the `Monitor` tool (or `tail -F`) — do NOT sleep-loop.

What to watch for in the log:
* `[run] comp_*` — cell starting
* `train epoch: N: X/16` — tqdm progress within an epoch
* `flow_loss=` should decrease monotonically (averaged per epoch).
  - For Compositional MSE: expect plateau around 1-3 (variance floor; this
    is the proposal's prediction, NOT a bug).
  - For Compositional Hilbert: similar magnitude, slightly lower in our
    Phase 5 history.
  - For deterministic references: drops to < 0.1 within 2 epochs (the
    spike-basin failure mode).
* `unigram_kl=` reported each epoch by the sample-eval probe. Healthy
  numbers are < 1.0; a value > 2 means mode collapse. Flag it.
* `[error] comp_<name>:` — a cell failed; the orchestrator continues to
  the next one. Note which one failed and why; do NOT abort the sweep.

If you see `Killed` (OOM) or `CUDA out of memory`, the GPU is too small;
add `transformer.d_model: 512, transformer.num_layers: 6` to the failing
cell's overrides and re-run only that cell with `--only`.

### 2.2 What "done" means

```bash
ls runs/comp_*/eval.json | wc -l
```

Expect 8 (smoke + 6 main + 2 reference).

## 3. Recovery diagnostic — per cell

Unconditional generation is a poor proof point for this recipe (proposal
§ 5.1). Recovery from perturbation is the headline metric. Run it for
every trained cell:

```bash
for cell in comp_mse_seed{42,43,44} comp_hilbert_seed{42,43,44} \
            comp_ref_det_mse comp_ref_det_hilbert; do
  uv run python scripts/recovery_check.py \
    --ckpt runs/$cell/epoch_final.pt \
    --alphas 0.05,0.10,0.20,0.30,0.40,0.50,1.00 \
    --n 256 --steps 200 \
    --out runs/$cell/recovery.json
done
```

Each call writes `runs/<cell>/recovery.json` with one row per α. Token
accuracy at α = 0.50 is the headline; KL_bi vs the corpus distribution
shows whether the recovered text stays text-like.

## 4. Tabulate

Aggregate one row per cell. Use this script (ad hoc — paste into the
shell):

```bash
uv run python - <<'PY'
import json
from pathlib import Path
import statistics as stats

cells = [
  ("Compositional MSE", ["comp_mse_seed42","comp_mse_seed43","comp_mse_seed44"]),
  ("Compositional Hilbert", ["comp_hilbert_seed42","comp_hilbert_seed43","comp_hilbert_seed44"]),
  ("Det. MSE (reference)", ["comp_ref_det_mse"]),
  ("Det. Hilbert (reference)", ["comp_ref_det_hilbert"]),
]
header = f"{'condition':<28} {'KL_uni':>8} {'KL_bi':>8} {'H_ratio':>8} "\
         f"{'acc@.10':>8} {'acc@.30':>8} {'acc@.50':>8} {'Δ@.50':>7}"
print(header); print("-" * len(header))
for label, names in cells:
    KLu, KLb, Hr, a10, a30, a50, d50 = ([] for _ in range(7))
    for n in names:
        ev = json.loads(Path(f"runs/{n}/eval.json").read_text())
        KLu.append(ev["unigram_kl"]); KLb.append(ev["bigram_kl"]); Hr.append(ev["H_ratio"])
        rc = json.loads(Path(f"runs/{n}/recovery.json").read_text())["rows"]
        rec = {r["alpha"]: r for r in rc if r["mode"]=="recovery"}
        a10.append(rec[0.10]["token_acc"]); a30.append(rec[0.30]["token_acc"]); a50.append(rec[0.50]["token_acc"])
        d50.append(rec[0.50]["token_acc"] - rec[0.50]["token_acc_perturbed"])
    m = lambda xs: stats.mean(xs)
    s = lambda xs: stats.pstdev(xs) if len(xs)>1 else 0.0
    print(f"{label:<28} {m(KLu):>8.3f} {m(KLb):>8.3f} {m(Hr):>8.3f} "
          f"{m(a10):>8.3f} {m(a30):>8.3f} {m(a50):>8.3f} {m(d50):>+7.3f}"
          + (f"  ±{s(a50):.3f}" if len(names)>1 else ""))
PY
```

Save the printed table to `runs/compositional_eqm_test_summary.md`
(create the file with the table prefixed by a one-paragraph note linking
to this runbook and the sweep YAML).

## 5. Report back

Send the user (whoever invoked this Claude session):
1. The summary table from step 4.
2. The proposal's headline question, answered with numbers from the
   table:
   - "Did Compositional MSE recover at α = 0.50?" — yes if `Δ@.50 > 0`
     for both MSE and Hilbert and `0` (or negative) for the deterministic
     references.
   - "Is Hilbert better than MSE?" — quote the mean Δ@.50 for each, with
     the across-seed std. Proposal § 5.3 expects a small effect (~ 2-3 %);
     if the effect is larger, flag it as a finding worth investigating.
3. Any cells that errored or behaved unexpectedly (mode collapse,
   harmful sampling, OOM that needed a smaller model).
4. Pointers: `runs/compositional_eqm_test.log`, `runs/comp_*/eval.json`,
   `runs/comp_*/recovery.json`.

## 6. Cleanup (only if the user asks)

```bash
# Free disk — keep eval.json + recovery.json + history.jsonl, drop ckpt.
for d in runs/comp_*/; do
  test -f "$d/eval.json" && rm -f "$d/epoch_final.pt"
done
```

Do NOT cleanup automatically. The user may want the checkpoints for
follow-up experiments (e.g. running the latent-EqM healing demo against
the trained models).

---

## Reference: cell expectations

Numbers below are from the existing `runs/dphase3_mse_dirichlet/` and
`runs/dphase5_hilbert_dirichlet/` runs (5 ep × 10k windows × default
model), so they are directly comparable to the new sweep cells. They are
sanity targets, not pass/fail thresholds.

| Condition                | KL_uni | KL_bi | acc@0.50 | Δ@0.50 |
|--------------------------|--------|-------|----------|--------|
| Compositional MSE        | ≈ 0.65 | ≈ 5.8 | ≈ 0.57   | +0.05  |
| Compositional Hilbert    | ≈ 0.55 | ≈ 5.6 | ≈ 0.58   | +0.06  |
| Det. MSE (reference)     | ≈ 0.05 | ≈ 2.0 | ≈ 1.00   |  0.00  |
| Det. Hilbert (reference) | ≈ 0.05 | ≈ 2.0 | ≈ 1.00   |  0.00  |

Read the deterministic references carefully: KL is *better* (smaller)
because the model overfits to its tiny degenerate energy field — but
Δ@0.50 ≈ 0 means the sampler is a no-op on perturbed inputs, which is
the failure mode the proposal targets. Compositional cells have higher
KL but the sampler does real work.

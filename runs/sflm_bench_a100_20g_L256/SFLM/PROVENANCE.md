# SFLM checkpoint provenance (resolved 2026-08-21)

This arm has no `epoch_final.pt`. The checkpoint that produced the committed
`recovery.json` and `eval_all.json` (Jun 25) was lost; both surviving files
are a same-recipe retrain dated Aug 9 (`bench.log`: auto-train via
`train_for_sflm_bench.py --scale a100_20g_L256 --only SFLMEBM --epochs 10`).

Reproduction check at alpha=0.5 (n=256, steps=200, seed=42, the published
protocol; per-alpha seeding makes rows exactly comparable):

| source                       | tok_acc | tok_acc_pt | Delta   |
|------------------------------|---------|------------|---------|
| published recovery.json      | 0.5571  | 0.4903     | +0.0668 |
| epoch_10.pt (payload epoch 10) | 0.5614  | 0.4946     | +0.0668 |
| epoch_20.pt (payload epoch 20) | 0.5863  | 0.5090     | +0.0773 |

`epoch_10.pt` matches the benchmark budget (10 epochs) and reproduces the
published Delta to four decimals; the small tok_acc/tok_acc_pt drift is
training nondeterminism in the retrained embedding. **`epoch_10.pt` is the
canonical surviving checkpoint for this arm**; `epoch_20.pt` is an extension
beyond the matched budget and must not back any tab:gen number.

Artifacts from the check: `recovery_provcheck_ep20.json` / `.log` (epoch_20),
and the alpha=0.5 row inside `recovery_grid.json` (epoch_10, part of the
2026-08-21 common-grid sweep, `scripts/run_grid_recovery.sh`).

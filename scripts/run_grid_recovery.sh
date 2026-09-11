#!/usr/bin/env bash
#
# Common-grid recovery sweep across all nine benchmark arms
# (runs/sflm_bench_a100_20g_L256/), filling each arm's alpha ladder up to the
# common ten-point grid {0.1..1.0} so Delta can be compared at matched damage.
#
# Published recovery.json already covers {0.1,0.3,0.5,0.7,1.0} for every arm;
# recovery_matched.json (2026-08-20) covers {0.72..0.90}/{0.75..0.95} for the
# two Fisher-Rao arms. This script runs ONLY the missing points, with the
# published protocol (n=256, steps=200, seed=42; per-alpha seed is
# seed+int(alpha*1000), val windows are val_ids[:n], so points are exactly
# consistent across invocations). Output: recovery_grid.json per arm.
#
# SFLM has no epoch_final.pt; epoch_10.pt (payload epoch=10) matches the
# benchmark budget. Its grid includes alpha=0.5 as a provenance check: the
# result must reproduce the published row (tok_acc_pt=0.4903, delta=+0.0670)
# exactly, else this script prints PROVENANCE MISMATCH.
#
# Resumable: an arm with recovery_grid.json present is skipped.
#
set -u
cd "$(dirname "$0")/.." || exit 1
BENCH=runs/sflm_bench_a100_20g_L256
echo "[grid] started $(date '+%Y-%m-%d %H:%M:%S')"

run_arm() {
  local arm="$1" ckpt="$2" alphas="$3"
  local out="$BENCH/$arm/recovery_grid.json"
  if [ -f "$out" ]; then
    echo "[grid] SKIP $arm — recovery_grid.json already present"
    return 0
  fi
  if [ ! -f "$BENCH/$arm/$ckpt" ]; then
    echo "[grid] FAILED $arm — no checkpoint at $BENCH/$arm/$ckpt"
    return 1
  fi
  echo "[grid] START $arm alphas=$alphas $(date '+%H:%M:%S')"
  PYTHONUNBUFFERED=1 uv run python scripts/recovery_check.py \
      --ckpt "$BENCH/$arm/$ckpt" \
      --alphas "$alphas" \
      --n 256 --steps 200 --seed 42 --skip-uncond \
      --out "$out" \
      > "$BENCH/$arm/recovery_grid.log" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[grid] DONE  $arm $(date '+%H:%M:%S')"
  else
    echo "[grid] FAILED $arm (exit $rc). Last 15 log lines:"
    tail -15 "$BENCH/$arm/recovery_grid.log"
  fi
  return $rc
}

# Transport arms first (they carry the paper claim), EqM arms last.
run_arm DirichletFM epoch_final.pt "0.2,0.4,0.6,0.8,0.9"
run_arm SFM         epoch_final.pt "0.2,0.4,0.6"
run_arm FisherFM    epoch_final.pt "0.2,0.4,0.6"
run_arm SFLM        epoch_10.pt    "0.2,0.4,0.5,0.6,0.8,0.9"
run_arm FMonCLR     epoch_final.pt "0.2,0.4,0.6,0.8,0.9"
run_arm DFM         epoch_final.pt "0.2,0.4,0.6,0.8,0.9"
run_arm EqM         epoch_final.pt "0.2,0.4,0.6,0.8,0.9"
run_arm EqM_OneHot  epoch_final.pt "0.2,0.4,0.6,0.8,0.9"
run_arm EqMAE       epoch_final.pt "0.2,0.4,0.6,0.8,0.9"

# SFLM provenance check against the published alpha=0.5 row.
uv run python - <<'PY'
import json
pub = {r["alpha"]: r for r in json.load(open("runs/sflm_bench_a100_20g_L256/SFLM/recovery.json"))["rows"] if r.get("mode") == "recovery"}
try:
    new = {r["alpha"]: r for r in json.load(open("runs/sflm_bench_a100_20g_L256/SFLM/recovery_grid.json"))["rows"] if r.get("mode") == "recovery"}
except FileNotFoundError:
    print("[grid] SFLM provenance check skipped — no recovery_grid.json")
    raise SystemExit
p, n = pub.get(0.5), new.get(0.5)
if p is None or n is None:
    print("[grid] SFLM provenance check skipped — alpha=0.5 row missing")
elif abs(p["token_acc"] - n["token_acc"]) < 1e-4 and abs(p["token_acc_perturbed"] - n["token_acc_perturbed"]) < 1e-4:
    print(f"[grid] SFLM PROVENANCE OK: epoch_10.pt reproduces published alpha=0.5 "
          f"(tok_acc {n['token_acc']:.4f}, pt {n['token_acc_perturbed']:.4f})")
else:
    print(f"[grid] SFLM PROVENANCE MISMATCH: published acc={p['token_acc']:.4f}/pt={p['token_acc_perturbed']:.4f} "
          f"vs epoch_10 acc={n['token_acc']:.4f}/pt={n['token_acc_perturbed']:.4f} — try epoch_20.pt")
PY

echo "[grid] ALL DONE $(date '+%Y-%m-%d %H:%M:%S')"

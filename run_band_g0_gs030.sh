#!/usr/bin/env bash
# 2x2 SEPARATION, ARM A at the benchmark budget: gamma_lo 0.005 -> 0, gamma* = gamma_hi
# = 0.03 held at the published value (sweeps/band_L256.yaml: band_L256_ep10_d10k_g0_gs030).
# Differs from runs/band_L256_ep10_d10k (KL_bi 0.265, the one run that transitioned) in
# eqm.gamma_lo alone. Completes the 2x2 over the band's edges:
#     gamma_lo \ gamma*     0.030                     0.03573
#     0.005                 published (0.265)         _g005_gs99 (1.453)
#     0                     THIS RUN                  _g0 (1.512)
# Generation only (SKIP=recovery,ood,control), like the other edge cells; if it
# transitions, re-run drive_band_L256.sh on the cell without SKIP to add the ladder.
#
# QUEUE DISCIPLINE (author, 2026-08-29): "make sure it runs after the other GPU tasks
# have completed". So this script (1) waits for the two chains known to be alive or
# queued when it was written, then (2) requires the card to be idle for three
# consecutive minutes -- no process of ours on the GPU and no compute process above
# 100 MiB -- before it takes it. Any waiter that polls faster than that (all of ours
# poll every 30-60 s) therefore starts first, and this run stays last in line.
#
#   nohup setsid ./run_band_g0_gs030.sh > run_band_g0_gs030.log 2>&1 &
#   tail -f run_band_g0_gs030.log
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset WANDB_PROJECT WANDB_ENTITY WANDB_NAME WANDB_RUN_GROUP
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
ts(){ date +%Y-%m-%d_%H:%M:%S; }
echo $$ > .run_band_g0_gs030.pid
CELL=band_L256_ep10_d10k_g0_gs030
OUT=runs/$CELL/eval.json

# (1) the chains alive or queued at 2026-08-29 16:xx: run_seed43.sh (-> drive_band_L256.sh
#     -> run_sweep s43 eval) and drive_plausible_var.sh (queued behind it; plausible-swap
#     Var row). Both identified by pid so a restart of either under a new pid is caught
#     by (2) instead.
WAIT_PIDS="3699571 65222"
echo "### [$(ts)] queued: waiting for pids $WAIT_PIDS, then for a stably idle card"
for p in $WAIT_PIDS; do
  if kill -0 "$p" 2>/dev/null; then
    echo "### [$(ts)]   waiting on pid $p: $(ps -o args= -p "$p" | cut -c1-80)"
    while kill -0 "$p" 2>/dev/null; do sleep 60; done
    echo "### [$(ts)]   pid $p gone"
  fi
done

# (2) stable idle: six consecutive checks 30 s apart with nothing of ours on the card
#     and no compute process above 100 MiB.
gpu_busy() {
  pgrep -f 'scripts/(run_sweep|recovery_check|band_ood_score|ood_plausible_swap|ood_bayes_linear|train_for_sflm_bench|eval_all|eval_generation|heal_bench|ood_bench|ood_gpt2)[a-z_]*\.py' >/dev/null 2>&1 && return 0
  pgrep -f '(drive_plausible_var|run_seed43|drive_band_L256|drive_eqm_uniform_gamma|drive_backfill|drive_recovery_fine|drive_band_g0|run_ladder|run_separation|gate_g0)\.sh' >/dev/null 2>&1 && return 0
  nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '$1+0>100{f=1} END{exit !f}' && return 0
  return 1
}
idle=0; was_busy=""
while [ "$idle" -lt 6 ]; do
  if gpu_busy; then
    idle=0
    [ "$was_busy" = 1 ] || { echo "### [$(ts)]   card busy, waiting"; was_busy=1; }
  else
    idle=$((idle+1))
    [ "$was_busy" = 0 ] || { echo "### [$(ts)]   card idle, confirming over 3 min"; was_busy=0; }
  fi
  sleep 30
done
echo "### [$(ts)] card clear -- starting $CELL (generation only, ~4.3 h train + n=256/400-step eval)"

CELL=$CELL SKIP=recovery,ood,control ./drive_band_L256.sh
rc=$?
echo "### [$(ts)] drive_band_L256.sh exit $rc"

if [ -s "$OUT" ]; then
  uv run python - "$CELL" <<'PYEOF'
import json, pathlib, sys
cell = sys.argv[1]
def probe(name):
    h = pathlib.Path(f"runs/{name}/history.jsonl")
    if not h.is_file(): return "-", []
    ks = [json.loads(l)["bigram_kl"] for l in h.read_text().splitlines() if l.strip()]
    best = float("inf"); ep = "none"
    for i, k in enumerate(ks, 1):
        if i > 1 and k < 0.75 * best: ep = str(i); break
        best = min(best, k)
    return ep, ks
print(f"{'cell':<34} {'g_lo':>6} {'g*':>6} | {'KL_uni':>7} {'KL_bi':>6} {'KL_tri':>7} {'H_rat':>6} {'ep*':>4}")
for name, lo, gs in (("band_L256_ep10_d10k", "0.005", "0.030"), (cell, "0", "0.030"),
                     ("band_L256_ep10_d10k_g005_gs99", "0.005", "0.0357"), ("band_L256_ep10_d10k_g0", "0", "0.0357")):
    f = pathlib.Path(f"runs/{name}/eval.json")
    ep, ks = probe(name)
    if f.is_file():
        d = json.loads(f.read_text())
        print(f"{name:<34} {lo:>6} {gs:>6} | {d['unigram_kl']:>7.4f} {d['bigram_kl']:>6.3f} {d['trigram_kl']:>7.3f} {d.get('H_ratio', float('nan')):>6.3f} {ep:>4}")
    else:
        print(f"{name:<34} {lo:>6} {gs:>6} | {'-':>7} {'-':>6} {'-':>7} {'-':>6} {ep:>4}")
ep, ks = probe(cell)
print("probe KL_bi per epoch:", " ".join(f"{k:.3f}" for k in ks))
kb = json.loads(pathlib.Path(f"runs/{cell}/eval.json").read_text())["bigram_kl"]
print("VERDICT:", "TRANSITIONED (KL_bi %.3f < 0.5) -- 2 of 6 runs; consider the recovery ladder on it" % kb if kb < 0.5
      else "flat (KL_bi %.3f) -- 1 of 6 runs transitioned" % kb)
PYEOF
  echo "### [$(ts)] COMPLETE: $OUT"
else
  echo "!!! [$(ts)] no $OUT -- stage 1 failed; see runs/$CELL/error.txt if present"
  [ -s "runs/$CELL/error.txt" ] && sed 's/^/!!!   /' "runs/$CELL/error.txt" | head -5
fi
touch ".run_band_g0_gs030.done"

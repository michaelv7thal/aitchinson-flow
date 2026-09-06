#!/usr/bin/env bash
# Hold the chain after cell 1, then run the gamma_lo / gamma* separation.
#
# The running chain (drive_band_L256.sh, CELLS="...g0 ...ep30_d30k_g0") re-execs
# itself into the 30k x 30ep cell when cell 1 finishes. That cell inherits BOTH
# changes at once and costs ~47 h, which is not worth spending until we know
# which of the two levers moved the number. So: watch for the chain announcing
# cell 2, kill it at that moment, and run the two separation arms instead.
#
# Stage 1 only for the arms -- the generation eval is what separates the levers.
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ts(){ date +%Y-%m-%d_%H:%M:%S; }
echo $$ > .run_separation.pid
PID="$(cat .drive_band_g0.pid)"

echo "### [$(ts)] watching chain pid $PID; will intercept before cell 2"
while kill -0 "$PID" 2>/dev/null; do
  if grep -q "next cell: band_L256_ep30_d30k_g0" drive_band_g0.log 2>/dev/null; then
    echo "### [$(ts)] chain announced cell 2 (30k x 30ep) — killing it, per plan"
    pkill -TERM -P "$PID" 2>/dev/null; kill -TERM "$PID" 2>/dev/null
    sleep 10
    pkill -f 'scripts/(run_sweep|recovery_check|band_ood_score)\.py' 2>/dev/null
    sleep 5
    break
  fi
  sleep 30
done
echo "### [$(ts)] chain done/stopped — cell 1 kept everything it finished"

# Wait for a clear GPU before starting the arms.
while pgrep -f 'scripts/(recovery_check|run_sweep|band_ood_score)\.py' >/dev/null 2>&1; do sleep 30; done

# Arm A (band_L256_ep10_d10k_g0_gs030, gamma_lo 0 at gamma* 0.03) was DROPPED.
# The 2x2 still closes by elimination: the published cell and g0 are two corners,
# and arm B moves gamma* alone. If arm B keeps the epoch-7 transition, gamma* is
# innocent and gamma_lo is what g0 lost. What the drop forfeits is the ability to
# see an INTERACTION between the two -- three corners fix the fourth only if the
# effects are additive.
for cell in band_L256_ep10_d10k_g005_gs99; do
  if [ -s "runs/$cell/eval.json" ]; then echo "### [$(ts)] $cell already done, skipping"; continue; fi
  echo "### [$(ts)] separation arm: $cell  (stage 1 only)"
  CELL="$cell" SKIP=recovery,ood,control ./drive_band_L256.sh
done

echo
echo "### [$(ts)] SEPARATION COMPLETE — 2x2 on KL_bi:"
uv run python - <<'PYEOF'
import json, pathlib
rows = [("band_L256_ep10_d10k",        "0.005", "0.030", "published baseline"),
        ("band_L256_ep10_d10k_g005_gs99","0.005","0.03573","arm B: gamma* only"),
        ("band_L256_ep10_d10k_g0",     "0.0",   "0.03573","both changes")]
print(f"{'cell':<32} {'g_lo':>6} {'g*':>8} {'KL_bi':>7} {'KL_tri':>7} {'H_ratio':>8}")
for name, lo, gs, note in rows:
    f = pathlib.Path(f"runs/{name}/eval.json")
    if f.is_file():
        d = json.loads(f.read_text())
        print(f"{name:<32} {lo:>6} {gs:>8} {d['bigram_kl']:>7.3f} {d['trigram_kl']:>7.3f} {d.get('H_ratio',float('nan')):>8.3f}  {note}")
    else:
        print(f"{name:<32} {lo:>6} {gs:>8} {'—':>7} {'—':>7} {'—':>8}  {note} (no eval.json)")
PYEOF

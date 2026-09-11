#!/usr/bin/env bash
# Gate the g0 chain on GENERATION quality.
#
# drive_band_L256.sh runs stages 1-4 unconditionally. Stage 1 (train + n=256
# generation eval) is the only stage worth paying for on a cell whose n-gram
# divergences are not competitive: recovery and OOD on a model that cannot
# generate are hours spent characterising a failure we already have.
#
# So: wait for stage 1's eval.json, read bigram_kl, and if it is above the
# threshold kill the chain before stage 2 starts. run_separation.sh watches
# `kill -0` on the same pid, so it takes over on its own the moment we do.
#
# Reference (same budget, 10ep/10k, L=256):
#   band gamma_lo=.005 gamma*=.03  KL_bi 0.265   <- the published band cell
#   Dirichlet FM                    KL_bi 0.959
#   EqM whole-path U[0,1]           KL_bi 1.582
# 0.5 sits clear of both the target and the failure mode.
set -uo pipefail
cd "$(dirname "$0")"
echo $$ > .gate_g0.pid
ts(){ date +%Y-%m-%d_%H:%M:%S; }

CHAIN="$(cat .drive_band_g0.pid)"
EVAL=runs/band_L256_ep10_d10k_g0/eval.json
THRESH=0.5

echo "### [$(ts)] gating chain $CHAIN on $EVAL (KL_bi <= $THRESH continues)"
while [ ! -s "$EVAL" ]; do
  kill -0 "$CHAIN" 2>/dev/null || { echo "### [$(ts)] chain exited before eval.json — nothing to gate"; exit 0; }
  sleep 15
done

read -r KLB KLT KLU HR <<<"$(uv run python -c "
import json;d=json.load(open('$EVAL'))
print(d['bigram_kl'], d['trigram_kl'], d['unigram_kl'], d.get('H_ratio',float('nan')))")"

echo "### [$(ts)] g0 GENERATION: KL_uni $KLU  KL_bi $KLB  KL_tri $KLT  H_ratio $HR"
echo "###           published band cell: KL_uni 0.0097  KL_bi 0.265  KL_tri 1.744  H_ratio 0.975"

if awk "BEGIN{exit !($KLB > $THRESH)}"; then
  echo "### [$(ts)] KL_bi $KLB > $THRESH — NOT competitive. Killing the chain before stage 2."
  echo "###           epoch_final.pt is kept; recovery/OOD can be run later if ever wanted."
  pkill -TERM -P "$CHAIN" 2>/dev/null; kill -TERM "$CHAIN" 2>/dev/null
  sleep 8
  pkill -f 'scripts/(recovery_check|band_ood_score)\.py' 2>/dev/null
  echo "### [$(ts)] chain stopped; run_separation.sh takes over"
else
  echo "### [$(ts)] KL_bi $KLB <= $THRESH — competitive. Letting stages 2-4 run."
fi

#!/usr/bin/env bash
# gamma_lo LADDER — generation only. Queued behind run_separation.sh.
# The gamma_lo=0 rung (arm A) was dropped; the ladder starts at the published
# cell. g0 (gamma_lo=0 but gamma*=0.03573) is the nearest off-spine anchor for
# what gamma_lo=0 does, and it lost the transition entirely.
#
# Spine: gamma* = gamma_hi = 0.03, clip 1.0, eta 0.005, 400 steps, 10ep/10k,
# seed 42, d1024/10L, B=8. ONLY gamma_lo moves, so the sampler config is
# identical across rungs and every delta is attributable to gamma_lo alone.
#
#   gamma_lo   kappa      W    cell                        provenance
#    0.00500   0.123   81.6%   band_L256_ep10_d10k         published, KL_bi 0.265
#    0.01208   0.300   61.4%   ..._g012_gs030              NEW
#    0.01900   0.475   35.1%   ..._g019_gs030              NEW
#
# Stage 1 only (SKIP=recovery,ood,control): a rung is only interesting if its
# n-gram divergences are competitive, so generation is the gate and nothing
# downstream is paid for until a rung clears it. Checkpoints are kept, so the
# recovery half of the support test can be added later on the same weights.
#
# What the generation column can and cannot say: it samples from x0 (gamma=0),
# which for gamma_lo > 0 is OUTSIDE the trained band by kappa noise radii. So a
# rung that degrades is ambiguous between "the field got worse" (Requirement N
# is not the binding constraint) and "the field is fine but the sampler cannot
# reach it" (Requirement S binds). Recovery would separate those because it
# starts inside the band; without it, a degrading rung means "this setting does
# not generate", which is the decision-relevant fact either way.
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ts(){ date +%Y-%m-%d_%H:%M:%S; }
echo $$ > .run_ladder.pid

SEP_PID="${SEP_PID:-2488500}"
echo "### [$(ts)] ladder queued; waiting on run_separation.sh (pid $SEP_PID)"
while kill -0 "$SEP_PID" 2>/dev/null; do sleep 60; done
echo "### [$(ts)] separation finished; waiting for a clear GPU"
while pgrep -f 'scripts/(recovery_check|run_sweep|band_ood_score)\.py' >/dev/null 2>&1; do sleep 30; done

for cell in band_L256_ep10_d10k_g012_gs030 band_L256_ep10_d10k_g019_gs030; do
  echo "### [$(ts)] ladder rung: $cell  (generation only)"
  CELL="$cell" SKIP=recovery,ood,control ./drive_band_L256.sh
done

echo
echo "### [$(ts)] LADDER COMPLETE"
uv run python - <<'PYEOF'
import json, pathlib
RUNGS = [("band_L256_ep10_d10k",           0.005,   0.123, 81.6),
         ("band_L256_ep10_d10k_g012_gs030",0.01208, 0.300, 61.4),
         ("band_L256_ep10_d10k_g019_gs030",0.019,   0.475, 35.1)]

def transition(name):
    """First epoch whose probe bigram_kl drops >25% below the running min."""
    h = pathlib.Path(f"runs/{name}/history.jsonl")
    if not h.is_file():
        return "—"
    ks = [json.loads(l).get("bigram_kl") for l in h.read_text().splitlines() if l.strip()]
    ks = [k for k in ks if k is not None]
    best = float("inf")
    for i, k in enumerate(ks, 1):
        if k < 0.75 * best and i > 1:
            return str(i)
        best = min(best, k)
    return "none"

print(f"{'gamma_lo':>8} {'kappa':>6} {'W':>6} | {'KL_uni':>7} {'KL_bi':>6} {'KL_tri':>7} {'H_ratio':>8} {'ep*':>4}")
for name, glo, kap, w in RUNGS:
    f = pathlib.Path(f"runs/{name}/eval.json")
    if f.is_file():
        d = json.loads(f.read_text())
        cols = (f"{d['unigram_kl']:>7.4f} {d['bigram_kl']:>6.3f} {d['trigram_kl']:>7.3f} "
                f"{d.get('H_ratio', float('nan')):>8.3f}")
    else:
        cols = f"{'—':>7} {'—':>6} {'—':>7} {'—':>8}"
    print(f"{glo:>8.5f} {kap:>6.3f} {w:>5.1f}% | {cols} {transition(name):>4}")
print("\ncompetitive means beating, at the same 10ep/10k budget and L=256:")
print("  Dirichlet FM      KL_uni 0.009  KL_bi 0.959  KL_tri 4.088")
print("  EqM whole path    KL_uni 0.018  KL_bi 1.582  KL_tri 5.042")
print("  published band    KL_uni 0.0097 KL_bi 0.265  KL_tri 1.744  H_ratio 0.975  ep* 7")
PYEOF

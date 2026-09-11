#!/usr/bin/env bash
# SEED REPLICATE of the published band cell — swapped in ahead of ladder rung 2.
#
# The published cell (gamma_lo .005, gamma* .030, seed 42) is the only config that
# has ever produced the epoch-7 transition and the KL_bi 0.265 headline. Three
# perturbations of it have now failed, so before the failures mean anything the
# 0.265 has to be shown to belong to the CONFIG rather than to that one run.
# band_L256_ep10_d10k_s43 differs from it in training.seed alone (verified).
#
# Ladder rung 2 (gamma_lo 0.019) is NOT run here; its cell stays in the sweep and
# can be picked up later if the replicate holds.
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ts(){ date +%Y-%m-%d_%H:%M:%S; }
echo $$ > .run_seed43.pid

echo "### [$(ts)] waiting for ladder rung 1 to finish before taking the GPU"
while pgrep -f 'scripts/(run_sweep|recovery_check|band_ood_score)\.py' >/dev/null 2>&1; do sleep 30; done
echo "### [$(ts)] GPU clear — starting the seed replicate (generation only)"

CELL=band_L256_ep10_d10k_s43 SKIP=recovery,ood,control ./drive_band_L256.sh

echo
echo "### [$(ts)] SEED REPLICATE COMPLETE"
uv run python - <<'PYEOF'
import json, pathlib
ROWS = [("band_L256_ep10_d10k",           "0.005", "0.030", "42", "published"),
        ("band_L256_ep10_d10k_s43",       "0.005", "0.030", "43", "SEED REPLICATE"),
        ("band_L256_ep10_d10k_g012_gs030","0.01208","0.030","42", "ladder kappa=0.30"),
        ("band_L256_ep10_d10k_g005_gs99", "0.005", "0.03573","42","arm B: gamma* pkg"),
        ("band_L256_ep10_d10k_g0",        "0.0",   "0.03573","42","both changes")]

def transition(name):
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

print(f"{'cell':<32} {'g_lo':>7} {'g*':>8} {'seed':>4} | "
      f"{'KL_uni':>7} {'KL_bi':>6} {'KL_tri':>7} {'H_rat':>6} {'ep*':>4}")
for name, lo, gs, sd, note in ROWS:
    f = pathlib.Path(f"runs/{name}/eval.json")
    if f.is_file():
        d = json.loads(f.read_text())
        cols = (f"{d['unigram_kl']:>7.4f} {d['bigram_kl']:>6.3f} {d['trigram_kl']:>7.3f} "
                f"{d.get('H_ratio', float('nan')):>6.3f}")
    else:
        cols = f"{'—':>7} {'—':>6} {'—':>7} {'—':>6}"
    print(f"{name:<32} {lo:>7} {gs:>8} {sd:>4} | {cols} {transition(name):>4}  {note}")

p = pathlib.Path("runs/band_L256_ep10_d10k_s43/eval.json")
if p.is_file():
    kb = json.loads(p.read_text())["bigram_kl"]
    print()
    if kb < 0.5:
        print(f"VERDICT: replicate KL_bi {kb:.3f} < 0.5 — the transition REPRODUCES.")
        print("  The published config is real and the island is narrow; the rungs'")
        print("  failures are the finding. docs/band_geometry_theory.md 6.4 stands.")
    else:
        print(f"VERDICT: replicate KL_bi {kb:.3f} >= 0.5 — the transition DID NOT reproduce.")
        print("  The epoch-7 event was run-to-run noise, not a property of the config.")
        print("  Retract: docs/band_eqm.md sec 8, the SNR reading in band_geometry_theory 6.4,")
        print("  and the band arm's KL_bi 0.265 headline. A third seed would confirm.")
PYEOF

#!/usr/bin/env bash
# Full Objective-3 benchmark: OOD detection + healing. ~5-6 h on one GPU, sequential.
#
# Launch and walk away:
#     nohup ./run_bench_all.sh > bench_all.log 2>&1 &
#     tail -f bench_all.log
#
# RESUME after a crash / power-off / Ctrl-C:
#     nohup ./run_bench_all.sh --resume > bench_all_resume.log 2>&1 &
#
# Each runner checkpoints its manifest after EVERY arm, so --resume skips the arms
# already recorded `done` **with an identical command line** and re-runs the rest.
# Keying on the manifest (not just "an output file exists") is what makes it safe:
# a stale JSON from older code has a different recorded cmd, so it is NOT trusted --
# it gets re-run. Without --resume, everything is re-run from scratch.
#
# It OVERWRITES bench_ood/, bench_heal/ and heal_poc_insulin/*.json. That is intended:
# those directories are currently STALE / MIXED and must not be trusted (see below).
#
# What changed since they were last produced (2026-07-13):
#   * P/R/F1 at a clean-calibrated threshold for every detector (AUROC alone only
#     ranks; it never says what a deployed threshold actually flags). Same threshold
#     convention as healing's loc_precision/loc_recall, so the two benches now compare.
#   * WORD-level metrics — the common unit between our CHAR-level detector and the
#     BPE-level GPT-2 baselines. Pooling up is exact; the old BPE->char spreading was
#     not. Word-level is now the headline localization metric.
#   * Spilled energy actually computes spilled energy. It was a per-token NLL. Now the
#     cross-step dE of Minut et al. (ICLR 2026), validated against the authors' code.
#     A new gpt2_nll arm keeps the honest same-step comparator.
#   * Healing operating point: cal-set F0.5 (precision-weighted; healing is
#     damage-averse) instead of F1, and heal_protein_poc now sweeps every FPR so the
#     insulin rows use the same least-damaging selection as the text8 arms.
#
# Re-runnable: everything is idempotent (each arm overwrites its own JSON).
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RESUME=""
[ "${1:-}" = "--resume" ] && RESUME="--resume"

ts() { date -Is; }

if [ -n "$RESUME" ]; then
  echo "######## [$(ts)] RESUMING — completed arms will be skipped ########"
fi

echo "######## [$(ts)] 1/4  BENCH_OOD — detector sweep (9 arms) ########"
uv run python scripts/run_bench_ood.py $RESUME               || echo "!! bench_ood returned $?"

echo "######## [$(ts)] 2/4  AGGREGATE OOD -> bench_ood/RESULTS.md + heatmaps ########"
uv run python scripts/bench_aggregate.py --bench-dir bench_ood || echo "!! aggregate returned $?"
# heatmaps render each detector in ITS OWN units: char cells for the flow-matching
# detectors, BPE-token cells for the GPT-2 baselines (red box = unit overlapping a
# corrupted char). Never GPT-2 on char cells — that is the attribution artifact.
uv run python scripts/plot_heatmaps_all.py --bench-dir bench_ood || echo "!! heatmaps returned $?"

echo "######## [$(ts)] 3/4  BENCH_HEAL — 6 text8 arms + 3 insulin arms ########"
uv run python scripts/run_bench_heal.py $RESUME             || echo "!! bench_heal returned $?"

echo "######## [$(ts)] 4/4  DONE ########"
echo "Results:"
echo "  bench_ood/RESULTS.md    + bench_ood/figs/     (manifest.json + repro.patch)"
echo "  bench_heal/RESULTS.md   + bench_heal/figs/    (manifest.json + repro.patch)"

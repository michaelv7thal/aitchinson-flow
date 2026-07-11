#!/usr/bin/env bash
# Rerun the full OOD-detection + healing routine against the epoch_best.pt
# checkpoint of the full-text8 L256 DirichletFM run, mirroring the previous
# epoch_final.pt routine (test split, fine corruption grid) but writing to
# *_best output dirs so the epoch_final results stay side-by-side.
#
#   OOD:  bayeslin_pca0 / nll / gpt2_se / latent_split
#   HEAL: heal_L256_test_blr / heal_L256_test_nll / heal_iterative_nll
#
# Continues past a failed arm (logs it) so one OOM doesn't sink the suite.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT="runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM/epoch_best.pt"
PY="uv run python"
SPLIT=test
SCHEMES="replace,shuffle,both"
RATES="0.05,0.1,0.15,0.2,0.25,0.3,0.5,0.7,1.0"
N=256
FITSEQS=512

OOD="ood_out_best"
HEAL="heal_out_best"
mkdir -p "$OOD/bayeslin_pca0" "$OOD/nll" "$OOD/gpt2_se" "$OOD/latent_split" "$HEAL"

ts(){ date +%H:%M:%S; }
run(){ # name logfile cmd...
  local name="$1" log="$2"; shift 2
  echo "### [$(ts)] START $name -> $log"
  if "$@" > "$log" 2>&1; then echo "### [$(ts)] OK    $name";
  else echo "### [$(ts)] FAIL  $name (see $log)"; fi
}

echo "###### HEAL+OOD BEST-CKPT RERUN — $CKPT ######"

# ---------------- OOD detection ----------------
run "ood/bayeslin" "$OOD/bayeslin_pca0/sweep_test.log" \
  $PY scripts/ood_bayes_linear.py --ckpt "$CKPT" --pca-dim 0 \
    --fit-seqs "$FITSEQS" --n "$N" --split "$SPLIT" \
    --schemes "$SCHEMES" --rates "$RATES" \
    --out "$OOD/bayeslin_pca0/bayes_linear_sweep.json"

run "ood/nll" "$OOD/nll/sweep_test.log" \
  $PY scripts/ood_denoiser_nll.py --ckpt "$CKPT" \
    --fit-seqs "$FITSEQS" --n "$N" --split "$SPLIT" --t-nll 3.0 --t-var 7.5 \
    --schemes "$SCHEMES" --rates "$RATES" \
    --out "$OOD/nll/denoiser_nll_sweep.json"

run "ood/gpt2_se" "$OOD/gpt2_se/sweep_test.log" \
  $PY scripts/ood_gpt2_spilled_energy.py --ckpt "$CKPT" --split "$SPLIT" \
    --fit-seqs "$FITSEQS" --n "$N" --schemes "$SCHEMES" --rates "$RATES" \
    --out "$OOD/gpt2_se/gpt2_spilled_energy_sweep.json"

run "ood/latent_split" "$OOD/latent_split/latent_split.log" \
  $PY scripts/plot_latent_split.py --ckpt "$CKPT" \
    --out-dir "$OOD/latent_split" --fit-seqs 192 --n "$N" --rate 0.3

# ---------------- Healing ----------------
run "heal/linear" "$HEAL/heal_L256_test_blr.log" \
  $PY scripts/heal_dirichlet.py --ckpt "$CKPT" --split "$SPLIT" \
    --localizer linear --n-demo 100 --n-seeds 5 --corrupt-rate 0.15 \
    --out "$HEAL/heal_L256_test_blr.json"

run "heal/nll" "$HEAL/heal_L256_test_nll.log" \
  $PY scripts/heal_dirichlet.py --ckpt "$CKPT" --split "$SPLIT" \
    --localizer nll --t-nll 3.0 --n-demo 100 --n-seeds 5 --corrupt-rate 0.15 \
    --out "$HEAL/heal_L256_test_nll.json"

run "heal/iterative" "$HEAL/heal_iterative_nll.log" \
  $PY scripts/heal_iterative.py --ckpt "$CKPT" --split "$SPLIT" \
    --localizer nll --iters 5 --n-demo 32 --n-seeds 3 --corrupt-rate 0.15 \
    --target-fpr 0.02 --nfe 100 \
    --out "$HEAL/heal_iterative_nll.json"

echo "###### DONE $(ts) -> $OOD / $HEAL ######"

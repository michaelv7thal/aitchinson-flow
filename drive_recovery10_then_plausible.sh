#!/usr/bin/env bash
# Sequencer: put the 10-epoch recovery ladder on the GPU ahead of the plausible
# latent stage, without losing any finished work.
#
# Order after this script starts:
#   1. let drive_latent_all.sh finish its nine-rate cheap ladder (app:latent-ladder)
#   2. stop it just as it enters the plausible stage (that stage writes nothing
#      until it completes, so at most a few minutes are recomputed)
#   3. run the fine α-ladder for the benchmark-budget arm — the last five
#      [TODO: ?] cells of tab:recovery, plus an independent re-derivation of the
#      five coarse values already published for that arm
#   4. relaunch drive_latent_all.sh, which skips stages A and B by their existing
#      outputs and runs only the plausible stage
#
# Rationale for the order: the recovery ladder is bounded (10 α at ~22 min), the
# plausible stage is not (48 model-scored candidates per swapped word slot).
set -uo pipefail
cd /home/michael/projects/aitchinson-flow

LOG=bench_ood/_driver
mkdir -p "$LOG"
ts(){ date +%Y-%m-%d_%H:%M:%S; }

echo "### SEQUENCER START $(ts)"

# ---- 1. wait for the cheap ladder to reach its last rate --------------------
while [ ! -s bench_ood/latent_ladder/rate_1.0/latent_split.json ]; do
  echo "### [$(ts)] cheap ladder still running — waiting"
  sleep 120
done
echo "### [$(ts)] cheap ladder complete"

# ---- 2. stop the latent driver before it sinks hours into plausible ---------
pkill -f "drive_latent_all.sh"
pkill -f "plot_latent_split.py"
sleep 5
echo "### [$(ts)] latent driver stopped"

# ---- 3. the benchmark-budget recovery ladder -------------------------------
echo "### [$(ts)] 10-epoch recovery ladder start"
bash runs/sflm_bench_a100_20g_L256_d1280L14_full/_driver/drive_recovery_fine.sh \
     runs/sflm_bench_a100_20g_L256/DirichletFM \
     > "$LOG/recovery_fine_10ep.out" 2>&1
echo "### [$(ts)] 10-epoch recovery ladder exit=$?"
tail -3 "$LOG/recovery_fine_10ep.out"

# ---- 4. hand the GPU back to the plausible stage ---------------------------
echo "### [$(ts)] relaunching latent driver for the plausible stage"
nohup bash drive_latent_all.sh > "$LOG/latent_ladder_resume.out" 2>&1 &
echo "### SEQUENCER DONE $(ts) (plausible now running under pid $!)"

# Paper map: every table and figure in the capstone paper, and what backs it

The paper is *Flow Matching on the Simplex for Character-Level Text* (the
`capstone-paper` repository). This map is the human-readable form of the
executable matrix `claims.tsv` (repo root), which `check_claims.py` re-reads
against the artifacts named here: **1597 rows, 1591 machine-checked passes, 6
honest weak rows, 0 failures** (2026-09-11).

Paper locations are given as `file:label` (for example
`chapters/results.tex:tab:gen`), not as line numbers: labels survive editing,
line numbers do not. A row whose number the paper does not print carries the
label of the passage it backs and says so in its `notes`.

Check any number yourself:

```bash
python3 check_claims.py claims.tsv          # <1 s, CPU, no venv needed
```

## The matrix

Claim-row prefixes: G = generation/recovery/latent, N = negative result,
D = detection, H = repair, X = appendix/config, P/E = provenance anchors,
M = matched benchmark budget.

| Paper label | Artifact(s) | Producing script | Checkpoint | Rows |
|---|---|---|---|---|
| `tab:setup` (Model selection column) | `runs/sflm_bench_a100_20g_L256/bench_budget.json` (d_model, layers/heads, epochs, windows, characters, batch, seed, and the shared n=256 / 200 steps, read out of each arm's checkpoint `cfg` and `eval_all.json`; `train_meta.json` carries none of it) | `scripts/bench_budget.py` | all nine benchmark arms | M1–M15 |
| `tab:gen` | `runs/sflm_bench_a100_20g_L256/<ARM>/{eval_all,recovery}.json`; lower block adds `<...>/DirichletFM_ep30_d30k` and `runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_converge` | `scripts/eval_generation.py`, `scripts/recovery_check.py` | per-arm `epoch_final.pt` | G1–G55 |
| `tab:recovery` | `<budget>/recovery_fine/alpha_*.json` | `scripts/recovery_check.py` | 10 ep / 30 ep / converge | G56–G125 |
| `fig:recovery-curve` | same `recovery_fine` trees | `scripts/paper_figs/fig_recovery_curve.py` | — | — |
| `tab:latent`, `fig:latent-seq`, `fig:latent-token` | `bench_ood_final/latent_split_all/`, `latent_split_plausible/` (see `bench_ood_final/manifest_addenda.json`) | `scripts/plot_latent_split.py`, `paper_figs/fig_latent_split.py` | converge | G126–G149 |
| `tab:ablation2x2` | `runs/{compu_mse_det,compu_mse_thick,dsmx_clr_det,dsmx_clr_dir}/` — **KL columns from `eval.json`, Δ from `recovery.json`** (separate sampler draws; do not cross-read) | `scripts/run_matched_recovery.sh`, `run_dsm_rescaled.sh` | dev-scale | N1–N14 |
| `tab:anneal-history` (app:anneal) | `runs/sflm_bench_a100_20g_L256_d1280L14_full/anneal_history.json` (per-epoch train/val loss of the four legs, parsed from the `_driver/train_DirichletFM*.log` tqdm bars) | `scripts/anneal_summary.py` | 10-epoch / best-val / converge marked | A1–A46 (+A47–A55 prose) |
| `tab:ckpt-compare` (app:anneal) | `bench_ood_final/_compare/checkpoint_comparison.json` (reads `bench_ood`+`bench_heal` = epoch_best vs `bench_ood_final`+`bench_heal_final` = converge; human-readable `docs/anneal_and_checkpoint_comparison.md`) | `scripts/anneal_summary.py` | best-val ep15 vs converge ep23 | A56–A93 |
| `tab:hilbert-null` | adds `runs/compu_hilbert_{det,thick}/recovery.json`; procedure: `RUNBOOK_COMPOSITIONAL_EQM_TEST.md` (**do not run its §6**) | as above | dev-scale | N15–N31 |
| `fig:energy-landscape` | `runs/compu_mse_det/{native_radius,recovery_native_path*}.json` | `paper_figs/fig_energy_landscape.py`, producer `scripts/recovery_native_path.py` | `compu_mse_det` | N65–N72 |
| `fig:ood-heatmap`, `-30` | `bench_ood_final/nll/heatmap_examples.json` | `paper_figs/fig_ood_heatmap.py`, producer `scripts/dump_heatmap_examples.py` | converge | D319ff |
| `tab:ood-word`, `tab:ood-prf`, `tab:ood-seq` (app:ood-supp) | `bench_ood_final/{nll,blr,blr_adv,blr_fi,bgmm,gpt2_se,gpt2_nll}/*_sweep.json` (manifest arms) | the six `scripts/ood_*.py` in `manifest.json` | converge | D1–D290, D372ff |
| `fig:ood-word-auroc`, `fig:ood-seq` | same sweeps | `paper_figs/fig_ood_word_auroc.py`, `fig_ood_seq.py` | — | — |
| `tab:ood-plausible` | `bench_ood_final/plausible/plausible_swap.json` (+ `plausible_freqmatched/` control, t_eval caveat in the addenda) | `scripts/ood_plausible_swap.py` | converge | D95ff |
| `tab:ood-transfer` | `bench_ood_final/transfer/semaglutide_transfer_gpt2.json` | `scripts/eval_detector_transfer.py` | converge | D291–D302 |
| `tab:heal-synth`, `tab:heal-word`, `tab:heal-fpr` | `bench_heal_final/{nll,blr,bgmm}_{replace,falseinfo}.json` — pointer `sweep[target_fpr=0.02]`, **never the `selected_fpr` field** (that records the deployable operating point, not the reported one) | `scripts/heal_dirichlet.py` | converge | H1–H34, D360ff (word), D166ff (fpr) |
| *(no paper label)* — insulin healing track | `heal_poc_insulin_final/*.json`; input text: `heal_poc_insulin/insulin_article_extract.json` | `scripts/heal_protein_poc.py` | converge | H35–H53 |
| `fig:latent-ladder`, `tab:latent-ladder`, `tab:latent-plausible` | `bench_ood_final/latent_ladder{,_plausible}/rate_*/latent_split.json` | `scripts/plot_latent_split.py`; figure `paper_figs/fig_latent_ladder.py` | converge | X28–X207 |
| `tab:config` | `train_meta.json` files, both manifests, `src/aitchinson_flow/config.py`, `batch_probe.json`, `runs/vae_a100_20g_L256/config.json` (sole source of the VAE row), `bench_ood_final/tsweep/t_selection.json` (the read-out-time selection) | — | — | X212–X233, P6–P10 |
| corpus constants (Σp², ‖μ₁‖, entropy, word counts) | `runs/text8_unigram_stats.json` | `scripts/dump_text8_unigram_stats.py` | — | E1–E5, H54 |
| `fig:image-text` | none (synthetic schematic) | `paper_figs/fig_image_text.py` | — | — |
| `tab:eqm-band-bench`, `fig:band-probe` | `runs/band_L256_ep10_d10k{,_g012_gs030,_g005_gs99,_g0,_s43}/eval.json` and `…/config.json` (**never `band_*/eval.json` for a recovery number**) | `scripts/band_geometry.py`, `paper_figs/fig_band_probe.py` | per-run `epoch_final.pt` | B1–B44, B75–B79 |
| `tab:eqm-band-recovery` | `runs/band_L256_ep10_d10k/recovery_bandmap_a{03,05,06,07,08,10}.json` | `scripts/recovery_check.py` | band run | B45–B58 |
| `tab:ood-seq` (app:ood-supp) | the seven `bench_ood_final/*/*_sweep.json` detector arms | the `scripts/ood_*.py` in `manifest.json` | converge | D43–D527 |
| `tab:ood-bpe` (app:ood-supp) | `bench_ood_final/gpt2_nll/{gpt2_nll_sweep,bpe_token_counts}.json` | `scripts/ood_gpt2_spilled_energy.py` (the `gpt2_nll` arm of `manifest.json`), counts from `scripts/dump_bpe_counts.py` | converge | D389–D415 |
| `tab:ood-varhead` (app:ood-supp) | `bench_ood_final/blr/bayes_linear_sweep.json` | `scripts/ood_bayes_linear.py` | converge | D416–D420 |
| `tab:ood-threshold` (app:ood-threshold) | `bench_ood_final/logreg_fi_r15/bayes_linear_logreg_fi_r15_sweep.json` | `scripts/ood_bayes_linear.py` (logistic head, see `logreg_fi_r15.log`) | converge | D566–D569 |
| `tab:ood-swap-surprise` | `bench_ood_final/plausible/word_level_surprise.json` | `scripts/word_level_surprise.py` | converge | D148–D565 |
| `tab:transfer-topic`, `tab:transfer-entity` | `runs/text8_unigram_stats.json` (text8 coverage of the topic and of the drug) | `scripts/dump_text8_unigram_stats.py` | — | H54 |
| `app:recovery` (prose, eqs) | `scripts/recovery_check.py` (per-arm perturbation branches, lines ~352-470) + the samplers `models/{dirichlet_fm,sfm,fisher_fm,dfm,fm_clr,sflm,eqm,eqm_ae}.py`; rho-bar from `runs/sflm_bench_a100_20g_L256/<ARM>_gp1p0/recovery.json rows[alpha=1.0].sigma_perturb` | `scripts/recovery_check.py` | per-arm `epoch_final.pt` | G226–G228 |

`fig:backbone` and `fig:data-pipeline` are hand-drawn TikZ in the paper
repository and have no dependency here.

## Checkpoints of record

| md5 | path | carries |
|---|---|---|
| `9946dce94cb2051de0e288e4caa3ffbd` | `runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_converge/epoch_final.pt` | **the paper's model**: full-corpus generation row, tab:recovery "full" column, every detector, every repair result, all latent readouts. A genuine annealed final (the last training leg ran with no early stop). |
| `f1d809e8a42d7c3c0a44550c119f9cbe` | `…_d1280L14_full/DirichletFM/epoch_best.pt` | the superseded `bench_ood/`+`bench_heal/` trees (best-val checkpoint, epoch 15 of 23), which now back only the best-validation rows of `tab:ckpt-compare` via `bench_ood_final/_compare/checkpoint_comparison.json` |

`epoch_final.pt` does **not** mean "last epoch" in this repository: with
`--early-stop-patience` set, the trainer restores the best-validation
checkpoint under that name (`src/aitchinson_flow/training/runner.py`).
Identify checkpoints by md5 against the manifests, never by filename.

## Traps that have actually bitten

- **Dirichlet FM ≠ Discrete FM.** The arm directory literally named `DFM/` is
  *Discrete* Flow Matching; the paper's "DFM" acronym expands to *Dirichlet*
  and lives in `DirichletFM/`. The paper's selected generator is Dirichlet FM.
- **Arm-name → paper-row map for `tab:gen`** (counterintuitive):
  `EqM_OneHot_gp1p0` = "EqM, deterministic clr" · `EqM_gp1p0` =
  "EqM, Dirichlet-thickened" · `EqMAE_gp1p0` = "EqM, VAE latent" ·
  `SFM` = "Statistical FM (Cheng)" ·
  `FisherFM` = "Fisher FM (Davis, the **smoothed** target)" · `SFLM` =
  "Hyperspherical flow" · `DFM` = "Discrete FM".
- **`runs/band_{g05,strict}/`** backs nothing the paper prints any more: the
  0.92 / 0.91 pair those `eval_fixed.json` files hold left `results.tex`, and
  no `claims.tsv` row reads the directory. The published band numbers are the
  `runs/band_L256_ep10_d10k*` trees (`tab:eqm-band-bench`). The sibling
  `eval.json` files and `runs/band_report.json` additionally hold **stale
  values that contradict the paper** (a sampler run that never applied the
  retargeted step size). Never read any of them as readouts.
- **`bench.log:11` misnames the SFLM arm's budget.** The header reads
  `=== TRAIN SFLMEBM [a100_20g_L256] (10 ep, d_model=768/8L, B=8, n_train_windows=10000) ===`,
  and `SFLM/PROVENANCE.md` names that same auto-train as the origin of the surviving
  `epoch_10.pt`. Read literally it would put the Hyperspherical flow row off the matched
  budget. The checkpoint's own `cfg` says 1024 / 10 layers / 16 heads and `_driver` carries
  two matching `TRAIN SFLM ... d_model=1024/10L/16H` headers. Resolve an arm's budget by
  its checkpoint `cfg`, never by that header (rows M13–M15).
- **Value decoys.** `runs/sflm_bench_a100_20g_L256/generation_eval_sflm_sfm.json`
  (superseded parallel SFLM/SFM readout), `runs/_archive`'s
  `…_d1280L14/DirichletFM_ep30_d30k` (same arm name and same KL_uni digit as
  the real 30-ep row's directory), and `runs/comp_mse_seed42` (collides with
  the published `compu_mse_thick` cell at 2 significant figures). Resolve
  pointers by path, never by value.
- **Dating key for detection-era documents:** shuffle word-max 0.963 and
  false-info NLL 0.813 mark the superseded `bench_ood/`; 0.967 and 0.807 mark
  `bench_ood_final/`. Replace at 0.981 agrees across both to three digits and
  can date nothing; GPT-2 rows are byte-identical across trees.
- **NaN in artifacts.** The sweeps store undefined cells (e.g. localization
  AUROC at replace rate 1.0) as bare `NaN`, which Python's `json` and `jq`
  accept but strict JSON parsers (JS, Go, Rust) reject. The NaN is the
  meaningful value; use Python or jq.
- **`ScoreDSM_CLR` artifacts** carry NaN throughout their energy/curvature
  blocks by construction (score models have no energy readout); the published
  columns are finite.

# runs/_archive — superseded run data, moved 2026-08-20

- `sflm_bench_a100_20g_L256_d1280L14/`, `sflm_bench_a100_20g_L256_d1280L14_b16/` —
  two UNREPORTED scaling points of the big backbone (30ep/30k: KL_bi 0.263,
  *worse* than the reported 0.208 at d1024; 50ep/50k: KL_bi 0.141, near the
  full-corpus 0.130 without the full corpus). **Decoy warning:** the first
  tree's arm `DirichletFM_ep30_d30k` shares its name AND its KL_uni digit with
  the directory the paper's 30-epoch row actually reads
  (`runs/sflm_bench_a100_20g_L256/DirichletFM_ep30_d30k`, KL_bi 0.208). Trace
  the paper's row there, never here. Checkpoints were already absent when
  these trees were archived.
- `generation_eval_sflm_sfm.json` — a superseded parallel readout of the SFLM
  and SFM arms whose values near-collide with `tab:gen` (SFLM 1.348 vs the
  published 1.378 from the per-arm `eval_all.json`). Kept as history; never a
  source.

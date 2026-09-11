# Capstone follow-up experiments — outputs

This directory holds artifacts from the protocol in
`/workspace/CAPSTONE_EXPERIMENTS.md`. Layout:

```
runs/capstone/
├── R/                  # Phase R — SDE sampling on continuous methods (eval-only)
│   ├── *.json            per-cell eval results
│   ├── phaseR_sde.json   aggregated sweep summary
│   └── phaseR_sde.png    figure: KL_bi vs noise scale α
├── S/                  # Phase S — K=2 binary collapse
├── T/                  # Phase T — conservative-gradient regression
├── U/                  # Phase U — cascade-audit four-AUROC table
├── V/                  # Phase V — SAPLMA-style probe replication
├── checkpoints/        # NEW model checkpoints produced for these phases
│   ├── eqm_data50k_ep5_v2/    re-trained base EqM (replaces missing v2 ckpt)
│   ├── dfm_data50k_ep5_v2/    re-trained base DFM
│   ├── fmclr_data50k_ep5_v2/  re-trained base FMonCLR
│   ├── lkflow_data50k_ep5/    re-trained base LogitKLFlow
│   ├── eqm_K2_data50k_ep5/    Phase S
│   ├── dfm_K2_data50k_ep5/    Phase S
│   ├── fmclr_K2_data50k_ep5/  Phase S
│   ├── lkflow_K2_data50k_ep5/ Phase S
│   ├── eqm_consgrad_*/        Phase T
│   ├── aud_gpt2_ctx/          Phase U auditor (replaces missing ckpt)
│   └── saplma_wiki/           Phase V
└── _meta/
    ├── PHASE_NOTES.md   running per-phase notes; each cell appended live
    └── README_dirty.md  what was already in /workspace/runs/ before this protocol
```

Each phase folder contains its `phase{X}_*.json` (numbers), `phase{X}_*.md`
(human-readable result block), and `phase{X}_*.png` (figure when applicable).

The top-level `runs/DECISION_LOG.md` is appended per cell as the protocol
specifies (hypothesis / result / decision / next).

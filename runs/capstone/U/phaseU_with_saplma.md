# Phase U — cascade audit four-AUROC table

| method | AUROC₁ corrupted aligned | AUROC₂ uncorrupted aligned | AUROC₃ corrupted shuffled | AUROC₄ uncorrupted shuffled |
|---|---:|---:|---:|---:|
| SE | 0.968 | 0.639 | 0.967 | 0.638 |
| topk_entropy | 0.665 | 0.632 | 0.668 | 0.631 |
| linear_probe | 0.976 | 0.783 | 0.975 | 0.786 |
| blr_laplace | 0.853 | 0.697 | 0.847 | 0.699 |
| svgp | 0.861 | 0.723 | 0.858 | 0.724 |
| eqm_auditor | 0.994 | 0.975 | 0.994 | 0.975 |
| saplma | 0.992 | 0.759 | 0.993 | 0.761 |

# Phase U — cascade audit four-AUROC table

| method | AUROC₁ corrupted aligned | AUROC₂ uncorrupted aligned | AUROC₃ corrupted shuffled | AUROC₄ uncorrupted shuffled |
|---|---:|---:|---:|---:|
| SE | 0.968 | 0.639 | 0.967 | 0.638 |
| topk_entropy | 0.665 | 0.632 | 0.668 | 0.631 |
| linear_probe | 0.966 | 0.772 | 0.963 | 0.774 |
| blr_laplace | 0.883 | 0.694 | 0.877 | 0.697 |
| svgp | 0.861 | 0.723 | 0.858 | 0.724 |
| eqm_auditor | 0.994 | 0.975 | 0.994 | 0.975 |

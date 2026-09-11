## DFM auditor — three-way mode ablation

| mode | params (M) | epochs | best AUROC | (epoch) | last train CE | ΔE AUROC | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `off` | 19.3 | 10 | **0.7723** | 6 | 1.559 | 0.713 | 0.246 |
| `hidden_only` | 19.4 | 10 | **0.8114** | 10 | 1.091 | 0.713 | 0.249 |
| `product_concat` | 19.4 | 10 | **0.7977** | 10 | 0.986 | 0.713 | 0.255 |

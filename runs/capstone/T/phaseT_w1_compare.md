# Phase T — W1 6-cell comparison

| family | sampler | KL_uni | KL_bi | KL_tri | H_ratio |
|---|---|---:|---:|---:|---:|
| eqm | Euler on f | 0.0850 | 1.6472 | 5.9238 | 0.930 |
| eqm | Euler on grad | 0.0269 | 1.3989 | 5.7782 | 0.989 |
| eqm | NAG | 0.0343 | 1.4792 | 5.8061 | 0.984 |
| consgrad | Euler on output | 0.0085 | 0.4078 | 2.5541 | 0.988 |
| consgrad | Euler on output (use_grad) | 0.0109 | 0.4136 | 2.5435 | 0.984 |
| consgrad | NAG | 0.2767 | 3.6082 | 7.4078 | 0.868 |

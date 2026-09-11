# Hilbert Flow Matching — standalone experiment

Self-contained exploratory test of **Hilbert Flow Matching** on character-level
text. Independent of `aitchinson_flow/`; pure PyTorch, no Lightning, no Hydra.

## Setup

The path is the **log-linear (Aitchison) geodesic** on the simplex:

    p_t = softmax((1 - t) log p_0 + t log p_1)

with uniform source `p_0 = 1/K`, label-smoothed target `p_1`. The model is an
**x_1-prediction** transformer: it consumes `log p_t` and `t`, and outputs
`log p_1_hat`. Loss is the **soft Hilbert projective metric**

    d_H^τ(p_hat, p_1) = τ·lse(r/τ) + τ·lse(-r/τ),   r = log p_hat - log p_1.

Sampling walks the same log-linear path with re-prediction at each step.

Vocab: 26 lowercase letters + space, `K = 27`. Sequence length `L = 64`.

## Layout

```
hilbert_fm/
├── README.md      ← this file
├── data.py        ← text8 / tiny-shakespeare loader, char tokenizer
├── model.py       ← transformer + softmax head
├── path.py        ← log-linear path, soft Hilbert loss, advance step
├── train.py       ← training loop (config dict at top)
├── sample.py      ← iterative re-prediction sampling
└── diagnose.py    ← required diagnostics (A–F) + RESULTS.md inputs
```

## Run

```bash
python -m hilbert_fm.train                              # 20k steps -> runs/default/final.pt
python -m hilbert_fm.sample hilbert_fm/runs/default/final.pt
python -m hilbert_fm.diagnose hilbert_fm/runs/default/final.pt
```

`train.py` writes `final.pt`, `history.json`, and periodic `ckpt_step{N}.pt`
into `runs/default/`. `diagnose.py` writes plots and `summary.json` into
`runs/default/diagnostics/`.

## Hyperparameters

The defaults from the spec are wired into `train.py:CONFIG`:

```python
K=27, L=64, d_model=256, n_layers=4, n_heads=4,
eps_smooth=0.01, tau=0.3,
batch_size=32, lr=3e-4, weight_decay=0.01,
n_steps=20_000, t_bias_power=0.5,
n_sample_steps=50, t_sample_max=0.99,
```

Fallback knobs if results are degenerate (in order): lower `tau` to 0.1,
raise `eps_smooth` to 0.05, deeper model + longer training (6 layers, 50k),
stronger time bias `t_bias_power=0.3`.

## Diagnostics (mandatory)

`diagnose.py` produces:

- **A** training loss curve
- **B** per-`t` bucket validation loss (`[0,⅓), [⅓,⅔), [⅔,1)`)
- **C** sample-vs-data unigram + KL, plus bigram coverage
- **D** five printed sample strings
- **E** reconstruction accuracy from `t = 0.5` (start from real text → re-noise → re-sample)
- **F** Hilbert distance between successive sampling iterates

## Acceptance criteria

- Loss drops ≥ 5× and plateaus.
- Sample unigram KL vs data < 0.5.
- ≥ 50% of sampled bigrams appear in training data.
- Reconstruction from `t = 0.5` ≥ 80% character accuracy.
- Hilbert distance trajectory monotone non-increasing.

## Pitfalls baked into the implementation

- All operations in log-space; targets and interpolants go through `log_softmax`.
- `t` is sampled in `(0, 0.99]` — never exactly 1, where the gradient vanishes.
- Soft Hilbert is computed on `log p_1_hat` vs `log p_1` (distributions), not
  on tangent vectors / velocities.
- Time conditioning is added *after* input projection so the model can
  distinguish `t = 0` (uniform input) from later `t`.
- Attention has no causal mask (this is non-autoregressive).

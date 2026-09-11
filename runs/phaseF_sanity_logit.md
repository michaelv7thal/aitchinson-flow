# Phase F sanity checks — aud_gpt2_logit

- ckpt: `runs/aud_gpt2_logit/epoch_final.pt`
- n_total=300, n_train=240, n_val=60, ctx_mode=off

## 1. Train vs held-out AUROC

|   stat   |  train  |   val   |  gap  |
|----------|--------:|--------:|------:|
| Seq_EqM_Egrad²         | 0.9716 | 0.9417 | +0.0300 |
| Seq_SE                 | 0.9995 | 0.9986 | +0.0009 |
| Tok_EqM_Upos           | 0.9654 | 0.9481 | +0.0173 |
| Tok_SE                 | 0.9674 | 0.9677 | -0.0003 |

**Verdict:** there is a noticeable train-vs-val gap; the AUROC on the full cache is inflated by training-set leakage.

## 2. Trivial-feature baselines on val split

| Feature | Tok AUROC | Note |
|---|---:|---|
| topk_entropy_tok | 0.7055 | per-position |
| topk_peak_gap_tok | 0.4063 | per-position |
| h_norm_tok | 0.4260 | per-position |
| SE_tok | 0.9677 | per-position |
| linear probe on h_LLM (test) | 0.9882 | trained on first 240 chunks |
| linear probe on h_LLM (train) | 0.9972 | sanity — should be ≥ test |

|  Seq feature  | AUROC |
|---|---:|
| topk_entropy_seq | 0.9711 |
| topk_peak_gap_seq | 0.1306 |
| h_norm_seq | 0.3606 |

**Verdict:** the trained EqM auditor's contribution is the gap between its AUROC and the *best* trivial baseline. If the gap is small the headline number is mostly recovering pre-existing LM signal.

## 3. Uncorrupted-position Tok AUROC (val split)

- EqM `U_pos` at **un**corrupted positions: 0.9229
- SE     at **un**corrupted positions: 0.6562
- n positions: 2890

Both should be ≈ 0.5 if the discriminator localises corruption. Significant deviation means the *invalid context* (cascade from the AR LM) bleeds into nearby hidden states and the discriminator picks that up indirectly.

## 4. Cross-seed corruption (val split)

|   stat   | original-seed | new-seed | Δ |
|---|---:|---:|---:|
| Seq_EqM_Egrad²         | 0.9417 | 0.9514 | +0.0097 |
| Seq_SE                 | 0.9986 | 1.0000 | +0.0014 |
| Tok_EqM_Upos           | 0.9481 | 0.9439 | -0.0042 |
| Tok_SE                 | 0.9677 | 0.9638 | -0.0039 |

Mask overlap with original seed: both=243, orig only=707, new only=719. AUROC should hold up across seeds; a sharp drop means the auditor memorised the specific corruption draw.

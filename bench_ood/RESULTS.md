# OOD-detection benchmark — consolidated results

- **git**: `ebe9fb38c4c4`  **gpu**: NVIDIA RTX PRO 1000 Blackwell Generation Laptop GPU  **created**: 2026-07-13T20:26:46+00:00
- **ckpt**: `runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM/epoch_best.pt` (md5 `f1d809e8a42d`)
- **config**: split=test n=256 fit_seqs=512 seed=42 rates=0.05,0.1,0.15,0.2,0.25,0.3,0.5,0.7,1.0
- **per-detector t**: {"NLL": {"t_nll": 3.0, "t_var": 7.5}, "BLR": {"t_eval": 4.5, "var_t_eval": 7.5}, "BGMM": {"t_eval": 7.5}, "GPT2_SE": {"score": "spilled", "char_attrib": "boundary"}, "GPT2_NLL": {"score": "nll", "char_attrib": "boundary"}}

## Claim 1 — localization
> **Evaluation protocol.** Each model is scored in the unit it ACTUALLY operates
> in, and the cross-model comparison happens at a common unit:
> * flow-matching detectors (NLL / BLR / BGMM) → native unit = **character**
> * GPT-2 baselines (spilled energy, NLL) → native unit = **BPE token**
> * both are pooled up to **words** → the fair head-to-head
>
> No score is ever attributed *downward* (BPE→char): GPT-2 has no per-character
> opinion, and the old "spread a token's score uniformly over its chars" rule
> smeared localization. Pooling *up* is exact, so words are the comparison unit —
> and words are what the corruptions actually use (`falseinfo`/`plausible` swap
> whole words). GPT-2's pre-tokenizer splits on whitespace, so a BPE token never
> crosses a word boundary: the word mapping is exact on both sides.

> **Two GPT-2 baselines, not interchangeable.** `GPT2_SE` is the *real* cross-step
> spilled energy (Minut et al., ICLR 2026): it pairs the logit energy at step i-1
> with the marginal energy at step i, so it straddles two decoding steps **by
> construction** — a sequence-level signal, not a localizer. `GPT2_NLL` is the same
> LM scored with the same-step per-token NLL: that is the fair localization
> comparator, and is what this repo previously reported *mislabelled* as spilled
> energy. Beating SE at localization would be a straw man — compare to **GPT2_NLL**.

### 1a. Native units (each model in its own tokenization)

_**Not cross-comparable** — a char AUROC and a BPE AUROC count different things. These say how well each model localizes in the units it actually has. Use the WORD table (1c) for the head-to-head._

**replace — flow-matching detectors, per-CHARACTER AUROC**

| rate | NLL | BLR | BLR_ADV | BLR_FI | BGMM |
|---|---|---|---|---|---|
| 0.05 | 0.981 | 0.879 | 0.896 | 0.767 | 0.886 |
| 0.1 | 0.978 | 0.880 | 0.897 | 0.761 | 0.863 |
| 0.15 | 0.968 | 0.874 | 0.888 | 0.756 | 0.830 |
| 0.2 | 0.958 | 0.874 | 0.881 | 0.750 | 0.800 |
| 0.25 | 0.947 | 0.866 | 0.874 | 0.739 | 0.771 |
| 0.3 | 0.931 | 0.861 | 0.865 | 0.731 | 0.741 |
| 0.5 | 0.854 | 0.821 | 0.819 | 0.683 | 0.653 |
| 0.7 | 0.780 | 0.770 | 0.773 | 0.664 | 0.597 |
| 1 | — | — | — | — | — |

**replace — GPT-2 baselines, per-BPE-TOKEN AUROC**

| rate | GPT2_SE | GPT2_NLL |
|---|---|---|
| 0.05 | 0.546 | 0.696 |
| 0.1 | 0.513 | 0.604 |
| 0.15 | 0.480 | 0.557 |
| 0.2 | 0.450 | 0.526 |
| 0.25 | 0.443 | 0.514 |
| 0.3 | 0.427 | 0.502 |
| 0.5 | 0.442 | 0.514 |
| 0.7 | 0.481 | 0.549 |
| 1 | — | — |

**shuffle — flow-matching detectors, per-CHARACTER AUROC**

| rate | NLL | BLR | BLR_ADV | BLR_FI | BGMM |
|---|---|---|---|---|---|
| 0.05 | 0.971 | 0.795 | 0.818 | 0.704 | 0.891 |
| 0.1 | 0.961 | 0.787 | 0.809 | 0.689 | 0.865 |
| 0.15 | 0.948 | 0.783 | 0.796 | 0.675 | 0.833 |
| 0.2 | 0.935 | 0.774 | 0.784 | 0.668 | 0.801 |
| 0.25 | 0.914 | 0.758 | 0.767 | 0.653 | 0.774 |
| 0.3 | 0.888 | 0.744 | 0.747 | 0.634 | 0.736 |
| 0.5 | 0.766 | 0.680 | 0.676 | 0.589 | 0.634 |
| 0.7 | 0.634 | 0.605 | 0.608 | 0.555 | 0.573 |
| 1 | 0.607 | 0.598 | 0.618 | 0.612 | 0.531 |

**shuffle — GPT-2 baselines, per-BPE-TOKEN AUROC**

| rate | GPT2_SE | GPT2_NLL |
|---|---|---|
| 0.05 | 0.554 | 0.699 |
| 0.1 | 0.522 | 0.628 |
| 0.15 | 0.493 | 0.581 |
| 0.2 | 0.484 | 0.560 |
| 0.25 | 0.472 | 0.549 |
| 0.3 | 0.472 | 0.548 |
| 0.5 | 0.499 | 0.578 |
| 0.7 | 0.529 | 0.641 |
| 1 | 0.538 | 0.753 |

**falseinfo — flow-matching detectors, per-CHARACTER AUROC**

| rate | NLL | BLR | BLR_ADV | BLR_FI | BGMM |
|---|---|---|---|---|---|
| 0.05 | 0.724 | 0.523 | 0.678 | 0.729 | 0.680 |
| 0.1 | 0.722 | 0.511 | 0.684 | 0.737 | 0.676 |
| 0.15 | 0.731 | 0.522 | 0.689 | 0.745 | 0.683 |
| 0.2 | 0.720 | 0.512 | 0.685 | 0.735 | 0.678 |
| 0.25 | 0.703 | 0.515 | 0.691 | 0.734 | 0.675 |
| 0.3 | 0.708 | 0.514 | 0.694 | 0.737 | 0.679 |
| 0.5 | 0.688 | 0.510 | 0.716 | 0.753 | 0.674 |
| 0.7 | 0.662 | 0.492 | 0.743 | 0.770 | 0.671 |
| 1 | 0.588 | 0.452 | 0.847 | 0.854 | 0.681 |

**falseinfo — GPT-2 baselines, per-BPE-TOKEN AUROC**

| rate | GPT2_SE | GPT2_NLL |
|---|---|---|
| 0.05 | 0.624 | 0.798 |
| 0.1 | 0.621 | 0.781 |
| 0.15 | 0.621 | 0.768 |
| 0.2 | 0.613 | 0.744 |
| 0.25 | 0.600 | 0.736 |
| 0.3 | 0.598 | 0.715 |
| 0.5 | 0.581 | 0.667 |
| 0.7 | 0.567 | 0.617 |
| 1 | 0.552 | 0.674 |

### 1b. Sequence level (comparable: per-character normalisation)

_Cross-tokenizer sequence scores are made comparable the standard way — total score divided by the number of CHARACTERS (the bits-per-character convention). A mean over BPE tokens would NOT be comparable: corrupted text tokenises into more BPE tokens, which dilutes it._

**replace — sequence AUROC**

| rate | NLL | BLR | BLR_ADV | BLR_FI | BGMM | GPT2_SE | GPT2_NLL |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.989 | 0.845 | 0.766 | 0.751 | 0.853 | 0.993 | 0.994 |
| 0.1 | 1.000 | 0.975 | 0.931 | 0.915 | 0.965 | 1.000 | 1.000 |
| 0.15 | 1.000 | 0.996 | 0.973 | 0.967 | 0.991 | 1.000 | 1.000 |
| 0.2 | 1.000 | 1.000 | 0.993 | 0.994 | 1.000 | 1.000 | 1.000 |
| 0.25 | 1.000 | 1.000 | 0.997 | 0.999 | 1.000 | 1.000 | 1.000 |
| 0.3 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.7 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

**falseinfo — sequence AUROC**

| rate | NLL | BLR | BLR_ADV | BLR_FI | BGMM | GPT2_SE | GPT2_NLL |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.589 | 0.550 | 0.562 | 0.582 | 0.562 | 0.657 | 0.677 |
| 0.1 | 0.664 | 0.584 | 0.614 | 0.652 | 0.611 | 0.771 | 0.797 |
| 0.15 | 0.743 | 0.635 | 0.683 | 0.733 | 0.685 | 0.873 | 0.893 |
| 0.2 | 0.786 | 0.680 | 0.722 | 0.789 | 0.727 | 0.910 | 0.928 |
| 0.25 | 0.850 | 0.727 | 0.780 | 0.840 | 0.785 | 0.950 | 0.957 |
| 0.3 | 0.880 | 0.764 | 0.820 | 0.882 | 0.826 | 0.966 | 0.972 |
| 0.5 | 0.953 | 0.905 | 0.937 | 0.968 | 0.949 | 0.995 | 0.996 |
| 0.7 | 0.978 | 0.947 | 0.977 | 0.993 | 0.982 | 0.999 | 0.999 |
| 1 | 0.994 | 0.991 | 0.993 | 1.000 | 0.997 | 1.000 | 1.000 |

### 1c. WORD level — the fair head-to-head (both FM and GPT-2)

_`max`-pool asks "is ANY part of this word surprising?" (suits a single replaced character); `mean`-pool asks "is it surprising on average?" (suits weak signal spread over the whole word, which is the false-info regime). Both are reported rather than picking the flattering one._

**replace — WORD AUROC (max-pool)**

| rate | NLL | BLR | BLR_ADV | BLR_FI | BGMM | GPT2_SE | GPT2_NLL |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.977 | 0.905 | 0.887 | 0.779 | 0.874 | 0.793 | 0.836 |
| 0.1 | 0.979 | 0.923 | 0.905 | 0.797 | 0.884 | 0.757 | 0.776 |
| 0.15 | 0.981 | 0.935 | 0.929 | 0.818 | 0.888 | 0.727 | 0.738 |
| 0.2 | 0.981 | 0.946 | 0.941 | 0.831 | 0.896 | 0.667 | 0.716 |
| 0.25 | 0.982 | 0.944 | 0.942 | 0.839 | 0.894 | 0.645 | 0.709 |
| 0.3 | 0.983 | 0.955 | 0.952 | 0.837 | 0.897 | 0.609 | 0.701 |
| 0.5 | 0.979 | 0.970 | 0.971 | 0.823 | 0.865 | 0.593 | 0.732 |
| 0.7 | 0.961 | 0.965 | 0.963 | 0.786 | 0.858 | 0.695 | 0.756 |
| 1 | — | — | — | — | — | — | — |

**replace — WORD AUROC (mean-pool)**

| rate | NLL | BLR | BLR_ADV | BLR_FI | BGMM | GPT2_SE | GPT2_NLL |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.957 | 0.833 | 0.788 | 0.735 | 0.871 | 0.633 | 0.733 |
| 0.1 | 0.956 | 0.855 | 0.810 | 0.744 | 0.883 | 0.576 | 0.657 |
| 0.15 | 0.959 | 0.869 | 0.842 | 0.758 | 0.891 | 0.524 | 0.596 |
| 0.2 | 0.960 | 0.882 | 0.862 | 0.762 | 0.896 | 0.437 | 0.558 |
| 0.25 | 0.956 | 0.880 | 0.863 | 0.763 | 0.896 | 0.400 | 0.534 |
| 0.3 | 0.955 | 0.884 | 0.873 | 0.759 | 0.889 | 0.357 | 0.498 |
| 0.5 | 0.916 | 0.884 | 0.887 | 0.678 | 0.812 | 0.303 | 0.429 |
| 0.7 | 0.872 | 0.840 | 0.851 | 0.560 | 0.708 | 0.391 | 0.397 |
| 1 | — | — | — | — | — | — | — |

**shuffle — WORD AUROC (max-pool)**

| rate | NLL | BLR | BLR_ADV | BLR_FI | BGMM | GPT2_SE | GPT2_NLL |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.963 | 0.854 | 0.831 | 0.761 | 0.882 | 0.785 | 0.828 |
| 0.1 | 0.963 | 0.864 | 0.845 | 0.765 | 0.887 | 0.762 | 0.775 |
| 0.15 | 0.963 | 0.872 | 0.857 | 0.762 | 0.886 | 0.724 | 0.746 |
| 0.2 | 0.962 | 0.872 | 0.861 | 0.753 | 0.878 | 0.703 | 0.722 |
| 0.25 | 0.960 | 0.882 | 0.870 | 0.754 | 0.885 | 0.685 | 0.724 |
| 0.3 | 0.950 | 0.876 | 0.876 | 0.729 | 0.876 | 0.665 | 0.719 |
| 0.5 | 0.908 | 0.852 | 0.865 | 0.671 | 0.869 | 0.678 | 0.753 |
| 0.7 | 0.855 | 0.837 | 0.856 | 0.653 | 0.859 | 0.731 | 0.805 |
| 1 | 0.842 | 0.871 | 0.873 | 0.688 | 0.865 | 0.738 | 0.801 |

**shuffle — WORD AUROC (mean-pool)**

| rate | NLL | BLR | BLR_ADV | BLR_FI | BGMM | GPT2_SE | GPT2_NLL |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.936 | 0.785 | 0.739 | 0.719 | 0.876 | 0.636 | 0.732 |
| 0.1 | 0.936 | 0.789 | 0.750 | 0.711 | 0.873 | 0.597 | 0.665 |
| 0.15 | 0.928 | 0.788 | 0.750 | 0.694 | 0.873 | 0.542 | 0.623 |
| 0.2 | 0.925 | 0.781 | 0.755 | 0.678 | 0.869 | 0.513 | 0.598 |
| 0.25 | 0.917 | 0.776 | 0.754 | 0.666 | 0.865 | 0.484 | 0.591 |
| 0.3 | 0.895 | 0.753 | 0.750 | 0.626 | 0.855 | 0.453 | 0.588 |
| 0.5 | 0.803 | 0.691 | 0.726 | 0.526 | 0.822 | 0.471 | 0.621 |
| 0.7 | 0.678 | 0.647 | 0.700 | 0.474 | 0.783 | 0.523 | 0.691 |
| 1 | 0.658 | 0.698 | 0.723 | 0.496 | 0.796 | 0.534 | 0.714 |

**falseinfo — WORD AUROC (max-pool)**

| rate | NLL | BLR | BLR_ADV | BLR_FI | BGMM | GPT2_SE | GPT2_NLL |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.807 | 0.736 | 0.768 | 0.802 | 0.706 | 0.691 | 0.885 |
| 0.1 | 0.793 | 0.728 | 0.766 | 0.790 | 0.699 | 0.689 | 0.871 |
| 0.15 | 0.813 | 0.744 | 0.779 | 0.800 | 0.707 | 0.690 | 0.853 |
| 0.2 | 0.796 | 0.732 | 0.766 | 0.790 | 0.699 | 0.677 | 0.834 |
| 0.25 | 0.788 | 0.742 | 0.768 | 0.789 | 0.703 | 0.664 | 0.823 |
| 0.3 | 0.797 | 0.740 | 0.769 | 0.786 | 0.702 | 0.666 | 0.803 |
| 0.5 | 0.785 | 0.744 | 0.773 | 0.794 | 0.694 | 0.640 | 0.748 |
| 0.7 | 0.771 | 0.745 | 0.772 | 0.790 | 0.695 | 0.628 | 0.698 |
| 1 | 0.678 | 0.770 | 0.814 | 0.846 | 0.796 | 0.645 | 0.765 |

**falseinfo — WORD AUROC (mean-pool)**

| rate | NLL | BLR | BLR_ADV | BLR_FI | BGMM | GPT2_SE | GPT2_NLL |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.805 | 0.709 | 0.748 | 0.795 | 0.766 | 0.678 | 0.870 |
| 0.1 | 0.790 | 0.681 | 0.741 | 0.787 | 0.756 | 0.672 | 0.853 |
| 0.15 | 0.808 | 0.711 | 0.747 | 0.796 | 0.768 | 0.672 | 0.838 |
| 0.2 | 0.792 | 0.696 | 0.739 | 0.785 | 0.759 | 0.660 | 0.816 |
| 0.25 | 0.781 | 0.704 | 0.741 | 0.781 | 0.753 | 0.645 | 0.806 |
| 0.3 | 0.789 | 0.704 | 0.745 | 0.780 | 0.762 | 0.642 | 0.784 |
| 0.5 | 0.774 | 0.714 | 0.748 | 0.785 | 0.752 | 0.620 | 0.729 |
| 0.7 | 0.752 | 0.702 | 0.737 | 0.770 | 0.740 | 0.602 | 0.678 |
| 1 | 0.581 | 0.623 | 0.682 | 0.765 | 0.746 | 0.627 | 0.751 |

**GPT-2 arms at native BPE granularity (no char attribution)**

_The per-char numbers above need a BPE->char rule (`boundary`). These do not:_
_they score GPT-2's own tokens, so they show whether a weak per-char result is_
_an attribution artifact or real._

| replace rate | GPT2_SE seq | GPT2_SE token | GPT2_NLL seq | GPT2_NLL token |
|---|---|---|---|---|
| 0.05 | 0.979 | 0.546 | 0.977 | 0.696 |
| 0.1 | 0.997 | 0.513 | 0.996 | 0.604 |
| 0.15 | 1.000 | 0.480 | 1.000 | 0.557 |
| 0.2 | 1.000 | 0.450 | 1.000 | 0.526 |
| 0.25 | 1.000 | 0.443 | 1.000 | 0.514 |
| 0.3 | 1.000 | 0.427 | 1.000 | 0.502 |
| 0.5 | 0.998 | 0.442 | 0.994 | 0.514 |
| 0.7 | 0.988 | 0.481 | 0.967 | 0.549 |
| 1 | 0.942 | — | 0.873 | — |

| shuffle rate | GPT2_SE seq | GPT2_SE token | GPT2_NLL seq | GPT2_NLL token |
|---|---|---|---|---|
| 0.05 | 0.965 | 0.554 | 0.963 | 0.699 |
| 0.1 | 0.997 | 0.522 | 0.997 | 0.628 |
| 0.15 | 1.000 | 0.493 | 1.000 | 0.581 |
| 0.2 | 1.000 | 0.484 | 1.000 | 0.560 |
| 0.25 | 1.000 | 0.472 | 1.000 | 0.549 |
| 0.3 | 1.000 | 0.472 | 1.000 | 0.548 |
| 0.5 | 1.000 | 0.499 | 0.999 | 0.578 |
| 0.7 | 0.999 | 0.529 | 0.996 | 0.641 |
| 1 | 0.998 | 0.538 | 0.993 | 0.753 |

| falseinfo rate | GPT2_SE seq | GPT2_SE token | GPT2_NLL seq | GPT2_NLL token |
|---|---|---|---|---|
| 0.05 | 0.671 | 0.624 | 0.707 | 0.798 |
| 0.1 | 0.790 | 0.621 | 0.831 | 0.781 |
| 0.15 | 0.906 | 0.621 | 0.936 | 0.768 |
| 0.2 | 0.945 | 0.613 | 0.964 | 0.744 |
| 0.25 | 0.978 | 0.600 | 0.987 | 0.736 |
| 0.3 | 0.987 | 0.598 | 0.993 | 0.715 |
| 0.5 | 1.000 | 0.581 | 1.000 | 0.667 |
| 0.7 | 1.000 | 0.567 | 1.000 | 0.617 |
| 1 | 1.000 | 0.552 | 1.000 | 0.674 |

## Claim 2 — false info: sequence triage vs per-token
_seq AUROC rises with rate (human-in-the-loop); per-token stays flat; see figs/claim2_falseinfo.png_

**falseinfo — sequence AUROC**

| rate | NLL | BLR | BLR_ADV | BLR_FI | BGMM | GPT2_SE | GPT2_NLL |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.589 | 0.550 | 0.562 | 0.582 | 0.562 | 0.657 | 0.677 |
| 0.1 | 0.664 | 0.584 | 0.614 | 0.652 | 0.611 | 0.771 | 0.797 |
| 0.15 | 0.743 | 0.635 | 0.683 | 0.733 | 0.685 | 0.873 | 0.893 |
| 0.2 | 0.786 | 0.680 | 0.722 | 0.789 | 0.727 | 0.910 | 0.928 |
| 0.25 | 0.850 | 0.727 | 0.780 | 0.840 | 0.785 | 0.950 | 0.957 |
| 0.3 | 0.880 | 0.764 | 0.820 | 0.882 | 0.826 | 0.966 | 0.972 |
| 0.5 | 0.953 | 0.905 | 0.937 | 0.968 | 0.949 | 0.995 | 0.996 |
| 0.7 | 0.978 | 0.947 | 0.977 | 0.993 | 0.982 | 0.999 | 0.999 |
| 1 | 0.994 | 0.991 | 0.993 | 1.000 | 0.997 | 1.000 | 1.000 |

**falseinfo — per-token AUROC**

| rate | NLL | BLR | BLR_ADV | BLR_FI | BGMM | GPT2_SE | GPT2_NLL |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.724 | 0.523 | 0.678 | 0.729 | 0.680 | 0.624 | 0.798 |
| 0.1 | 0.722 | 0.511 | 0.684 | 0.737 | 0.676 | 0.621 | 0.781 |
| 0.15 | 0.731 | 0.522 | 0.689 | 0.745 | 0.683 | 0.621 | 0.768 |
| 0.2 | 0.720 | 0.512 | 0.685 | 0.735 | 0.678 | 0.613 | 0.744 |
| 0.25 | 0.703 | 0.515 | 0.691 | 0.734 | 0.675 | 0.600 | 0.736 |
| 0.3 | 0.708 | 0.514 | 0.694 | 0.737 | 0.679 | 0.598 | 0.715 |
| 0.5 | 0.688 | 0.510 | 0.716 | 0.753 | 0.674 | 0.581 | 0.667 |
| 0.7 | 0.662 | 0.492 | 0.743 | 0.770 | 0.671 | 0.567 | 0.617 |
| 1 | 0.588 | 0.452 | 0.847 | 0.854 | 0.681 | 0.552 | 0.674 |

## Claim 3 — plausible (model-fluent) substitutions
_char-model detectors (NLL/BLR/BGMM) → chance on plausible; see figs/claim3_plausible.png_

> **`random` here IS the false-info corruption** (`corrupt_false_info` on the
> SAME word slots as the plausible swap — only the planted word differs: a
> random real word vs the model's own most-fluent candidate). So the two
> columns below are a **cross-transfer matrix** for the supervised heads:
> `Logistic_fi` is trained on false-info, `Logistic_plaus` on plausible. Read
> the diagonal (each head on its own corruption) against the off-diagonal.

- swapped-char denoiser NLL: random=0.564 plausible=0.006 (clean=0.130) — the plausible swap is scored as **less surprising than the truth**, which is why a likelihood readout must invert.

AUROC (ranking) and per-token F1 at the clean-calibrated threshold (deployment):

| detector | false-info seq | plausible seq | false-info tok | plausible tok | false-info tok F1 | plausible tok F1 |
|---|---|---|---|---|---|---|
| BGMM | 0.682 | 0.426 | 0.693 | 0.458 | 0.149 | 0.017 |
| BLR | 0.651 | 0.455 | 0.602 | 0.492 | 0.196 | 0.011 |
| GPT2_NLL | 0.896 | 0.728 | 0.559 | 0.507 | 0.226 | 0.116 |
| GPT2_SE | 0.860 | 0.700 | 0.558 | 0.514 | 0.150 | 0.068 |
| NLL | 0.725 | 0.448 | 0.741 | 0.264 | 0.263 | 0.003 |
| ALL_linear | 0.706 | 0.611 | 0.740 | 0.597 | 0.254 | 0.031 |
| ALL_mlp | 0.654 | 0.686 | 0.706 | 0.710 | 0.241 | 0.065 |
| ALL_quad | 0.740 | 0.704 | 0.727 | 0.683 | 0.267 | 0.229 |
| Logistic_fi | 0.737 | 0.342 | 0.783 | 0.473 | 0.301 | 0.007 |
| Logistic_plaus | 0.384 | 0.808 | 0.471 | 0.904 | 0.022 | 0.537 |

## Operating point — precision / recall / F1
_AUROC is threshold-free (pure ranking). These are what a **deployed** detector gives once it must commit to a threshold, calibrated GT-free at 5% FPR on clean data — the same convention the healing bench uses for `loc_precision`/`loc_recall`, so the two benches are directly comparable. `F1_best` is the oracle-threshold ceiling. See figs/prf_f1.png._

**replace @ rate=0.15**

| detector | tok P | tok R | tok F1 | tok F1_best | seq P | seq R | seq F1 | seq F1_best |
|---|---|---|---|---|---|---|---|---|
| NLL | 0.640 | 0.897 | 0.747 | 0.797 | 0.952 | 1.000 | 0.975 | 1.000 |
| BLR | 0.516 | 0.750 | 0.611 | 0.648 | 0.951 | 0.988 | 0.969 | 0.971 |
| BLR_ADV | 0.523 | 0.738 | 0.612 | 0.635 | 0.947 | 0.902 | 0.924 | 0.938 |
| BLR_FI | 0.293 | 0.648 | 0.403 | 0.407 | 0.944 | 0.863 | 0.902 | 0.912 |
| BGMM | 0.485 | 0.477 | 0.481 | 0.493 | 0.951 | 0.984 | 0.967 | 0.975 |
| GPT2_SE | 0.214 | 0.054 | 0.086 | 0.357 | 0.952 | 1.000 | 0.975 | 1.000 |
| GPT2_NLL | 0.212 | 0.194 | 0.203 | 0.385 | 0.952 | 1.000 | 0.975 | 1.000 |

_token prevalence (corrupt fraction) = 0.149 — a random detector's precision. Threshold calibrated at 5% FPR on clean tokens._

**shuffle @ rate=0.15**

| detector | tok P | tok R | tok F1 | tok F1_best | seq P | seq R | seq F1 | seq F1_best |
|---|---|---|---|---|---|---|---|---|
| NLL | 0.595 | 0.819 | 0.689 | 0.719 | 0.952 | 1.000 | 0.975 | 1.000 |
| BLR | 0.387 | 0.554 | 0.455 | 0.463 | 0.950 | 0.961 | 0.955 | 0.958 |
| BLR_ADV | 0.415 | 0.522 | 0.462 | 0.464 | 0.934 | 0.719 | 0.812 | 0.910 |
| BLR_FI | 0.232 | 0.515 | 0.320 | 0.321 | 0.942 | 0.824 | 0.879 | 0.926 |
| BGMM | 0.458 | 0.540 | 0.496 | 0.497 | 0.951 | 0.984 | 0.967 | 0.979 |
| GPT2_SE | 0.187 | 0.056 | 0.086 | 0.290 | 0.952 | 1.000 | 0.975 | 1.000 |
| GPT2_NLL | 0.178 | 0.172 | 0.175 | 0.311 | 0.952 | 1.000 | 0.975 | 1.000 |

_token prevalence (corrupt fraction) = 0.135 — a random detector's precision. Threshold calibrated at 5% FPR on clean tokens._

**falseinfo @ rate=0.15**

| detector | tok P | tok R | tok F1 | tok F1_best | seq P | seq R | seq F1 | seq F1_best |
|---|---|---|---|---|---|---|---|---|
| NLL | 0.350 | 0.223 | 0.272 | 0.329 | 0.649 | 0.094 | 0.164 | 0.731 |
| BLR | 0.216 | 0.123 | 0.156 | 0.198 | 0.649 | 0.094 | 0.164 | 0.687 |
| BLR_ADV | 0.316 | 0.205 | 0.249 | 0.287 | 0.618 | 0.082 | 0.145 | 0.697 |
| BLR_FI | 0.332 | 0.290 | 0.310 | 0.327 | 0.711 | 0.125 | 0.213 | 0.729 |
| BGMM | 0.195 | 0.101 | 0.134 | 0.270 | 0.567 | 0.066 | 0.119 | 0.706 |
| GPT2_SE | 0.224 | 0.104 | 0.142 | 0.199 | 0.866 | 0.328 | 0.476 | 0.842 |
| GPT2_NLL | 0.285 | 0.190 | 0.228 | 0.236 | 0.867 | 0.332 | 0.480 | 0.858 |

_token prevalence (corrupt fraction) = 0.110 — a random detector's precision. Threshold calibrated at 5% FPR on clean tokens._

## Figures
- `figs/claim1_word_max.png`
- `figs/claim1_word_mean.png`
- `figs/claim2_falseinfo.png`
- `figs/claim3_plausible.png`
- `figs/prf_f1.png`

# OOD-detection benchmark — consolidated results

- **git**: `12c9cebfe8ed`  **gpu**: NVIDIA RTX PRO 1000 Blackwell Generation Laptop GPU  **created**: 2026-08-10T19:14:11+00:00
- **ckpt**: `runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_converge/epoch_final.pt` (md5 `9946dce94cb2`)
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
| 0.05 | 0.983 | 0.893 | 0.911 | 0.780 | 0.918 |
| 0.1 | 0.978 | 0.893 | 0.909 | 0.772 | 0.892 |
| 0.15 | 0.967 | 0.889 | 0.897 | 0.762 | 0.858 |
| 0.2 | 0.952 | 0.887 | 0.887 | 0.753 | 0.823 |
| 0.25 | 0.934 | 0.880 | 0.877 | 0.739 | 0.787 |
| 0.3 | 0.910 | 0.872 | 0.867 | 0.725 | 0.755 |
| 0.5 | 0.810 | 0.822 | 0.818 | 0.655 | 0.635 |
| 0.7 | 0.756 | 0.771 | 0.772 | 0.627 | 0.557 |
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
| 0.05 | 0.974 | 0.826 | 0.856 | 0.718 | 0.919 |
| 0.1 | 0.965 | 0.814 | 0.836 | 0.705 | 0.886 |
| 0.15 | 0.950 | 0.806 | 0.815 | 0.689 | 0.852 |
| 0.2 | 0.934 | 0.797 | 0.801 | 0.678 | 0.813 |
| 0.25 | 0.911 | 0.780 | 0.780 | 0.664 | 0.775 |
| 0.3 | 0.879 | 0.761 | 0.756 | 0.640 | 0.736 |
| 0.5 | 0.724 | 0.682 | 0.678 | 0.583 | 0.629 |
| 0.7 | 0.604 | 0.605 | 0.608 | 0.550 | 0.566 |
| 1 | 0.634 | 0.596 | 0.596 | 0.600 | 0.531 |

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
| 0.05 | 0.713 | 0.526 | 0.663 | 0.711 | 0.707 |
| 0.1 | 0.713 | 0.527 | 0.670 | 0.720 | 0.704 |
| 0.15 | 0.725 | 0.529 | 0.676 | 0.731 | 0.711 |
| 0.2 | 0.703 | 0.527 | 0.675 | 0.723 | 0.707 |
| 0.25 | 0.690 | 0.535 | 0.678 | 0.722 | 0.700 |
| 0.3 | 0.694 | 0.531 | 0.680 | 0.724 | 0.702 |
| 0.5 | 0.669 | 0.540 | 0.697 | 0.743 | 0.695 |
| 0.7 | 0.643 | 0.540 | 0.720 | 0.761 | 0.684 |
| 1 | 0.583 | 0.541 | 0.801 | 0.835 | 0.675 |

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
| 0.05 | 0.985 | 0.845 | 0.766 | 0.762 | 0.881 | 0.993 | 0.994 |
| 0.1 | 0.997 | 0.972 | 0.934 | 0.921 | 0.975 | 1.000 | 1.000 |
| 0.15 | 1.000 | 0.993 | 0.977 | 0.973 | 0.993 | 1.000 | 1.000 |
| 0.2 | 1.000 | 1.000 | 0.996 | 0.997 | 1.000 | 1.000 | 1.000 |
| 0.25 | 1.000 | 1.000 | 0.999 | 0.999 | 1.000 | 1.000 | 1.000 |
| 0.3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.7 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

**falseinfo — sequence AUROC**

| rate | NLL | BLR | BLR_ADV | BLR_FI | BGMM | GPT2_SE | GPT2_NLL |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.589 | 0.550 | 0.571 | 0.589 | 0.567 | 0.657 | 0.677 |
| 0.1 | 0.660 | 0.581 | 0.622 | 0.666 | 0.625 | 0.771 | 0.797 |
| 0.15 | 0.727 | 0.618 | 0.697 | 0.749 | 0.699 | 0.873 | 0.893 |
| 0.2 | 0.777 | 0.675 | 0.742 | 0.805 | 0.750 | 0.910 | 0.928 |
| 0.25 | 0.843 | 0.723 | 0.797 | 0.858 | 0.808 | 0.950 | 0.957 |
| 0.3 | 0.874 | 0.746 | 0.844 | 0.898 | 0.848 | 0.966 | 0.972 |
| 0.5 | 0.951 | 0.885 | 0.952 | 0.977 | 0.951 | 0.995 | 0.996 |
| 0.7 | 0.972 | 0.932 | 0.986 | 0.996 | 0.979 | 0.999 | 0.999 |
| 1 | 0.984 | 0.982 | 0.996 | 1.000 | 0.993 | 1.000 | 1.000 |

### 1c. WORD level — the fair head-to-head (both FM and GPT-2)

_`max`-pool asks "is ANY part of this word surprising?" (suits a single replaced character); `mean`-pool asks "is it surprising on average?" (suits weak signal spread over the whole word, which is the false-info regime). Both are reported rather than picking the flattering one._

**replace — WORD AUROC (max-pool)**

| rate | NLL | BLR | BLR_ADV | BLR_FI | BGMM | GPT2_SE | GPT2_NLL |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.980 | 0.890 | 0.911 | 0.789 | 0.909 | 0.793 | 0.836 |
| 0.1 | 0.979 | 0.912 | 0.926 | 0.814 | 0.914 | 0.757 | 0.776 |
| 0.15 | 0.981 | 0.930 | 0.943 | 0.830 | 0.910 | 0.727 | 0.738 |
| 0.2 | 0.982 | 0.945 | 0.952 | 0.842 | 0.904 | 0.667 | 0.716 |
| 0.25 | 0.981 | 0.943 | 0.950 | 0.846 | 0.891 | 0.645 | 0.709 |
| 0.3 | 0.980 | 0.953 | 0.959 | 0.845 | 0.879 | 0.609 | 0.701 |
| 0.5 | 0.967 | 0.971 | 0.973 | 0.792 | 0.818 | 0.593 | 0.732 |
| 0.7 | 0.954 | 0.971 | 0.968 | 0.713 | 0.809 | 0.695 | 0.756 |
| 1 | — | — | — | — | — | — | — |

**replace — WORD AUROC (mean-pool)**

| rate | NLL | BLR | BLR_ADV | BLR_FI | BGMM | GPT2_SE | GPT2_NLL |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.962 | 0.796 | 0.833 | 0.746 | 0.885 | 0.633 | 0.733 |
| 0.1 | 0.956 | 0.831 | 0.855 | 0.761 | 0.899 | 0.576 | 0.657 |
| 0.15 | 0.957 | 0.857 | 0.882 | 0.774 | 0.899 | 0.524 | 0.596 |
| 0.2 | 0.952 | 0.875 | 0.893 | 0.778 | 0.889 | 0.437 | 0.558 |
| 0.25 | 0.942 | 0.877 | 0.890 | 0.777 | 0.875 | 0.400 | 0.534 |
| 0.3 | 0.930 | 0.878 | 0.897 | 0.767 | 0.859 | 0.357 | 0.498 |
| 0.5 | 0.871 | 0.886 | 0.902 | 0.643 | 0.735 | 0.303 | 0.429 |
| 0.7 | 0.872 | 0.875 | 0.875 | 0.474 | 0.661 | 0.391 | 0.397 |
| 1 | — | — | — | — | — | — | — |

**shuffle — WORD AUROC (max-pool)**

| rate | NLL | BLR | BLR_ADV | BLR_FI | BGMM | GPT2_SE | GPT2_NLL |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.971 | 0.836 | 0.866 | 0.775 | 0.908 | 0.785 | 0.828 |
| 0.1 | 0.968 | 0.858 | 0.873 | 0.782 | 0.903 | 0.762 | 0.775 |
| 0.15 | 0.967 | 0.869 | 0.877 | 0.781 | 0.897 | 0.724 | 0.746 |
| 0.2 | 0.965 | 0.868 | 0.877 | 0.770 | 0.876 | 0.703 | 0.722 |
| 0.25 | 0.962 | 0.882 | 0.884 | 0.773 | 0.874 | 0.685 | 0.724 |
| 0.3 | 0.950 | 0.873 | 0.886 | 0.746 | 0.860 | 0.665 | 0.719 |
| 0.5 | 0.881 | 0.846 | 0.865 | 0.650 | 0.828 | 0.678 | 0.753 |
| 0.7 | 0.796 | 0.825 | 0.854 | 0.596 | 0.810 | 0.731 | 0.805 |
| 1 | 0.806 | 0.866 | 0.875 | 0.620 | 0.783 | 0.738 | 0.801 |

**shuffle — WORD AUROC (mean-pool)**

| rate | NLL | BLR | BLR_ADV | BLR_FI | BGMM | GPT2_SE | GPT2_NLL |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.946 | 0.749 | 0.789 | 0.733 | 0.884 | 0.636 | 0.732 |
| 0.1 | 0.942 | 0.769 | 0.791 | 0.730 | 0.876 | 0.597 | 0.665 |
| 0.15 | 0.931 | 0.776 | 0.786 | 0.715 | 0.864 | 0.542 | 0.623 |
| 0.2 | 0.923 | 0.775 | 0.787 | 0.697 | 0.846 | 0.513 | 0.598 |
| 0.25 | 0.911 | 0.771 | 0.784 | 0.683 | 0.834 | 0.484 | 0.591 |
| 0.3 | 0.876 | 0.749 | 0.776 | 0.643 | 0.806 | 0.453 | 0.588 |
| 0.5 | 0.708 | 0.670 | 0.737 | 0.498 | 0.744 | 0.471 | 0.621 |
| 0.7 | 0.567 | 0.610 | 0.704 | 0.403 | 0.714 | 0.523 | 0.691 |
| 1 | 0.584 | 0.686 | 0.748 | 0.415 | 0.648 | 0.534 | 0.714 |

**falseinfo — WORD AUROC (max-pool)**

| rate | NLL | BLR | BLR_ADV | BLR_FI | BGMM | GPT2_SE | GPT2_NLL |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.796 | 0.713 | 0.773 | 0.807 | 0.745 | 0.691 | 0.885 |
| 0.1 | 0.783 | 0.701 | 0.773 | 0.806 | 0.740 | 0.689 | 0.871 |
| 0.15 | 0.807 | 0.726 | 0.793 | 0.812 | 0.743 | 0.690 | 0.853 |
| 0.2 | 0.786 | 0.723 | 0.777 | 0.804 | 0.738 | 0.677 | 0.834 |
| 0.25 | 0.778 | 0.725 | 0.777 | 0.799 | 0.734 | 0.664 | 0.823 |
| 0.3 | 0.786 | 0.726 | 0.777 | 0.801 | 0.729 | 0.666 | 0.803 |
| 0.5 | 0.771 | 0.733 | 0.777 | 0.803 | 0.727 | 0.640 | 0.748 |
| 0.7 | 0.751 | 0.733 | 0.775 | 0.797 | 0.708 | 0.628 | 0.698 |
| 1 | 0.616 | 0.762 | 0.813 | 0.830 | 0.729 | 0.645 | 0.765 |

**falseinfo — WORD AUROC (mean-pool)**

| rate | NLL | BLR | BLR_ADV | BLR_FI | BGMM | GPT2_SE | GPT2_NLL |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.791 | 0.654 | 0.750 | 0.802 | 0.792 | 0.678 | 0.870 |
| 0.1 | 0.777 | 0.640 | 0.749 | 0.797 | 0.784 | 0.672 | 0.853 |
| 0.15 | 0.800 | 0.658 | 0.761 | 0.803 | 0.792 | 0.672 | 0.838 |
| 0.2 | 0.779 | 0.656 | 0.755 | 0.795 | 0.786 | 0.660 | 0.816 |
| 0.25 | 0.767 | 0.665 | 0.752 | 0.789 | 0.775 | 0.645 | 0.806 |
| 0.3 | 0.775 | 0.659 | 0.752 | 0.787 | 0.776 | 0.642 | 0.784 |
| 0.5 | 0.755 | 0.675 | 0.753 | 0.789 | 0.768 | 0.620 | 0.729 |
| 0.7 | 0.726 | 0.670 | 0.745 | 0.773 | 0.738 | 0.602 | 0.678 |
| 1 | 0.507 | 0.603 | 0.706 | 0.748 | 0.643 | 0.627 | 0.751 |

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
| 0.05 | 0.589 | 0.550 | 0.571 | 0.589 | 0.567 | 0.657 | 0.677 |
| 0.1 | 0.660 | 0.581 | 0.622 | 0.666 | 0.625 | 0.771 | 0.797 |
| 0.15 | 0.727 | 0.618 | 0.697 | 0.749 | 0.699 | 0.873 | 0.893 |
| 0.2 | 0.777 | 0.675 | 0.742 | 0.805 | 0.750 | 0.910 | 0.928 |
| 0.25 | 0.843 | 0.723 | 0.797 | 0.858 | 0.808 | 0.950 | 0.957 |
| 0.3 | 0.874 | 0.746 | 0.844 | 0.898 | 0.848 | 0.966 | 0.972 |
| 0.5 | 0.951 | 0.885 | 0.952 | 0.977 | 0.951 | 0.995 | 0.996 |
| 0.7 | 0.972 | 0.932 | 0.986 | 0.996 | 0.979 | 0.999 | 0.999 |
| 1 | 0.984 | 0.982 | 0.996 | 1.000 | 0.993 | 1.000 | 1.000 |

**falseinfo — per-token AUROC**

| rate | NLL | BLR | BLR_ADV | BLR_FI | BGMM | GPT2_SE | GPT2_NLL |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.713 | 0.526 | 0.663 | 0.711 | 0.707 | 0.624 | 0.798 |
| 0.1 | 0.713 | 0.527 | 0.670 | 0.720 | 0.704 | 0.621 | 0.781 |
| 0.15 | 0.725 | 0.529 | 0.676 | 0.731 | 0.711 | 0.621 | 0.768 |
| 0.2 | 0.703 | 0.527 | 0.675 | 0.723 | 0.707 | 0.613 | 0.744 |
| 0.25 | 0.690 | 0.535 | 0.678 | 0.722 | 0.700 | 0.600 | 0.736 |
| 0.3 | 0.694 | 0.531 | 0.680 | 0.724 | 0.702 | 0.598 | 0.715 |
| 0.5 | 0.669 | 0.540 | 0.697 | 0.743 | 0.695 | 0.581 | 0.667 |
| 0.7 | 0.643 | 0.540 | 0.720 | 0.761 | 0.684 | 0.567 | 0.617 |
| 1 | 0.583 | 0.541 | 0.801 | 0.835 | 0.675 | 0.552 | 0.674 |

## Claim 3 — plausible (model-fluent) substitutions
_char-model detectors (NLL/BLR/BGMM) → chance on plausible; see figs/claim3_plausible.png_

> **`random` here IS the false-info corruption** (`corrupt_false_info` on the
> SAME word slots as the plausible swap — only the planted word differs: a
> random real word vs the model's own most-fluent candidate). So the two
> columns below are a **cross-transfer matrix** for the supervised heads:
> `Logistic_fi` is trained on false-info, `Logistic_plaus` on plausible. Read
> the diagonal (each head on its own corruption) against the off-diagonal.

- swapped-char denoiser NLL: random=0.535 plausible=0.003 (clean=0.124) — the plausible swap is scored as **less surprising than the truth**, which is why a likelihood readout must invert.

AUROC (ranking) and per-token F1 at the clean-calibrated threshold (deployment):

| detector | false-info seq | plausible seq | false-info tok | plausible tok | false-info tok F1 | plausible tok F1 |
|---|---|---|---|---|---|---|
| BGMM | 0.694 | 0.430 | 0.722 | 0.464 | 0.172 | 0.023 |
| BLR | 0.620 | 0.479 | 0.624 | 0.570 | 0.194 | 0.025 |
| GPT2_NLL | 0.896 | 0.724 | 0.559 | 0.509 | 0.226 | 0.119 |
| GPT2_SE | 0.860 | 0.698 | 0.558 | 0.527 | 0.150 | 0.082 |
| NLL | 0.724 | 0.478 | 0.728 | 0.264 | 0.263 | 0.000 |
| ALL_linear | 0.713 | 0.635 | 0.743 | 0.604 | 0.288 | 0.048 |
| ALL_mlp | 0.683 | 0.689 | 0.717 | 0.720 | 0.271 | 0.104 |
| ALL_quad | 0.769 | 0.679 | 0.735 | 0.662 | 0.288 | 0.225 |
| Logistic_fi | 0.758 | 0.409 | 0.795 | 0.480 | 0.329 | 0.006 |
| Logistic_plaus | 0.403 | 0.792 | 0.478 | 0.898 | 0.040 | 0.525 |

## Operating point — precision / recall / F1
_AUROC is threshold-free (pure ranking). These are what a **deployed** detector gives once it must commit to a threshold, calibrated GT-free at 5% FPR on clean data — the same convention the healing bench uses for `loc_precision`/`loc_recall`, so the two benches are directly comparable. `F1_best` is the oracle-threshold ceiling. See figs/prf_f1.png._

**replace @ rate=0.15**

| detector | tok P | tok R | tok F1 | tok F1_best | seq P | seq R | seq F1 | seq F1_best |
|---|---|---|---|---|---|---|---|---|
| NLL | 0.583 | 0.917 | 0.713 | 0.791 | 0.952 | 1.000 | 0.975 | 0.998 |
| BLR | 0.494 | 0.789 | 0.608 | 0.650 | 0.951 | 0.980 | 0.965 | 0.969 |
| BLR_ADV | 0.492 | 0.781 | 0.604 | 0.648 | 0.947 | 0.898 | 0.922 | 0.934 |
| BLR_FI | 0.283 | 0.700 | 0.403 | 0.414 | 0.944 | 0.852 | 0.895 | 0.925 |
| BGMM | 0.457 | 0.635 | 0.531 | 0.534 | 0.951 | 0.988 | 0.969 | 0.977 |
| GPT2_SE | 0.214 | 0.054 | 0.086 | 0.357 | 0.952 | 1.000 | 0.975 | 1.000 |
| GPT2_NLL | 0.212 | 0.194 | 0.203 | 0.385 | 0.952 | 1.000 | 0.975 | 1.000 |

_token prevalence (corrupt fraction) = 0.149 — a random detector's precision. Threshold calibrated at 5% FPR on clean tokens._

**shuffle @ rate=0.15**

| detector | tok P | tok R | tok F1 | tok F1_best | seq P | seq R | seq F1 | seq F1_best |
|---|---|---|---|---|---|---|---|---|
| NLL | 0.577 | 0.839 | 0.684 | 0.723 | 0.952 | 1.000 | 0.975 | 0.996 |
| BLR | 0.405 | 0.620 | 0.490 | 0.494 | 0.950 | 0.957 | 0.953 | 0.966 |
| BLR_ADV | 0.410 | 0.590 | 0.484 | 0.485 | 0.934 | 0.719 | 0.812 | 0.899 |
| BLR_FI | 0.238 | 0.576 | 0.336 | 0.337 | 0.941 | 0.809 | 0.870 | 0.932 |
| BGMM | 0.416 | 0.645 | 0.506 | 0.513 | 0.951 | 0.988 | 0.969 | 0.984 |
| GPT2_SE | 0.187 | 0.056 | 0.086 | 0.290 | 0.952 | 1.000 | 0.975 | 1.000 |
| GPT2_NLL | 0.178 | 0.172 | 0.175 | 0.311 | 0.952 | 1.000 | 0.975 | 1.000 |

_token prevalence (corrupt fraction) = 0.135 — a random detector's precision. Threshold calibrated at 5% FPR on clean tokens._

**falseinfo @ rate=0.15**

| detector | tok P | tok R | tok F1 | tok F1_best | seq P | seq R | seq F1 | seq F1_best |
|---|---|---|---|---|---|---|---|---|
| NLL | 0.355 | 0.226 | 0.276 | 0.339 | 0.723 | 0.133 | 0.224 | 0.724 |
| BLR | 0.260 | 0.151 | 0.191 | 0.210 | 0.594 | 0.074 | 0.132 | 0.676 |
| BLR_ADV | 0.331 | 0.226 | 0.269 | 0.286 | 0.683 | 0.109 | 0.189 | 0.717 |
| BLR_FI | 0.338 | 0.305 | 0.321 | 0.337 | 0.711 | 0.125 | 0.213 | 0.737 |
| BGMM | 0.227 | 0.128 | 0.164 | 0.285 | 0.536 | 0.059 | 0.106 | 0.713 |
| GPT2_SE | 0.224 | 0.104 | 0.142 | 0.199 | 0.866 | 0.328 | 0.476 | 0.842 |
| GPT2_NLL | 0.285 | 0.190 | 0.228 | 0.236 | 0.867 | 0.332 | 0.480 | 0.858 |

_token prevalence (corrupt fraction) = 0.110 — a random detector's precision. Threshold calibrated at 5% FPR on clean tokens._

## Figures
- `figs/claim1_word_max.png`
- `figs/claim1_word_mean.png`
- `figs/claim2_falseinfo.png`
- `figs/claim3_plausible.png`
- `figs/prf_f1.png`

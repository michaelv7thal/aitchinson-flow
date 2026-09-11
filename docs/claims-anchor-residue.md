# Claims-matrix anchor residue

Companion to `claims.tsv` and `docs/paper-map.md`, written 2026-09-11 when
every `paper_loc` was re-anchored from line numbers to `file:label` form.

`check_claims.py` is unaffected by any of this: it never reads `paper_loc`.
Every row below still passes against its artifact. What is listed here is a
**documentation** gap, the anchor no longer lands on the number it names.

After the sweep, 1404 of 1597 rows carry an anchor whose value is
present at the anchor. The remaining 193 fall into three groups.

## 1. Figure series, not text-verifiable (32 rows) - no action

These are correctly anchored. The value is plotted in the rendered figure and
never appears in the `.tex` source, so it cannot be confirmed by reading the
paper's text. `fig:ood-word-auroc` carries the word-max AUROC ladder at every
rate except 0.15, which `tab:ood-word` prints.

    D100, D101, D102, D103, D105, D106, D107, D108, D110, D111, D112, D497, D499, D500, D501, D78, D80, D81, D82, D83, D85, D86, D87, D88, D90, D91, D92, D93, D95, D96, D97, D98

## 2. Rows the paper deliberately does not print (46 rows) - no action

Their `notes` already say so ("not printed", "anchor row", "paper prints the
complement"). They back a sentence rather than a printed digit, and are now
anchored to the section that holds that sentence.

- `D117` 0.263 - unprinted (former tab:ood-plausible column) NLL token F1 / false info
- `D124` 0.025 - unprinted (former tab:ood-plausible column) BLR token F1 / plausible
- `D135` 0.15 - unprinted (former tab:ood-plausible column) GPT-2 SE token F1 / false info
- `D142` 0.119 - unprinted (former tab:ood-plausible column) GPT-2 NLL token F1 / plausible
- `D147` 0.329 - unprinted (former tab:ood-plausible column) LOG_fi token F1 / false info
- `D153` 0.04 - unprinted (former tab:ood-plausible column) LOG_pl token F1 / false info
- `D159` 0.288 - unprinted (former tab:ood-plausible column) Unified linear token F1 / false in
- `D161` 0.769 - unprinted (former tab:ood-plausible column) Unified quadratic sequence AUROC /
- `D162` 0.679 - unprinted (former tab:ood-plausible column) Unified quadratic sequence AUROC /
- `D164` 0.662 - unprinted (former tab:ood-plausible column) Unified quadratic token AUROC / pl
- `D165` 0.288 - unprinted (former tab:ood-plausible column) Unified quadratic token F1 / false
- `D168` 0.689 - unprinted (former tab:ood-plausible column) Unified MLP sequence AUROC / plaus
- `D169` 0.717 - unprinted (former tab:ood-plausible column) Unified MLP token AUROC / false in
- `D171` 0.271 - unprinted (former tab:ood-plausible column) Unified MLP token F1 / false info
- `D542` 0.112 - unprinted (former tab:ood-plausible column) Var token F1 false info
- `D543` 0.076 - unprinted (former tab:ood-plausible column) Var token F1 plausible
- `D566` -0.0932 - tab:ood-threshold LOG_fi character threshold at nominal 5% FPR
- `D567` 0.6826 - tab:ood-threshold LOG_fi word threshold at nominal 5% FPR
- `D568` 15980 - tab:ood-threshold LOG_fi realized FPR on replace, stored ingredient fp
- `D569` 4289 - tab:ood-threshold LOG_fi realized FPR on false information, stored ingredient 
- `L1` 1.422 - Langevin sweep, EqM noise-free control bigram KL
- `L2` 1.299 - Langevin sweep, EqM best noisy bigram KL (alpha=0.1)
- `L3` 1.478 - Langevin sweep, FM on clr noise-free control bigram KL
- `L4` 1.280 - Langevin sweep, FM on clr best noisy bigram KL (alpha=0.3)
- `L5` 1.601 - Langevin sweep, logit-space flow noise-free control bigram KL
- `L6` 1.301 - Langevin sweep, logit-space flow best noisy bigram KL (alpha=0.3)
- `L7` 2.211 - Langevin sweep, EqM bigram KL at alpha=1.0
- `L8` 0.204 - Langevin sweep, EqM unigram KL at alpha=1.0
- `M6` 2560000 - training characters at the matched budget
- `N68` 0.947 - accuracy after the descent at gamma=0.03 (paper prints the complement 5.3%%)
- `N69` -2.7057 - log mass on the true character at gamma=0.05, before the descent
- `N70` -0.0068 - log mass on the true character at gamma=0.05, after the descent
- `N71` -11.4532 - log mass on the true character at the source, after the descent
- `P1` 4.018 - additive gamma-embedding EqM, KL_bi at default sample time
- `P2` 0.579 - additive gamma-embedding EqM, H_ratio
- `P3` 3.618 - additive gamma-embedding EqM, KL_bi at gamma=0.5
- `P4` 12.927 - concat-embedding variant, KL_bi
- `P5` 10.276 - concat-embedding variant, KL_bi at gamma=0.5
- `S1` 0.9156 - fig:image-text caption, fraction of positions carrying their final character a
- `S2` 0.7060 - same, after 1 descent step
- `S3` 0.9999 - mean probability on the sampled character at the source ("a clear one-hot stru
- `S4` 12.2733 - per-position radius of the sample ("directly moves to the vertex")
- `S5` 0.4521 - fraction of positions where the sample keeps the character the source already 
- `S6` 0.0373 - chance agreement implied by the source and sample marginals
- `S7` 0.0314 - KL of the sample's letter frequencies to the corpus unigram
- `X2` -2.706 - ln p(true char) at gamma=0.05 before descent (paper prints exp = 0.067)

## 3. Stale anchors on values that should be printed (115 rows) - needs a decision

Each still carries a line number from an older revision of the paper, and the
value is not at that line any more. Three things can be true, and the rows are
not separable mechanically:

1. the number moved and only the anchor is stale;
2. the paper dropped the sentence and the row should be retired or re-noted
   as a backing row;
3. the row is derived (a min/max/band over cells the paper bounds in prose)
   and never had a literal printed form.

Known from the paper's own history: the `heal-insulin` family (H36-H52) belongs
to the insulin healing track that left the paper on 2026-09-07, and `X230`
(`tab:config` insulin `n_demo`) goes with it. `G179`, `N32`, `N33`, `N34` and
`N36` still carry the superseded sqrt-u EqM values that `tab:gen` no longer
prints in any form.

### B (14)

- `B25` 0.01208 @ `chapters/results.tex:tab:eqm-band-bench` - config eqm.gamma_lo, band_L256_ep10_d10k_g012_gs030
- `B26` 0.03573 @ `chapters/results.tex:tab:eqm-band-bench` - config eqm.gamma_star, band_L256_ep10_d10k_g005_gs99
- `B32` 1.191 @ `chapters/results.tex:tab:eqm-band-bench` - config eqm.sample_grad_clip, band_L256_ep10_d10k_g0
- `B52` 0.016 @ `chapters/results.tex:tab:eqm-band-recovery` - rescale factor at alpha=0.5 (caption)
- `B53` 0.013 @ `chapters/results.tex:tab:eqm-band-recovery` - rescale factor at alpha=0.6 (caption)
- `B54` 0.012 @ `chapters/results.tex:tab:eqm-band-recovery` - rescale factor at alpha=0.7 (caption)
- `B55` 0.010 @ `chapters/results.tex:tab:eqm-band-recovery` - rescale factor at alpha=0.8 (caption)
- `B56` 0.008 @ `chapters/results.tex:tab:eqm-band-recovery` - rescale factor at alpha=1.0 (caption)
- `B57` 0.65 @ `chapters/results.tex:sec:eqm-results recovery prose` - band field token accuracy after descent at alpha=0.5 (prose 0.65)
- `B62` 0.94 @ `chapters/results.tex:sec:eqm-results non-reproduction prose` - lowest H_ratio of the flat runs (prose 0.94)
- `B63` 0.99 @ `chapters/results.tex:sec:eqm-results non-reproduction prose` - highest H_ratio of the flat runs (prose 0.99)
- `B70` 1.48 @ `chapters/results.tex:sec:eqm-results non-reproduction prose` - seed-43 replicate KL_bi as printed in prose
- `B76` 0.952 @ `chapters/results.tex:492` - sec:res-eqm-band A_K(t(0.03)), per-position decodability at gamma*=0.03
- `B77` 0.11 @ `chapters/results.tex:492` - sec:res-eqm-band A_K(t(0.005)), per-position decodability at gamma_lo

### D (46)

- `D285` 0.149 @ `chapters/results.tex:628` - tab:ood-prf caption prevalence, replace per character
- `D287` 0.561 @ `chapters/results.tex:628` - tab:ood-prf caption prevalence, replace per word
- `D302` 512 @ `chapters/results.tex:384` - disjoint clean fit sequences
- `D305` 0.979 @ `chapters/results.tex:399` - NLL replace word-max floor over rates<=0.3 (binds at 0.1)
- `D309` 0.854 @ `chapters/results.tex:400` - BLR_all shuffle word-max @0.7
- `D313` 0.78 @ `chapters/results.tex:403` - BLR_fi band low end at 0.15 (shuffle)
- `D314` 0.83 @ `chapters/results.tex:403` - BLR_fi band high end at 0.15 (replace)
- `D315` 0.72 @ `chapters/results.tex:403` - GPT-2 band low end at 0.15 (SE shuffle)
- `D316` 0.75 @ `chapters/results.tex:403` - GPT-2 band high end at 0.15 (NLL shuffle)
- `D317` 0.557 @ `chapters/results.tex:431` - GPT-2 NLL native BPE token AUROC, replace@0.15
- `D319` 13361 @ `chapters/results.tex:432` - clean BPE token count over the 256 windows
- `D320` 22801 @ `chapters/results.tex:432` - BPE token count, replace@0.15
- `D321` 33014 @ `chapters/results.tex:432` - BPE token count, replace@0.5
- `D323` 0.771 @ `chapters/results.tex:432` - share of BPE tokens touching a corrupted char, replace@0.5
- `D324` 1.024 @ `chapters/results.tex:434` - BPE token inflation under falseinfo@0.15 (+2.4%)
- `D325` 0.158 @ `chapters/results.tex:434` - BPE prevalence, falseinfo@0.15
- `D327` 0.555 @ `chapters/results.tex:434` - GPT-2 NLL char-attributed token AUROC, falseinfo@0.15
- `D328` 0.836 @ `chapters/results.tex:436` - GPT-2 NLL replace word-max @0.05
- `D329` 0.701 @ `chapters/results.tex:436` - GPT-2 NLL replace word-max @0.3
- `D330` 0.979 @ `chapters/results.tex:437` - NLL replace word-max band low end over 0.05-0.3
- `D331` 0.982 @ `chapters/results.tex:437` - NLL replace word-max band high end over 0.05-0.3
- `D333` 0.993 @ `chapters/results.tex:441` - BLR energy-channel sequence AUROC, replace@0.15
- `D335` 0.889 @ `chapters/results.tex:441` - BLR energy-channel token AUROC, replace@0.15
- `D336` 0.705 @ `chapters/results.tex:441` - BLR variance-channel token AUROC, falseinfo@0.15
- `D337` 0.529 @ `chapters/results.tex:441` - BLR energy-channel token AUROC, falseinfo@0.15
- `D338` 0.921 @ `chapters/results.tex:478` - weakest sequence AUROC on replace@0.1 (BLR_fi), paper bound 'above 0.92'
- `D339` 0.9965 @ `chapters/results.tex:478` - weakest sequence AUROC on replace@0.2 (BLR_all), paper bound 'above 0.996'
- `D340` 0.998 @ `chapters/results.tex:478` - weakest sequence AUROC on shuffle@0.3 (BLR_all), paper bound '>=0.997'
- `D341` 0.55 @ `chapters/results.tex:480` - falseinfo sequence band low end @0.05 (BLR)
- `D348` 0.765 @ `chapters/results.tex:481` - GPT-2 NLL falseinfo word-max @1.0
- `D351` 0.853 @ `chapters/results.tex:486` - GPT-2 NLL falseinfo word-max @0.15
- `D352` 0.812 @ `chapters/results.tex:486` - BLR_fi falseinfo word-max @0.15
- `D354` 0.83 @ `chapters/results.tex:489` - BLR_fi falseinfo word-max @1.0
- `D356` 0.748 @ `chapters/results.tex:491` - GPT-2 NLL falseinfo word-max @0.5
- `D358` 64 @ `chapters/results.tex:540` - plausible-scheme sequences, n
- `D359` 48 @ `chapters/results.tex:540` - same-length candidate words per slot
- `D366` 0.64 @ `chapters/results.tex:564` - freq-matched control: LOG_pl sequence AUROC on plausible
- `D371` 0.593 @ `chapters/results.tex:571` - PC 1-2 probe, plausible token AUROC @0.15
- `D376` 0.895 @ `chapters/results.tex:613` - weakest sequence F1 on replace@0.15 (BLR_fi), paper bound 'at least 0.89'
- `D381` 0.48 @ `chapters/results.tex:620` - GPT-2 best falseinfo sequence F1 @5%-FPR (GPT-2 NLL)
- `D385` 128 @ `chapters/results.tex:662` - transfer track sequences per domain, n
- `D386` 0.15 @ `chapters/results.tex:662` - transfer track evaluation rate
- `D535` 0.692 @ `chapters/results.tex:697` - sec:ood-falseinfo prose: Var word-max AUROC @0.05
- `D536` 0.666 @ `chapters/results.tex:697` - sec:ood-falseinfo prose: Var word-max AUROC @0.7
- `D537` 0.796 @ `chapters/results.tex:697` - sec:ood-falseinfo prose: Var word-max AUROC @1.0
- `D561` 0.006 @ `chapters/results.tex:676` - word-level NLL, plausible planted word (mean-pool)

### E (1)

- `E1` 0.075 @ `chapters/results.tex:286` - unigram chance sum_k p_k^2

### G (13)

- `G159` 18.7 @ `chapters/results.tex:25` - setup: peak GPU memory at batch 48 (GB)
- `G162` 0.256 @ `chapters/results.tex:94` - gen-scaling: transport gain before the final anneal
- `G165` 0.609 @ `chapters/results.tex:94` - gen-scaling: ten-epoch KL_tri (the 'within 0.001' comparator)
- `G168` 0.3 @ `chapters/results.tex:165` - latent: corruption rate
- `G171` 192 @ `chapters/results.tex:196` - tab:latent caption: axis fit sequences
- `G172` 4000 @ `chapters/results.tex:226` - fig:latent-token caption: tokens per class
- `G175` 0.959 @ `chapters/results.tex:42` - prose: Dirichlet FM KL_bi below one
- `G176` 0.09 @ `chapters/results.tex:42` - prose: Dirichlet FM largest transport gain
- `G177` 0.019 @ `chapters/results.tex:43` - prose: transport-arm KL_uni upper bound (SFLM, the max)
- `G179` 2.226 @ `chapters/results.tex:45` - prose: EqM KL_bi lower bound (EqM_OneHot, the min)
- `G205` 1.000 @ `chapters/results.tex:83` - prose: Fisher FM perturbed accuracy at alpha=0.5
- `G206` 0.999 @ `chapters/results.tex:83` - prose: Statistical FM perturbed accuracy at alpha=0.5
- `G207` 0.49 @ `chapters/results.tex:83` - prose: low end of perturbed-accuracy band, seven non-Fisher-Rao arms at al

### H (8)

- `H36` 0.587 @ `chapters/results.tex:805` - heal-insulin NLL track A fix rate
- `H37` 0.037 @ `chapters/results.tex:805` - heal-insulin NLL track A damage rate
- `H38` 0.377 @ `chapters/results.tex:805` - heal-insulin NLL track A net/corrupt
- `H43` 0.443 @ `chapters/results.tex:806` - heal-insulin BGMM track A loc F1
- `H44` 0.261 @ `chapters/results.tex:806` - heal-insulin BGMM track A fix rate
- `H45` 0.026 @ `chapters/results.tex:806` - heal-insulin BGMM track A damage rate
- `H50` -0.047 @ `chapters/results.tex:806` - heal-insulin BGMM track C net/corrupt
- `H52` 32 @ `chapters/results.tex:794` - heal-insulin caption n_demo

### M (1)

- `M12` 9 @ `chapters/introduction.tex:ch:introduction prose` - arms in the budget-matched benchmark

### N (18)

- `N13` 0.031 @ `chapters/results.tex:319` - KL_uni r1c1 via consolidated readout
- `N3` 0.000 @ `chapters/results.tex:319` - Delta@.50, fixed-interpolant x deterministic clr
- `N31` 0.528 @ `chapters/results.tex:330` - perturbed input accuracy at alpha=0.50, before any descent
- `N32` 0.093 @ `chapters/results.tex:249` - KL_uni, EqM deterministic clr, L=256
- `N33` 2.226 @ `chapters/results.tex:249` - KL_bi, EqM deterministic clr, L=256
- `N34` 0.611 @ `chapters/results.tex:250` - KL_uni, EqM Dirichlet-thickened, L=256
- `N36` 2.612 @ `chapters/results.tex:250` - KL_bi, EqM VAE-latent, L=256
- `N38` 0.959 @ `chapters/results.tex:259` - KL_bi, Dirichlet FM matched budget, L=256
- `N44` 1.692 @ `chapters/results.tex:296` - KL_bi of independent draws from the corpus unigram distribution
- `N45` 3.2958 @ `chapters/results.tex:297` - H_gen of the thickened cell vs the uniform value ln 27
- `N54` 24 @ `chapters/results.tex:310` - number of annealed-Langevin sigma levels, DSM cells
- `N55` 0.01 @ `chapters/results.tex:310` - sigma_min of the DSM ladder
- `N56` 12.3 @ `chapters/results.tex:310` - sigma_max of the DSM ladder
- `N58` 0.5 @ `chapters/results.tex:311` - auxiliary cross-entropy anchor weight, DSM cells
- `N62` 256 @ `chapters/results.tex:308` - n evaluation samples
- `N64` 0.0001 @ `chapters/results.tex:274` - label smoothing epsilon behind the 12.51 clr gap
- `N65` 0.038 @ `chapters/results.tex:286` - token accuracy at the source before the descent
- `N72` 12.2723 @ `chapters/results.tex:310` - clr feature norm (data radius) the DSM sigma_max is matched to

### X (14)

- `X10` 0.947 @ `appendix/appendix.tex:74` - token accuracy after descent at gamma=0.03 (paper prints 5.3% wrong)
- `X11` 0.997 @ `appendix/appendix.tex:74` - token accuracy of the input at gamma=0.04 (paper prints 0.3% wrong)
- `X12` 0.995 @ `appendix/appendix.tex:74` - token accuracy after descent at gamma=0.04 (paper prints 0.5% wrong)
- `X15` 0.0008 @ `appendix/appendix.tex:app:training prose` - auxiliary token-CE at the end of the first epoch
- `X202` 192 @ `appendix/appendix.tex:120` - fit sequences for the latent probes
- `X204` 4.5 @ `appendix/appendix.tex:120` - feature read-out time t_eval
- `X205` 4000 @ `appendix/appendix.tex:162` - per-token class balance cap
- `X230` 32 @ `appendix/appendix.tex:234` - tab:config Healing: insulin n_demo (+32)
- `X24` 0.0807 @ `appendix/appendix.tex:99` - KL_uni, 3.4x larger backbone ('markedly worse on the unigram')
- `X27` 3.35 @ `appendix/appendix.tex:104` - KL_bi, best of seven sample times, additive gamma-conditioned EqM
- `X3` -0.0068 @ `appendix/appendix.tex:70` - ln p(true char) at gamma=0.05 after descent (paper prints exp = 0.993, '0.
- `X4` -11.453 @ `appendix/appendix.tex:71` - ln p(true char) at gamma=0 after descent (paper prints exp = 1.1e-5)
- `X5` -0.0489 @ `appendix/appendix.tex:app:training prose` - ln p(true char) of the raw interpolant at gamma=0.5 (paper prints CE 0.049
- `X9` 0.953 @ `appendix/appendix.tex:74` - token accuracy of the input at gamma=0.03 (paper prints 4.7% wrong)


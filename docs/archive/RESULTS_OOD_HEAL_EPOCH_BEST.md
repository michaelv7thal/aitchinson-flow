# OOD detection + healing — `epoch_best.pt` rerun


> **SUPERSEDED — dated 2026-07-03. Its recommendation is now reversed; do not pick a checkpoint from this document.**
>
> This compares two checkpoints of the *interrupted* full-text8 run: `DirichletFM/epoch_final.pt` (Jun 20) and `DirichletFM/epoch_best.pt` (Jul 2). Neither is the model the paper uses. The paper's downstream readouts all come from the **fully-annealed** `DirichletFM_converge/epoch_final.pt` (Aug 9, md5 `9946dce9…`), which did not exist when this was written, and which reaches a net recovery of **+0.432** — better than both the +0.407 and the +0.343 below.
>
> The paper's finding is therefore the opposite of this document's: "the annealed model repairs better and localizes character-level corruption at least as well, at a small cost on the false-information axis" (`chapters/results.tex` §Experimental Setup).
>
> Kept as the record of the best-vs-final ablation the paper refers to in that sentence, and because its observation that the GPT-2 rows are byte-identical across checkpoints is the reason a GPT-2 number can never date a result. Its seven data pointers (`ood_out*`, `heal_out*`) resolve at the git revision this file was archived at; the trees were removed from HEAD in the 2026-08-20 cleanup.

Full OOD-detection + healing routine re-run against the **best-val** checkpoint of the
full-text8 L256 DirichletFM run, mirroring the earlier `epoch_final.pt` routine
(test split, fine corruption grid, n=256, fit-seqs=512, 5 corruption seeds).

- **Checkpoint:** `runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM/epoch_best.pt` (epoch 5, best val)
- **Baseline:** `epoch_final.pt` (committed results in `ood_out/`, `heal_out/`)
- **New outputs:** `ood_out_best/`, `heal_out_best/` (driver: `drive_heal_ood_best.sh`, log: `heal_out_best/driver.log`)
- Ran locally on an 8 GB laptop GPU (original ran on a 20 GB A100); all 7 arms completed.

## Headline: OOD unchanged, healing improved

Swapping `epoch_final` → `epoch_best` leaves **OOD detection flat** (within ±0.01 AUROC noise)
but makes the model a **meaningfully better healer/denoiser**.

### Healing @ selected fpr=0.02 (mean over 5 seeds, test split)

| localizer | ckpt  | fix   | dmg   | **net/corrupt** | loc_P | loc_R | Δ_ws    |
|-----------|-------|-------|-------|-----------------|-------|-------|---------|
| NLL       | final | 0.507 | 0.029 | +0.343          | 0.808 | 0.710 | +0.0512 |
| NLL       | **best**  | **0.570** | 0.029 | **+0.407**  | 0.816 | 0.750 | +0.0606 |
| linear    | final | 0.439 | 0.051 | +0.147          | 0.651 | 0.692 | +0.0219 |
| linear    | **best**  | **0.461** | 0.047 | **+0.194**  | 0.666 | 0.692 | +0.0289 |

NLL localizer stays the winner (~2× net recovery vs the BLR/linear energy head).

### Iterative heal (NLL, corrupt 0.15, fpr 0.02, 5 iters)

| ckpt  | iter-1 acc | iter-1 net/cor | peak acc | cal F1 | cal thr |
|-------|-----------|----------------|----------|--------|---------|
| final | 0.9071    | +0.384         | 0.9071   | 0.758  | 1.300   |
| **best**  | **0.9173**    | **+0.452**     | **0.9173**   | **0.788**  | 1.478   |

`epoch_best` is more surprised by corruptions (iter-0 mean corrupt NLL 0.754 vs 0.631),
which drives higher localizer precision/recall and larger net recovery.

## OOD detection — unchanged (representative rows, seq-energy / seq-NLL AUROC)

| detector    | scheme  | rate | best   | final  |
|-------------|---------|------|--------|--------|
| BayesLinHead (seq E) | shuffle | 0.10 | 0.9348 | 0.9296 |
| BayesLinHead (seq E) | shuffle | 0.30 | 1.0000 | 0.9989 |
| BayesLinHead (seq E) | replace | 0.10 | 0.9667 | 0.9778 |
| DenoiserNLL (seq NLL)| shuffle | 0.05 | 0.9702 | 0.9643 |
| DenoiserNLL (seq NLL)| shuffle | 0.10 | 0.9990 | 0.9980 |
| DenoiserNLL (tok NLL)| shuffle | 0.30 | 0.8883 | 0.8633 |
| GPT-2 SE (seq)       | shuffle | 0.10 | 1.0000 | 1.0000 |

- Both native detectors saturate to seq-AUROC ≈ 1.0 by rate ≈ 0.15 on all schemes (best ≈ final).
- DenoiserNLL per-token AUROC is marginally **higher** with `epoch_best` at most rates.
- **GPT-2 spilled-energy is byte-identical** best vs final — expected: the external GPT-2
  scorer never reads the DirichletFM weights (same seed + test split → same numbers).
- Latent-split viz regenerated in `ood_out_best/latent_split/`.

## Takeaway

The best-val checkpoint is the one to use for the **healing** story (net/corrupt +0.407 NLL,
peak acc 0.917 iterative) and is a strict, if small, improvement there; the **OOD-detection**
verdicts are identical to `epoch_final`, so those figures/claims carry over unchanged.

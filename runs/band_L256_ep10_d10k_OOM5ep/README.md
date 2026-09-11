# Aborted: 5 of 10 epochs (CUDA OOM in the in-loop unigram probe)

Run of `band_L256_ep10_d10k` on 2026-08-26/27 that died during epoch 5's
`_unigram_kl_probe` (runner.py:305). Training itself was healthy at ~5.3 GB;
the probe samples n=64 x L=256 in ONE batch under second-order autograd, on
top of model + Adam states, and tipped over 7.46 GiB of a 7.57 GiB card.

`run_sweep.py` caught the error, wrote error.txt and exited 0, so
drive_band_L256.sh walked on to stage 2 and built a recovery ladder against
this HALF-TRAINED checkpoint. Those JSONs are quarantined here so the
driver's skip-if-done logic cannot silently mix them with a full 10-epoch
generation number.

Contents:
  epoch_final.pt              epoch 5, global_step 6250 (no optimizer state)
  recovery_bandmap_a03.json   Delta@0.3, 5-epoch model, n=256, 400 steps
  recovery_bandmap_a05.json   Delta@0.5, 5-epoch model, n=256, 400 steps
  error.txt                   the OOM

There is NO eval.json: the generation eval never ran.

Per-epoch probe metrics before the crash (diagnostic protocol, n=64/100 steps
- NOT comparable to tab:gen, which is n=256/400):
  ep1  KL_uni 0.161  KL_bi 2.102  KL_tri 7.807  H_gen 3.149
  ep2  KL_uni 0.065  KL_bi 1.531  KL_tri 5.275  H_gen 2.690
  ep3  KL_uni 0.044  KL_bi 1.573  KL_tri 4.980  H_gen 2.687
  ep4  KL_uni 0.037  KL_bi 1.479  KL_tri 5.057  H_gen 2.759

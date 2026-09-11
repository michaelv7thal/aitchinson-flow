Control-arm artifacts produced by hand on 2026-08-26 (~22:59-23:11) while
drive_band_L256.sh was still in stage-1 training. Moved aside because the
driver's skip-if-done logic would have reused them at the WRONG sample size:

  band_ood.json            n=32   vs the driver's stage-4 --n 256  (8x fewer)
  recovery_bandmap_a05.json  n/steps unrecorded, provenance unclear

Comparing a band arm at n=256 against a control at n=32 is not the like-for-
like control the driver is built to produce, so stage 4 now recomputes both at
matched N. Kept, not deleted: the band_ood.json here is still a valid n=32
reading of the whole-path arm (band_trained=false).

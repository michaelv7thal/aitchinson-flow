# CONFOUNDED — do not report these weights (2026-08-23)

Trained 2026-08-23 10:22–14:30 (EqMAE, gamma_power=1.0, seed 42, L=256).
Discarded and retrained; superseded by `EqMAE_gp1p0/`.

## What was wrong

`scripts/train_for_sflm_bench.py` resolved the frozen-AE config from
`payload["cfg"]["autoencoder"]`, but the VAE checkpoint it was handed
(`runs/vae_a100_20g_L256/epoch_final.pt`) is a *reconstruction* — a bare state
dict with no cfg payload (see that dir's RECONSTRUCTED.md). So `ae_cfg` was
`{}`, every field kept its dataclass default, and `cfg.autoencoder.mode` stayed
at `"ae"` instead of the `"vae"` the checkpoint actually is.

`TextAutoencoder` therefore built a deterministic `to_latent` head, which the
VAE checkpoint does not contain (it has `mu_head`/`logsig_head`). With
`strict=False` the load reported only:

    [EqMAE] AE load: missing=2 unexpected=4;
      first missing: ['to_latent.weight', 'to_latent.bias']
      first unexpected: ['mu_head.weight', 'mu_head.bias', 'logsig_head.weight']

and left `to_latent` at **random init**, then froze it. The dims matched by
luck (the dataclass defaults equal this VAE's d_model=256/d_latent=64/2 layers),
so `mode` was the only wrong field — but it sits on the critical path:
`EqMAE.encode()` → `ae.encode()` → `to_latent(h)`.

Consequences: every flow target `x1` was a frozen *random* linear projection of
the encoder features; that latent is neither KL-calibrated to N(0,I) (which is
what EqMAE's `x0 = source_sigma·randn` source and Euler sampler assume) nor
readable by the frozen decoder, which was trained to decode `μ`.

## Why the published row is unaffected

`runs/sflm_bench_a100_20g_L256/EqMAE/` (Jun 11) loaded the *original* VAE
checkpoint, which carried a cfg payload. Proof: the VAE reconstruction was made
by extracting the 63 `ae.*` tensors out of that published EqMAE checkpoint, and
those tensors are `mu_head`/`logsig_head` with no `to_latent` — i.e. the
published run built the AE in vae mode and loaded it exactly.

So γ was NOT the only changed variable here, and these rows cannot be read
against the published √u row. That is the whole reason for the discard.

## Fix

`_model_cfg()` now falls back to the sibling `config.json` when the checkpoint
carries no cfg payload, and then cross-checks `mode` against the state-dict keys
(`mu_head.weight` present ⇒ vae), overriding and logging on disagreement.
Verified: the AE load is now 0 missing / 0 unexpected, `to_latent is None`.

# epoch_final.pt is a reconstruction (2026-08-21)

The original VAE weights in this directory were lost before the 2026-08-20
cleanup (only config.json and history.jsonl survived; see the cleanup round's
notes: the tab:config VAE row was marked permanently T0/T1).

`epoch_final.pt` was reconstructed on 2026-08-21 by extracting the 63 `ae.*`
tensors from `runs/sflm_bench_a100_20g_L256/EqMAE/epoch_final.pt` and stripping
the `ae.` prefix. This is faithful: EqMAE freezes the AE at construction
(`requires_grad_(False)`, models/eqm_ae.py) and never updates it, so the
embedded copy is identical to the original weights the bench model loaded.

The file is a bare state dict (no cfg/optimizer payload, unlike the original).
Consumers that read `payload["model_state_dict"]` with a bare-dict fallback
(recovery_check.py, eqm_ae.py) handle this. It exists so that EqMAE-class
checkpoints whose cfg points here can be rebuilt for eval (T2).

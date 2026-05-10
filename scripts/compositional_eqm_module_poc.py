"""compositional_eqm_module_poc.py — end-to-end POC of Compositional EqM
through the actual project module (Config → Text8DataModule → EqM → fit).

Verifies that the proposal recipe (Dirichlet-thickened CLR + Hilbert loss +
EqM conservative-gradient + NAG) works through the *real* training pipeline
with a tiny transformer and a small subset of text8.

Run:
    python scripts/compositional_eqm_module_poc.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch

import aitchinson_flow.models  # noqa: F401  populate model REGISTRY
from aitchinson_flow.config import Config
from aitchinson_flow.training import (
    build_training_datamodule,
    fit,
    seed_all,
)


def build_tiny_compositional_cfg() -> Config:
    cfg = Config()

    cfg.training = replace(
        cfg.training,
        epochs=1,
        lr=3e-3,
        L=24,
        K=27,
        B=32,
        log_every=20,
        eval_every=1,
        sample_eval_every=1,
        sample_eval_n=16,
        sample_eval_steps=50,
        checkpoint_dir=str(ROOT / "runs" / "compositional_eqm_poc"),
        checkpoint_every=1,
    )

    cfg.text8_dataset = replace(
        cfg.text8_dataset,
        L=24,
        K=27,
        batch_size=32,
        max_train_windows=512,
        max_eval_windows=128,
        train_corrupt_rate=0.0,
        eval_corrupt_rate=0.0,
    )

    cfg.transformer = replace(
        cfg.transformer,
        d_model=64,
        nhead=4,
        num_layers=2,
        d_latent=64,
    )

    cfg.transformation = replace(
        cfg.transformation,
        dirichlet_sampling=True,
        dirichlet_alpha_peak=10.0,
        dirichlet_alpha_base=0.1,
    )

    cfg.loss = replace(
        cfg.loss,
        mode="hilbert_soft",
        hilbert_alpha=1.0,
        hilbert_alpha_start=1.0,
    )

    cfg.eqm = replace(
        cfg.eqm,
        gamma_power=0.5,
        gradient_lambda=1.0,
        ce_min_gamma=0.5,
        lambda_ce=0.5,
        source_sigma=0.1,
        sample_max_steps=50,
    )

    cfg.loader_settings = replace(
        cfg.loader_settings,
        batch_size=32,
        num_workers=0,
    )

    cfg.wandb = replace(cfg.wandb, enabled=False)

    return cfg


def main() -> None:
    cfg = build_tiny_compositional_cfg()
    Path(cfg.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    seed_all(cfg.training.seed)

    print("=" * 72)
    print("Compositional EqM module POC")
    print("=" * 72)
    print(f"  device      : {cfg.training.device}")
    print(f"  d_model     : {cfg.transformer.d_model}")
    print(f"  layers      : {cfg.transformer.num_layers}")
    print(f"  L, K, B     : {cfg.training.L}, {cfg.training.K}, {cfg.training.B}")
    print(f"  train wins  : {cfg.text8_dataset.max_train_windows}")
    print(f"  epochs      : {cfg.training.epochs}")
    print(f"  dirichlet   : peak={cfg.transformation.dirichlet_alpha_peak} "
          f"base={cfg.transformation.dirichlet_alpha_base}")
    print(f"  loss        : {cfg.loss.mode} (alpha={cfg.loss.hilbert_alpha})")
    print(f"  out_dir     : {cfg.training.checkpoint_dir}")
    print()

    datamodule, meta = build_training_datamodule(cfg)

    # Probe: confirm collate is producing Dirichlet-sampled CLR (variance > 0).
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    x = batch["x"]
    tids = batch["token_ids"]
    print(f"  batch x   shape = {tuple(x.shape)}, mean={x.mean():.4f}, std={x.std():.4f}")
    print(f"  batch tok shape = {tuple(tids.shape)}, range=[{int(tids.min())}, {int(tids.max())}]")
    # Sanity: the same token in two different rows should give *different* x rows
    # under Dirichlet sampling (they would be identical under deterministic CLR).
    flat_x = x.reshape(-1, x.shape[-1])
    flat_t = tids.reshape(-1)
    dup = None
    for tok in range(cfg.training.K):
        rows = (flat_t == tok).nonzero(as_tuple=True)[0]
        if rows.numel() >= 2:
            r0, r1 = rows[0].item(), rows[1].item()
            d = (flat_x[r0] - flat_x[r1]).norm().item()
            dup = (tok, d)
            break
    if dup is not None:
        tok, d = dup
        print(f"  same-token x-row diff (tok={tok}): {d:.4f}  "
              f"({'>0 ⇒ Dirichlet OK' if d > 1e-3 else 'WARN: deterministic'})")
    print()

    model = fit(cfg=cfg, datamodule=datamodule, wandb_logger=None)

    print()
    print("=" * 72)
    print("POC complete — training ran end-to-end through the module pipeline.")
    print("Checkpoints written to:", cfg.training.checkpoint_dir)
    print("=" * 72)

    # Tiny recovery probe — confirms NAG-GD reaches the model.
    val_loader = datamodule.val_dataloader()
    if val_loader is not None and hasattr(model, "sample"):
        val_batch = next(iter(val_loader))
        token_ids = val_batch["token_ids"][:8].to(cfg.training.device)
        from aitchinson_flow.data.transforms import token_ids_to_features_dirichlet
        x_clean = token_ids_to_features_dirichlet(
            token_ids, K=cfg.training.K,
            alpha_peak=cfg.transformation.dirichlet_alpha_peak,
            alpha_base=cfg.transformation.dirichlet_alpha_base,
        ).to(cfg.training.device)
        embed_norm = x_clean.norm(dim=-1).mean().item()

        print()
        print("Recovery probe (Dirichlet-data, NAG-GD on conservative gradient):")
        print(f"  embed_norm = {embed_norm:.3f}")
        print(f"  {'α':>5} {'σ_pert':>8} {'acc_pert':>9} {'acc_recov':>9} {'Δacc':>7}")
        model.eval()
        for alpha in (0.10, 0.30, 0.50):
            sigma = alpha * embed_norm
            torch.manual_seed(123 + int(alpha * 100))
            x_pert = x_clean + sigma * torch.randn_like(x_clean)
            ids_pert = x_pert.argmax(-1)
            with torch.no_grad():
                # `sample` builds its own x0; for recovery we want to start from x_pert.
                # EqM models expose `_compute_grad`/sample-from-x0; use a hand NAG loop.
                pass
            # Hand NAG-GD on the model's conservative gradient.
            x_recov = _nag_recover(model, x_pert, n_steps=80, eta=0.1, mu=0.9, K=cfg.training.K)
            ids_recov = x_recov.argmax(-1)
            acc_p = (ids_pert == token_ids).float().mean().item()
            acc_r = (ids_recov == token_ids).float().mean().item()
            print(f"  {alpha:>5.2f} {sigma:>8.3f} {acc_p:>9.3f} {acc_r:>9.3f} {acc_r-acc_p:>+7.3f}")


def _nag_recover(model, x_init, n_steps: int, eta: float, mu: float, K: int) -> torch.Tensor:
    x = x_init.detach().clone()
    x_last = x.clone()

    def grad_at(x_in):
        x_req = x_in.detach().requires_grad_(True)
        with torch.enable_grad():
            f = model(x_req)
            energy = (x_req * f).sum()
            g = torch.autograd.grad(energy, x_req)[0].detach()
        return g

    g = grad_at(x)
    for _ in range(n_steps):
        x_last = x
        x = x - eta * g
        g = grad_at(x + mu * (x - x_last))
    return x


if __name__ == "__main__":
    main()

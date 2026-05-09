"""Train and evaluate a DirichletFM auditor on cached HaluEval-QA features.

End-to-end script: builds the datamodule, trains the denoiser on clean rows
only, evaluates the closed-form mixture-of-Dirichlets EBM on held-out rows
(both clean and hallucinated), and reports row-level AUROC against the
per-pair label.

This is the operational test of the proposal in the conversation:
*Stark et al.'s Dirichlet conditional path induces a tractable EBM
``-log p_t(x|h)`` via the trained denoiser; we use it as a UQ surface
on top-K LM features.*

Usage::

    python scripts/run_dirichlet_fm_auditor.py [--epochs 5 ...]

Outputs land under ``runs/dfm_auditor_hallueval/`` by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401  populates registry
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.hallueval_dfm import HalluevalDFMDataModule  # noqa: E402
from aitchinson_flow.models.factory import build_model  # noqa: E402


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Direction-aware ROC-AUC: returns max(auc, 1-auc).

    scores: (n,) higher = more "positive" by the chosen direction; we
    auto-flip if AUC < 0.5 so the reported value is the discriminative
    power, not the sign convention.
    """
    s = scores.detach().cpu().double()
    y = labels.detach().cpu().long()
    # Sort by score
    order = torch.argsort(s, descending=True)
    y_sorted = y[order]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    cum_pos = torch.cumsum((y_sorted == 1).double(), dim=0)
    auc = (cum_pos * (y_sorted == 0).double()).sum() / (n_pos * n_neg)
    auc = float(auc.item())
    return max(auc, 1.0 - auc)


def expected_calibration_error(
    probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15
) -> float:
    """Standard ECE on probabilities in [0, 1]."""
    p = probs.detach().cpu().double().clamp(0.0, 1.0)
    y = labels.detach().cpu().double()
    if len(p) == 0:
        return float("nan")
    bins = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    for i in range(n_bins):
        in_bin = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if not in_bin.any():
            continue
        bin_acc = float(y[in_bin].mean().item())
        bin_conf = float(p[in_bin].mean().item())
        ece += (in_bin.sum().item() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def evaluate(
    model: torch.nn.Module,
    val_loader,
    *,
    device: torch.device,
    energy_t: float,
) -> dict[str, object]:
    """Run the closed-form EBM on the val loader and report AUROC/ECE."""
    model.eval()
    E_mean_all: list[torch.Tensor] = []
    E_max_all: list[torch.Tensor] = []
    DeltaE_all: list[torch.Tensor] = []
    label_all: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in val_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            out = model.energy_at_lm_distribution(batch, t=energy_t)
            E_mean_all.append(out["E_seq_mean"])
            E_max_all.append(out["E_seq_max"])
            DeltaE_all.append(out["DeltaE_seq"])
            label_all.append(out["label"].long())

    E_mean = torch.cat(E_mean_all).cpu()
    E_max = torch.cat(E_max_all).cpu()
    DeltaE = torch.cat(DeltaE_all).cpu()
    label = torch.cat(label_all).cpu()

    auc_E_mean = auroc(E_mean, label)
    auc_E_max = auroc(E_max, label)
    auc_deltaE = auroc(DeltaE, label)

    # Calibrate the EBM into a probability via 1D Platt-like sigmoid on
    # standardised E_seq_mean — the sigmoid slope/intercept are NOT trained;
    # we use the cheap z-score sigmoid that any UQ method can produce. ECE
    # is reported only as a sanity that the score is on a usable scale.
    z = (E_mean - E_mean.mean()) / (E_mean.std() + 1e-9)
    prob = torch.sigmoid(z)
    ece = expected_calibration_error(prob, label)

    return {
        "auc_E_seq_mean": auc_E_mean,
        "auc_E_seq_max": auc_E_max,
        "auc_DeltaE_seq": auc_deltaE,
        "ece_E_seq_mean_zsigmoid": ece,
        "n_val": int(label.numel()),
        "n_clean": int((label == 0).sum().item()),
        "n_halluc": int((label == 1).sum().item()),
        "energy_t": float(energy_t),
        "_E_mean": E_mean,
        "_E_max": E_max,
        "_DeltaE": DeltaE,
        "_label": label,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(REPO / "runs" / "dfm_auditor_hallueval"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-rows", type=int, default=4000)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--energy-t", type=float, default=4.0)
    parser.add_argument("--t-max", type=float, default=8.0)
    parser.add_argument("--ctx-proj-dim", type=int, default=64)
    parser.add_argument(
        "--mode",
        choices=("off", "hidden_only", "product_concat"),
        default="product_concat",
        help="context conditioning: off=simplex only, hidden_only=h_LLM only, "
             "product_concat=concat(simplex, h_LLM)",
    )
    # Back-compat: --no-context is equivalent to --mode off.
    parser.add_argument("--no-context", action="store_true",
                        help="alias for --mode off (kept for back-compat)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--save-best", action="store_true",
                        help="also save the checkpoint with best val AUROC over training")
    # Allow the cluster runbook to override the cache paths and ctx_hidden
    # without editing the dataclass defaults.
    parser.add_argument("--topk-cache", default=None)
    parser.add_argument("--hidden-cache", default=None)
    parser.add_argument("--ctx-hidden", type=int, default=768,
                        help="raw LM hidden dim (768=GPT-2, 2048=Llama-1B, 4096=Llama-7B)")
    parser.add_argument(
        "--backbone",
        choices=("transformer", "mlp"),
        default="transformer",
        help="encoder type — `transformer` is the default cross-attention "
             "model; `mlp` is a weight-shared per-position MLP with no "
             "cross-positional information flow (cascade-clean by construction)",
    )
    parser.add_argument(
        "--train-on-all", action="store_true",
        help="strictly self-supervised: train slot CE on ALL rows without "
             "inspecting labels (vs. default one-class clean-only filter)",
    )
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    cfg.training = replace(
        cfg.training,
        model_name="DirichletFMAuditor",
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )
    cfg.transformer = replace(
        cfg.transformer,
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
        d_latent=args.d_model,
        dropout=0.0,
    )
    mode = "off" if args.no_context else args.mode
    cfg.dirichlet_fm = replace(
        cfg.dirichlet_fm,
        t_max=args.t_max,
        energy_t=args.energy_t,
        context_features=mode,
        ctx_hidden=args.ctx_hidden,
        ctx_proj_dim=args.ctx_proj_dim,
        dropout=0.0,
        backbone_kind=args.backbone,
        train_clean_only=not args.train_on_all,
    )
    he = cfg.hallueval_dfm_auditor
    cfg.hallueval_dfm_auditor = replace(
        he,
        enabled=True,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        topk_cache_path=args.topk_cache or he.topk_cache_path,
        hidden_cache_path=args.hidden_cache or he.hidden_cache_path,
    )

    print(
        f"[dfm-auditor] cfg: model_name={cfg.training.model_name} "
        f"epochs={args.epochs} lr={args.lr} d_model={args.d_model} "
        f"L={args.num_layers} t_max={args.t_max} energy_t={args.energy_t} "
        f"context={cfg.dirichlet_fm.context_features}"
    )

    dm = HalluevalDFMDataModule(cfg.hallueval_dfm_auditor)
    print(f"[dfm-auditor] dataset: {dm.meta}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    # The cache dims may differ from the constructor defaults — reshape now.
    model.set_dims(K=dm.meta["K"], L=dm.meta["L"])
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[dfm-auditor] model params: {n_params / 1e6:.2f} M, device={device}")

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    if val_loader is None:
        raise RuntimeError("val_loader is None — check train_frac")

    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup = max(1, args.warmup_steps)

    def lr_at(step_idx: int) -> float:
        if step_idx < warmup:
            return args.lr * step_idx / warmup
        # cosine to lr/20 over the remaining schedule
        import math
        progress = (step_idx - warmup) / max(1, total_steps - warmup)
        progress = min(1.0, max(0.0, progress))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.lr * (0.05 + 0.95 * cos)

    global_step = 0
    history: list[dict] = []
    best_auc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        ce_sum = 0.0
        ce_n = 0
        for step, batch in enumerate(train_loader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            for g in optim.param_groups:
                g["lr"] = lr_at(global_step)
            out = model.training_step(batch, step)
            loss = out["loss"]
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            ce_sum += float(loss.detach().item()) * batch["label"].shape[0]
            ce_n += batch["label"].shape[0]
            global_step += 1
        train_ce = ce_sum / max(ce_n, 1)

        eval_out = evaluate(model, val_loader, device=device, energy_t=args.energy_t)
        log = {
            "epoch": epoch,
            "train_ce": train_ce,
            "auc_E_seq_mean": eval_out["auc_E_seq_mean"],
            "auc_E_seq_max": eval_out["auc_E_seq_max"],
            "auc_DeltaE_seq": eval_out["auc_DeltaE_seq"],
            "ece_E_seq_mean": eval_out["ece_E_seq_mean_zsigmoid"],
            "n_val": eval_out["n_val"],
        }
        history.append(log)
        print(
            f"[dfm-auditor] ep {epoch:02d}  train_ce={train_ce:.4f}  "
            f"auc(E_mean)={log['auc_E_seq_mean']:.4f}  "
            f"auc(E_max)={log['auc_E_seq_max']:.4f}  "
            f"auc(ΔE)={log['auc_DeltaE_seq']:.4f}  "
            f"ece={log['ece_E_seq_mean']:.4f}  "
            f"lr={lr_at(global_step):.2e}"
        )
        if args.save_best and log["auc_E_seq_mean"] > best_auc:
            best_auc = log["auc_E_seq_mean"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Save final checkpoint and history
    ckpt = {
        "model_state_dict": model.state_dict(),
        "cfg": cfg,
        "epoch": args.epochs,
        "K": dm.meta["K"],
        "L": dm.meta["L"],
    }
    torch.save(ckpt, out_dir / "epoch_final.pt")
    if args.save_best and best_state is not None:
        torch.save(
            {**ckpt, "model_state_dict": best_state, "best_auc": best_auc},
            out_dir / "best.pt",
        )
        print(f"[dfm-auditor] best val AUROC = {best_auc:.4f}; saved best.pt")
    summary = {
        "history": history,
        "cfg": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_rows": args.max_rows,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "nhead": args.nhead,
            "energy_t": args.energy_t,
            "t_max": args.t_max,
            "context_features": cfg.dirichlet_fm.context_features,
            "ctx_proj_dim": args.ctx_proj_dim,
            "n_params_million": n_params / 1e6,
        },
        "data": dm.meta,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[dfm-auditor] wrote {out_dir / 'epoch_final.pt'}")
    print(f"[dfm-auditor] wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

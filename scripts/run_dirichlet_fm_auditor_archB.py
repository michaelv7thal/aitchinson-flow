"""Architecture B — DirichletFM auditor with joint slot + hallucination heads.

Trains the encoder for two objectives simultaneously:

  * Slot prediction CE on **clean** rows (preserves the EBM):
      ``L_slot = CE(slot_head(encoder(x_t, t, h)), next_slot)`` for clean rows.
  * Hallucination BCE on **all** rows (supervised UQ head):
      ``L_halluc = BCE(halluc_head(encoder(x_t, t, h)), row_label)`` over
      answer-mask positions, all rows.

Total loss: ``λ_slot * L_slot + λ_halluc * L_halluc``.

This is the "encoder-feature joint training" architecture; Architecture A
(post-hoc SVGP on a frozen B-trained encoder) can be applied on top by
running ``run_dfm_auditor_archA.py`` with the resulting checkpoint.

Targets (vs Architecture A on the slot-only encoder):

  * row AUROC ≥ 0.92
  * per-tok AUROC at answer span ≥ 0.80
  * per-tok AUROC at non-answer ≤ 0.55 (preserve cascade locality)
  * per-tok ECE ≤ 0.05

Usage::

    python scripts/run_dirichlet_fm_auditor_archB.py \\
        --epochs 15 --batch-size 8 --d-model 512 --num-layers 6 \\
        --mode product_concat --save-best \\
        --out-dir runs/dfm_auditor_archB
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.hallueval_dfm import HalluevalDFMDataModule  # noqa: E402
from aitchinson_flow.models.factory import build_model  # noqa: E402


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    s = scores.detach().cpu().double()
    y = labels.detach().cpu().long()
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = torch.argsort(s, descending=True)
    ys = y[order]
    npos = int(y.sum())
    nneg = int(len(y) - npos)
    cum = torch.cumsum((ys == 1).double(), 0)
    auc = float(((cum * (ys == 0).double()).sum() / (npos * nneg)).item())
    return max(auc, 1.0 - auc)


def expected_calibration_error(p: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> float:
    p = p.detach().cpu().double().clamp(0.0, 1.0)
    y = y.detach().cpu().double()
    if len(p) == 0:
        return float("nan")
    bins = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if not in_bin.any():
            continue
        ece += (in_bin.sum().item() / n) * abs(
            float(y[in_bin].mean()) - float(p[in_bin].mean())
        )
    return float(ece)


@torch.no_grad()
def evaluate(
    model,
    val_loader,
    *,
    device: torch.device,
    energy_t: float,
) -> dict[str, object]:
    """Return AUROC + ECE for both the EBM signal and the halluc-head signal.

    The EBM is computed at the LM's actual top-K distribution; the halluc
    head is computed on the same encoder features.  We log:

      * EBM row-level mean/max AUROC
      * Halluc head per-token AUROC at answer / non-answer (locality)
      * Halluc head row-pooled AUROC + ECE
    """
    model.eval()

    E_mean_buf, E_max_buf, DeltaE_buf, label_buf = [], [], [], []
    P_pos_buf, ans_buf, label_pos_buf = [], [], []

    for batch in val_loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        out_E = model.energy_at_lm_distribution(batch, t=energy_t)
        E_mean_buf.append(out_E["E_seq_mean"].cpu())
        E_max_buf.append(out_E["E_seq_max"].cpu())
        DeltaE_buf.append(out_E["DeltaE_seq"].cpu())
        label_buf.append(out_E["label"].long().cpu())

        if model.halluc_head is not None:
            out_H = model.halluc_score_at_lm(batch, t=energy_t)
            B, L = out_H["prob_pos"].shape
            P_pos_buf.append(out_H["prob_pos"].cpu())
            ans_buf.append(out_H["answer_mask"].cpu())
            label_pos_buf.append(
                out_H["label"].long().unsqueeze(-1).expand(B, L).cpu()
            )

    E_mean = torch.cat(E_mean_buf)
    E_max = torch.cat(E_max_buf)
    DeltaE = torch.cat(DeltaE_buf)
    label = torch.cat(label_buf)

    log = {
        "auc_E_seq_mean": auroc(E_mean, label),
        "auc_E_seq_max": auroc(E_max, label),
        "auc_DeltaE_seq": auroc(DeltaE, label),
        "n_val": int(label.numel()),
    }

    if P_pos_buf:
        P = torch.cat(P_pos_buf, dim=0)            # (B_total, L)
        ans = torch.cat(ans_buf, dim=0)
        ypos = torch.cat(label_pos_buf, dim=0)
        # Per-token at answer span
        log["auc_tok_ans"] = auroc(P[ans], ypos[ans])
        log["auc_tok_non_ans"] = auroc(P[~ans], ypos[~ans])
        log["ece_tok_ans"] = expected_calibration_error(P[ans], ypos[ans])
        # Row-level: mean over answer mask, fallback to all positions
        denom = ans.sum(dim=-1).clamp_min(1).float()
        prow = (torch.where(ans, P, torch.zeros_like(P))).sum(dim=-1) / denom
        log["auc_row_halluc"] = auroc(prow, label)
        log["ece_row_halluc"] = expected_calibration_error(prow, label)

    return log


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(REPO / "runs" / "dfm_auditor_archB"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=4000)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--energy-t", type=float, default=4.0)
    parser.add_argument("--t-max", type=float, default=8.0)
    parser.add_argument("--ctx-proj-dim", type=int, default=64)
    parser.add_argument(
        "--mode",
        choices=("off", "hidden_only", "product_concat"),
        default="product_concat",
    )
    parser.add_argument("--lambda-slot", type=float, default=1.0)
    parser.add_argument("--lambda-halluc", type=float, default=1.0)
    parser.add_argument("--halluc-pos-weight", type=float, default=0.0,
                        help="<=0 = auto-compute from train class balance")
    # Larger context dims for Llama-class LMs (1B=2048, 3B=3072, 7B=4096).
    parser.add_argument("--ctx-hidden", type=int, default=768,
                        help="raw LM hidden dim (768=GPT-2, 2048=Llama-1B, 4096=Llama-7B)")
    parser.add_argument(
        "--backbone",
        choices=("transformer", "mlp"),
        default="transformer",
        help="encoder type — `mlp` is the cascade-clean per-position variant",
    )
    # Path to a different cache (e.g. a Llama-built one) without touching cfg.
    parser.add_argument("--topk-cache", default=None)
    parser.add_argument("--hidden-cache", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--save-best", action="store_true")
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
    cfg.dirichlet_fm = replace(
        cfg.dirichlet_fm,
        t_max=args.t_max,
        energy_t=args.energy_t,
        context_features=args.mode,
        ctx_hidden=args.ctx_hidden,
        ctx_proj_dim=args.ctx_proj_dim,
        dropout=0.0,
        joint_halluc=True,
        lambda_slot=args.lambda_slot,
        lambda_halluc=args.lambda_halluc,
        halluc_pos_weight=args.halluc_pos_weight,
        backbone_kind=args.backbone,
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
        f"[archB] cfg: model_name={cfg.training.model_name} "
        f"epochs={args.epochs} lr={args.lr} d_model={args.d_model} "
        f"L={args.num_layers} mode={args.mode} t_max={args.t_max} "
        f"energy_t={args.energy_t} ctx_hidden={args.ctx_hidden} "
        f"λ_slot={args.lambda_slot} λ_halluc={args.lambda_halluc}"
    )

    dm = HalluevalDFMDataModule(cfg.hallueval_dfm_auditor)
    print(f"[archB] dataset: {dm.meta}")
    if dm.meta["H"] != args.ctx_hidden:
        raise RuntimeError(
            f"--ctx-hidden ({args.ctx_hidden}) does not match cache H ({dm.meta['H']}). "
            f"Set --ctx-hidden to the LM's actual hidden dim or rebuild the cache."
        )

    # Auto-compute pos_weight if not set. BCEWithLogitsLoss(pos_weight=p)
    # scales the positive-class loss by p; balanced training needs
    # p = N_neg / N_pos. On HaluEval-QA halluc answers are the majority
    # among answer-span positions (~6:1) so pos_weight ends up < 1.
    if args.halluc_pos_weight <= 0:
        n_pos = 0
        n_neg = 0
        for b in dm.train_dataloader():
            ans = b["answer_mask"].bool()
            label = b["label"].bool()
            B, L = ans.shape
            label_pos = label.unsqueeze(-1).expand(B, L)
            n_pos += int((ans & label_pos).sum().item())
            n_neg += int((ans & ~label_pos).sum().item())
        pos_weight = float(n_neg) / max(1, n_pos)
        cfg.dirichlet_fm = replace(cfg.dirichlet_fm, halluc_pos_weight=pos_weight)
        print(
            f"[archB] auto pos_weight={pos_weight:.4f}  "
            f"(n_pos={n_pos} halluc-ans-pos, n_neg={n_neg} clean-ans-pos)"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    model.set_dims(K=dm.meta["K"], L=dm.meta["L"])
    if not hasattr(model, "halluc_head") or model.halluc_head is None:
        raise RuntimeError(
            "Architecture B requires joint_halluc=True; halluc_head is None"
        )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[archB] model params: {n_params/1e6:.2f} M  device={device}")

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
        progress = (step_idx - warmup) / max(1, total_steps - warmup)
        progress = min(1.0, max(0.0, progress))
        return args.lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))

    history: list[dict] = []
    best_auc = -1.0
    best_state = None
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        slot_sum = halluc_sum = 0.0
        n = 0
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
            slot_sum += float(out.get("ce_slot", torch.zeros(())).item())
            halluc_sum += float(out.get("bce_halluc", torch.zeros(())).item())
            n += 1
            global_step += 1
        log = {
            "epoch": epoch,
            "ce_slot": slot_sum / max(n, 1),
            "bce_halluc": halluc_sum / max(n, 1),
            "lr": lr_at(global_step),
        }
        log.update(evaluate(model, val_loader, device=device, energy_t=args.energy_t))
        history.append(log)
        print(
            f"[archB] ep {epoch:02d}  "
            f"ce={log['ce_slot']:.3f}  bce={log['bce_halluc']:.3f}  "
            f"row(EBM)={log['auc_E_seq_mean']:.4f}  "
            f"row(halluc)={log.get('auc_row_halluc', float('nan')):.4f}  "
            f"tok(ans)={log.get('auc_tok_ans', float('nan')):.4f}  "
            f"tok(non)={log.get('auc_tok_non_ans', float('nan')):.4f}  "
            f"ece={log.get('ece_row_halluc', float('nan')):.4f}"
        )
        if args.save_best:
            score = log.get("auc_row_halluc", float("nan"))
            if not math.isnan(score) and score > best_auc:
                best_auc = score
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

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
            {**ckpt, "model_state_dict": best_state, "best_auc_row_halluc": best_auc},
            out_dir / "best.pt",
        )
        print(f"[archB] best val row AUROC = {best_auc:.4f}; saved best.pt")

    summary = {
        "history": history,
        "cfg": {
            "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
            "max_rows": args.max_rows, "d_model": args.d_model,
            "num_layers": args.num_layers, "nhead": args.nhead,
            "energy_t": args.energy_t, "t_max": args.t_max,
            "context_features": args.mode, "ctx_proj_dim": args.ctx_proj_dim,
            "ctx_hidden": args.ctx_hidden,
            "lambda_slot": args.lambda_slot, "lambda_halluc": args.lambda_halluc,
            "halluc_pos_weight": cfg.dirichlet_fm.halluc_pos_weight,
            "n_params_million": n_params / 1e6,
        },
        "data": dm.meta,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[archB] wrote {out_dir / 'epoch_final.pt'}")
    print(f"[archB] wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

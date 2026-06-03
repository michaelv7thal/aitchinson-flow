"""Ablation: does the flow-matching stage matter for OOD, or is a direct
hinge comparable?  (Answers the "FM-then-hinge vs hinge-only" question.)

Four arms, all using the SAME linear energy-head hinge readout
(``model.energy_head`` on pooled features) trained on **replace-only**
negatives (``corrupt_token_ids`` @ 0.15) — so they differ only in the
*representation* / what is trainable. Evaluated on a corruption ladder; the
discriminating axis is **shuffle** (histogram-preserving → needs sequence
order), which is also the *transfer* test (train on replace, eval on shuffle).

  A  FM + hinge      — load the FM-pretrained DFM, freeze backbone, train
                       energy_head (+pooler) via the hinge.            [current design]
  B  random + hinge  — same, but the backbone is random-init (FM ablated).
                       Isolates "does FM pretraining matter?".
  C  end-to-end hinge— random-init, train backbone+pooler+energy_head with the
                       hinge only (no FM/CE). The literal "direct hinge".
  D  FM + linear probe— FM-pretrained backbone frozen; a logistic-regression
                       probe on pooled features. Isolates "is the hinge special
                       vs any readout on the same representation?".

Prediction: A ≈ D ≫ B on shuffle (representation does the work, not the head);
C strong on replace but weak on replace→shuffle transfer (corruption-specific).

Usage (cluster, after the L256 DirichletFMSvgp Stage-1 is trained — see
CLUSTER_RUNBOOK_L256.md §3; a plain DirichletFM ckpt also works, keys remapped):
    python scripts/ablate_hinge_vs_fm.py \
        --dfm-ckpt runs/dfm_svgp_L256/epoch_final.pt \
        --n-train 4000 --n-eval 1000 --epochs 3 \
        --out runs/sflm_bench_a100_20g_L256/ablate_hinge_vs_fm.json
    # local sanity (no HF / no ckpt, synthetic tokens):
    python scripts/ablate_hinge_vs_fm.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace as _replace
from pathlib import Path


def _bootstrap() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    if str(root) not in sys.path:
        sys.path.append(str(root))


_bootstrap()

import torch  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.corruption import (  # noqa: E402
    corrupt_token_ids,
    partially_shuffle_token_ids,
)
from aitchinson_flow.models import build_model  # noqa: E402

_TRAIN_CORRUPT = 0.15  # negatives the hinge trains on (replace only)


def _pool(model, tok: torch.Tensor, t: torch.Tensor, *, grad: bool) -> torch.Tensor:
    """token_ids (B,L) → pooled features (B, d_embed). ``grad`` controls
    whether the backbone forward is differentiable (we build the path by hand
    because ``model.pool_features`` gates grad on a config flag)."""
    x_t = model.dfm._sample_xt(tok.long(), t)
    with torch.set_grad_enabled(grad):
        h = model.get_hidden_states(x_t, t)
        return model.pooler(h)


@torch.no_grad()
def _energy(model, tok: torch.Tensor, t_eval: float) -> torch.Tensor:
    """Energy-head OOD score (higher = more OOD). (B,)."""
    t = torch.full((tok.shape[0],), float(t_eval), device=tok.device)
    z = _pool(model, tok, t, grad=False)
    return model.energy_head(z).squeeze(-1)


def _set_trainable(model, *, backbone: bool, pooler: bool, energy: bool) -> list:
    for p in model.dfm.parameters():
        p.requires_grad_(backbone)
    for p in model.pooler.parameters():
        p.requires_grad_(pooler)
    for p in model.energy_head.parameters():
        p.requires_grad_(energy)
    return [p for p in model.parameters() if p.requires_grad]


def _hinge_train(
    model,
    clean: torch.Tensor,
    K: int,
    *,
    t_eval: float,
    margin: float,
    epochs: int,
    lr: float,
    bs: int,
    backbone: bool,
    seed: int,
) -> float:
    """Train energy_head (+pooler, +backbone if requested) so that
    E(replace-negative) > E(clean) + margin. Returns final mean hinge."""
    device = clean.device
    params = _set_trainable(model, backbone=backbone, pooler=True, energy=True)
    opt = torch.optim.Adam(params, lr=lr)
    model.train()
    if not backbone:
        model.dfm.eval()  # frozen backbone: keep dropout/BN off
    n = clean.shape[0]
    last = float("nan")
    for ep in range(epochs):
        g = torch.Generator(device="cpu").manual_seed(seed + ep)
        order = torch.randperm(n, generator=g)
        for s in range(0, n, bs):
            idx = order[s : s + bs]
            cl = clean[idx]
            neg = corrupt_token_ids(
                cl,
                vocab_size=K,
                corrupt_rate=_TRAIN_CORRUPT,
                seed=seed + ep * 100000 + s,
            )
            t = torch.full((cl.shape[0],), float(t_eval), device=device)
            e_c = model.energy_head(_pool(model, cl, t, grad=True)).squeeze(-1)
            e_n = model.energy_head(_pool(model, neg, t, grad=True)).squeeze(-1)
            hinge = torch.relu(margin + e_c - e_n).mean()
            reg = 1e-3 * (e_c.pow(2).mean() + e_n.pow(2).mean())  # anchor scale
            loss = hinge + reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last = float(hinge.detach())
    model.eval()
    return last


def _auroc(scores_clean: torch.Tensor, scores_corrupt: torch.Tensor) -> float:
    from sklearn.metrics import roc_auc_score
    import numpy as np

    s = torch.cat([scores_clean, scores_corrupt]).float().cpu().numpy()
    y = np.concatenate(
        [np.zeros(scores_clean.numel()), np.ones(scores_corrupt.numel())]
    )
    if (y == 1).sum() < 2 or (y == 0).sum() < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _eval_arm(
    score_fn, clean: torch.Tensor, K: int, rates: list[float], seed: int
) -> dict:
    """score_fn: token_ids -> (B,) OOD score (higher=OOD). Returns AUROC per
    {replace, shuffle} × rate."""
    sc_clean = score_fn(clean)
    out: dict = {"replace": {}, "shuffle": {}}
    for r in rates:
        rep = corrupt_token_ids(clean, vocab_size=K, corrupt_rate=r, seed=seed + 1)
        sh = partially_shuffle_token_ids(clean, shuffle_rate=r, seed=seed + 2)
        out["replace"][f"{r:.1f}"] = _auroc(sc_clean, score_fn(rep))
        out["shuffle"][f"{r:.1f}"] = _auroc(sc_clean, score_fn(sh))
    return out


def _load_clean(cfg, n_train: int, n_eval: int, seed: int, smoke: bool, device):
    if smoke:
        K, L = cfg.text8_dataset.K, cfg.text8_dataset.L
        g = torch.Generator().manual_seed(seed)
        tr = torch.randint(0, K, (n_train, L), generator=g)
        ev = torch.randint(0, K, (n_eval, L), generator=g)
        return tr.to(device), ev.to(device)
    from aitchinson_flow.training import build_training_datamodule

    dm, _ = build_training_datamodule(cfg)
    tr = dm.splits.train.long()
    ev = getattr(dm.splits, "test", None)
    ev = ev if (ev is not None and ev.numel()) else dm.splits.val
    ev = ev.long()
    g = torch.Generator().manual_seed(seed)
    tr = tr[torch.randperm(tr.shape[0], generator=g)[:n_train]]
    ev = ev[torch.randperm(ev.shape[0], generator=g)[:n_eval]]
    return tr.to(device), ev.to(device)


def _fresh_model(cfg, device):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return build_model(cfg).to(device)


def _load_fm_into_svgp(model, state) -> float:
    """Load an FM-pretrained backbone into a DirichletFMSvgp. Accepts a
    DirichletFMSvgp checkpoint (keys already under ``dfm.``) OR a plain
    DirichletFM/DFM checkpoint (top-level keys → prefixed here). Loads only
    shape-matching keys, then **asserts the backbone actually received
    weights** — train_for_sflm_bench's "DFM" arm is a *different* model, and a
    silent strict=False miss would leave arm A random and invalidate the
    ablation."""
    msd = model.state_dict()
    if not any(k.startswith("dfm.") for k in state):
        state = {f"dfm.{k}": v for k, v in state.items()}
    # Merge matching ckpt keys onto the model's *own* full state and load
    # strict=True: a partial state (e.g. a plain DirichletFM ckpt with no svgp
    # keys) otherwise trips gpytorch's variational-strategy load hook.
    merged = dict(msd)
    matched_keys = []
    for k, v in state.items():
        if k in merged and merged[k].shape == v.shape:
            merged[k] = v
            matched_keys.append(k)
    model.load_state_dict(merged, strict=True)
    dfm_keys = [k for k in msd if k.startswith("dfm.")]
    matched = sum(1 for k in dfm_keys if k in matched_keys)
    frac = matched / max(len(dfm_keys), 1)
    if frac < 0.5:
        raise RuntimeError(
            f"FM backbone load covered only {matched}/{len(dfm_keys)} dfm.* "
            f"params ({frac:.0%}) — wrong checkpoint type? Arm A would be random. "
            "Pass a DirichletFMSvgp (preferred) or DirichletFM checkpoint."
        )
    print(f"[load] FM backbone matched {matched}/{len(dfm_keys)} dfm.* params ({frac:.0%})")
    return frac


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dfm-ckpt",
        default=None,
        help="FM-pretrained checkpoint for arms A & D — a DirichletFMSvgp "
        "Stage-1 (preferred) or a DirichletFM checkpoint (keys remapped). "
        "NOT the train_for_sflm_bench 'DFM' arm (different model).",
    )
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--t-eval", type=float, default=None)
    ap.add_argument("--rates", type=str, default="0.3,0.5,1.0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="synthetic tokens, no HF/ckpt; tiny self-test of all arms.",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rates = [float(x) for x in args.rates.split(",") if x.strip()]
    torch.manual_seed(args.seed)

    # --- config -------------------------------------------------------------
    if args.smoke:
        from scripts.eval_full import _config_from_payload  # noqa: F401  (parity)

        cfg = Config()
        cfg.training = _replace(
            cfg.training, model_name="DirichletFMSvgp", L=24, K=cfg.text8_dataset.K
        )
        cfg.text8_dataset = _replace(cfg.text8_dataset, L=24)
        cfg.transformer = _replace(cfg.transformer, d_model=64, num_layers=2, nhead=4)
        args.n_train, args.n_eval, args.epochs = 64, 64, 1
        state = None
    else:
        if not args.dfm_ckpt:
            ap.error("--dfm-ckpt is required unless --smoke")
        from scripts.eval_full import _config_from_payload

        payload = torch.load(args.dfm_ckpt, map_location="cpu", weights_only=False)
        cfg = _config_from_payload(payload)
        cfg.training = _replace(cfg.training, model_name="DirichletFMSvgp")
        state = payload.get("model_state_dict", payload)

    t_eval = float(args.t_eval if args.t_eval is not None else cfg.dfm_svgp.t_eval)
    K = cfg.text8_dataset.K
    clean_tr, clean_ev = _load_clean(
        cfg, args.n_train, args.n_eval, args.seed, args.smoke, device
    )
    common = dict(
        t_eval=t_eval,
        margin=args.margin,
        epochs=args.epochs,
        lr=args.lr,
        bs=args.bs,
        seed=args.seed,
    )
    results: dict = {}

    def _run_hinge_arm(name: str, load_fm: bool, backbone: bool):
        model = _fresh_model(cfg, device)
        if load_fm:
            if state is None:
                print(f"[{name}] (smoke) no FM weights — using random init")
            else:
                _load_fm_into_svgp(model, state)
        fh = _hinge_train(model, clean_tr, K, backbone=backbone, **common)
        sc = _eval_arm(
            lambda tok, m=model: _energy(m, tok, t_eval), clean_ev, K, rates, args.seed
        )
        results[name] = {"final_hinge": fh, **sc}
        print(
            f"[{name}] final_hinge={fh:.3f}  "
            f"shuffle={ {k: round(v, 3) for k, v in sc['shuffle'].items()} }"
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("== A: FM + hinge ==")
    _run_hinge_arm("A_fm_hinge", True, False)
    print("== B: random + hinge ==")
    _run_hinge_arm("B_rand_hinge", False, False)
    print("== C: end-to-end hinge ==")
    _run_hinge_arm("C_e2e_hinge", False, True)

    # --- D: FM + linear probe ----------------------------------------------
    print("== D: FM + linear probe ==")
    from sklearn.linear_model import LogisticRegression

    model = _fresh_model(cfg, device)
    if state is not None:
        _load_fm_into_svgp(model, state)
    model.eval()
    t = torch.full((clean_tr.shape[0],), t_eval, device=device)
    with torch.no_grad():
        zc = _pool(model, clean_tr, t, grad=False).cpu().numpy()
        neg = corrupt_token_ids(
            clean_tr, vocab_size=K, corrupt_rate=_TRAIN_CORRUPT, seed=args.seed
        )
        zn = _pool(model, neg, t, grad=False).cpu().numpy()
    import numpy as np

    X = np.concatenate([zc, zn])
    y = np.concatenate([np.zeros(len(zc)), np.ones(len(zn))])
    clf = LogisticRegression(max_iter=1000).fit(X, y)

    def _probe_score(tok):
        tt = torch.full((tok.shape[0],), t_eval, device=device)
        with torch.no_grad():
            z = _pool(model, tok, tt, grad=False).cpu().numpy()
        return torch.tensor(clf.decision_function(z))

    scD = _eval_arm(_probe_score, clean_ev, K, rates, args.seed)
    results["D_fm_probe"] = scD
    print(
        f"[D_fm_probe] shuffle={ {k: round(v, 3) for k, v in scD['shuffle'].items()} }"
    )

    # --- report -------------------------------------------------------------
    rmax = f"{max(rates):.1f}"
    print("\n" + "=" * 72)
    print(f"ABLATION — AUROC (1=perfect). Headline = shuffle@{rmax} (order axis,")
    print("trained on replace-only negatives → this is the replace→shuffle transfer).")
    print("=" * 72)
    print(f"{'arm':14s}  {'replace@' + rmax:>12s}  {'shuffle@' + rmax:>12s}")
    for name in ("A_fm_hinge", "B_rand_hinge", "C_e2e_hinge", "D_fm_probe"):
        r = results[name]
        print(f"{name:14s}  {r['replace'][rmax]:>12.3f}  {r['shuffle'][rmax]:>12.3f}")
    print(
        "\nRead: A≈D≫B on shuffle ⇒ the FM representation (not the hinge) does the\n"
        "work; C high on replace but low on shuffle ⇒ direct hinge doesn't transfer."
    )

    out = {
        "_meta": {
            "smoke": args.smoke,
            "t_eval": t_eval,
            "rates": rates,
            "n_train": args.n_train,
            "n_eval": args.n_eval,
            "epochs": args.epochs,
            "ckpt": args.dfm_ckpt,
        },
        **results,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

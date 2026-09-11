"""Training loop for Hilbert Flow Matching.

Single config dict at the top; no Hydra, no Lightning. Run with::

    python -m hilbert_fm.train
"""

from __future__ import annotations

import json
import os
import time

import torch

from .data import K, encode, get_batch, load_corpus, split_train_val
from .model import HilbertFMTransformer
from .path import log_p1_from_ids, log_pt, make_log_p0, soft_hilbert


CONFIG: dict = dict(
    K=K,  # 27 (26 letters + space)
    L=64,
    d_model=256,
    n_layers=4,
    n_heads=4,
    eps_smooth=0.01,
    tau=0.1,
    batch_size=32,
    lr=3e-4,
    weight_decay=0.01,
    grad_clip=1.0,
    n_steps=20_000,
    t_bias_power=1.0,  # t = u^0.5; mild bias toward t -> 1
    t_train_max=0.99,  # never sample t = 1.0 (gradient vanishes)
    # Stochastic source: each (b, l) draws a random token whose label-smoothed
    # log-one-hot is the source. With "uniform" the source is the constant
    # 1/K and every sample produces the same trajectory at inference, so
    # sampling from noise collapses to the data unigram. See path.py.
    source_kind="random_token",
    log_every=100,
    val_every=500,
    save_every=2_000,
    out_dir="hilbert_fm/runs/default",
    seed=0,
    device=None,  # None -> auto (cuda if available)
)


def sample_t(B: int, t_bias_power: float, t_max: float, device) -> torch.Tensor:
    """``t = u^p`` with ``u ~ U(0, 1)``, then clamped into ``(0, t_max]``."""
    u = torch.rand(B, device=device)
    return u.pow(t_bias_power).clamp(min=1e-4, max=t_max)


def _per_t_val_loss(
    model, ids_pool, cfg, device, n_batches: int = 4
) -> dict[str, float]:
    """B. Per-t bucket validation loss in ``[0, 1/3), [1/3, 2/3), [2/3, t_max]``."""
    model.eval()
    buckets: dict[str, list[float]] = {"low": [], "mid": [], "high": []}
    bounds = [
        ("low", 1e-4, 0.33),
        ("mid", 0.33, 0.66),
        ("high", 0.66, cfg["t_train_max"]),
    ]
    with torch.no_grad():
        for _ in range(n_batches):
            ids = get_batch(ids_pool, cfg["batch_size"], cfg["L"], device)
            log_p1 = log_p1_from_ids(ids, cfg["K"], cfg["eps_smooth"])
            log_p0 = make_log_p0(
                cfg["source_kind"],
                cfg["batch_size"],
                cfg["L"],
                cfg["K"],
                cfg["eps_smooth"],
                device,
            )
            for name, lo, hi in bounds:
                t = torch.empty(cfg["batch_size"], device=device).uniform_(lo, hi)
                lpt = log_pt(log_p0, log_p1, t)
                lph = model(lpt, t)
                buckets[name].append(soft_hilbert(lph, log_p1, cfg["tau"]).item())
    return {k: round(sum(v) / len(v), 4) for k, v in buckets.items()}


def train(config: dict | None = None):
    cfg = dict(CONFIG if config is None else config)
    device = cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)

    text = load_corpus()
    ids_all = encode(text)
    train_ids, val_ids = split_train_val(ids_all)
    print(
        f"[train] corpus={len(ids_all):,} train={len(train_ids):,} val={len(val_ids):,}"
    )

    model = HilbertFMTransformer(
        K=cfg["K"],
        L=cfg["L"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params={n_params / 1e6:.2f}M device={device}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    history: list[dict] = []
    bucket_history: list[dict] = []
    t_start = time.time()

    for step in range(1, cfg["n_steps"] + 1):
        model.train()
        ids = get_batch(train_ids, cfg["batch_size"], cfg["L"], device)
        log_p1 = log_p1_from_ids(ids, cfg["K"], cfg["eps_smooth"])
        log_p0 = make_log_p0(
            cfg["source_kind"],
            cfg["batch_size"],
            cfg["L"],
            cfg["K"],
            cfg["eps_smooth"],
            device,
        )
        t = sample_t(cfg["batch_size"], cfg["t_bias_power"], cfg["t_train_max"], device)
        lpt = log_pt(log_p0, log_p1, t)
        lph = model(lpt, t)
        loss = soft_hilbert(lph, log_p1, cfg["tau"])

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg["grad_clip"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()

        if step % cfg["log_every"] == 0:
            elapsed = time.time() - t_start
            history.append(dict(step=step, loss=float(loss.item()), elapsed=elapsed))
            print(
                f"[train] step={step:6d} loss={loss.item():.4f} elapsed={elapsed:.1f}s"
            )

        if step % cfg["val_every"] == 0:
            buckets = _per_t_val_loss(model, val_ids, cfg, device)
            bucket_history.append(dict(step=step, **buckets))
            print(f"[val]   step={step:6d} buckets={buckets}")

        if step % cfg["save_every"] == 0:
            ckpt_path = os.path.join(cfg["out_dir"], f"ckpt_step{step}.pt")
            torch.save(dict(model=model.state_dict(), config=cfg, step=step), ckpt_path)

    final_path = os.path.join(cfg["out_dir"], "final.pt")
    torch.save(
        dict(model=model.state_dict(), config=cfg, step=cfg["n_steps"]), final_path
    )
    with open(os.path.join(cfg["out_dir"], "history.json"), "w") as f:
        json.dump(dict(loss=history, buckets=bucket_history, config=cfg), f, indent=2)
    print(f"[train] saved {final_path}")
    return model, history, bucket_history


PRESETS: dict[str, dict] = {
    # First fallback in the spec's escalation ladder: sharper soft-Hilbert
    # (tau -> 0 approaches the true Hilbert metric) to push samples toward the
    # vertex. Trained into a separate run dir so default and sharp coexist.
    "sharp": dict(tau=0.1, out_dir="hilbert_fm/runs/sharp"),
}


def get_config(preset: str | None = None, **overrides) -> dict:
    cfg = dict(CONFIG)
    if preset:
        if preset not in PRESETS:
            raise KeyError(f"unknown preset {preset!r}; available: {list(PRESETS)}")
        cfg.update(PRESETS[preset])
    cfg.update(overrides)
    return cfg


if __name__ == "__main__":
    import sys

    preset = sys.argv[1] if len(sys.argv) > 1 else None
    train(get_config(preset))

"""Required diagnostics for the Hilbert FM experiment.

Implements A (loss curve), B (per-t bucket loss), C (sample-vs-data unigram +
KL), D (printed sample strings), E (reconstruction accuracy from t=0.5), and
F (Hilbert-distance trajectory during sampling). Run with::

    python -m hilbert_fm.diagnose <ckpt.pt>
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .data import K, decode, encode, get_batch, load_corpus, split_train_val
from .path import advance, hilbert_distance, log_p1_from_ids, log_pt, make_log_p0
from .sample import load_model, sample


# ----- helpers -----------------------------------------------------------------


def _kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return float((p * (p.log() - q.log())).sum())


def _unigram(ids: torch.Tensor, K_: int) -> torch.Tensor:
    counts = Counter(ids.flatten().tolist())
    p = torch.tensor([counts.get(i, 0) for i in range(K_)], dtype=torch.float32)
    return p / p.sum().clamp(min=1)


def _bigram_freq_1d(seq: torch.Tensor, K_: int) -> torch.Tensor:
    """Bigram frequency table for a 1D token sequence, shape ``(K_ * K_,)``.

    Always returns a CPU tensor so callers can mix CPU corpus stats with GPU
    sample stats without device-mismatch errors.
    """
    seq = seq.detach().to("cpu", torch.long)
    counts = torch.bincount(seq[:-1] * K_ + seq[1:], minlength=K_ * K_).float()
    return counts / counts.sum().clamp(min=1)


def _bigram_freq_2d(ids: torch.Tensor, K_: int) -> torch.Tensor:
    """Bigram frequencies across a ``(B, L)`` batch (no cross-row bigrams)."""
    ids = ids.detach().to("cpu", torch.long)
    a = ids[:, :-1].reshape(-1)
    b = ids[:, 1:].reshape(-1)
    counts = torch.bincount(a * K_ + b, minlength=K_ * K_).float()
    return counts / counts.sum().clamp(min=1)


# ----- plots -------------------------------------------------------------------


def plot_loss_curve(history: list[dict], path: str) -> None:
    steps = [h["step"] for h in history]
    losses = [h["loss"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, losses)
    ax.set_xlabel("step")
    ax.set_ylabel("soft Hilbert loss")
    ax.set_title("A. Training loss")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_bucket_curve(buckets: list[dict], path: str) -> None:
    steps = [b["step"] for b in buckets]
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in ("low", "mid", "high"):
        ax.plot(steps, [b[k] for b in buckets], label=k)
    ax.set_xlabel("step")
    ax.set_ylabel("val loss")
    ax.set_title("B. Per-t bucket validation loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_unigram(sample_ids: torch.Tensor, train_ids: torch.Tensor, path: str) -> float:
    sample_p = _unigram(sample_ids, K)
    train_p = _unigram(train_ids, K)
    kl = _kl(sample_p, train_p)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = list(range(K))
    labels = list("abcdefghijklmnopqrstuvwxyz") + ["_"]
    ax.bar([i - 0.2 for i in x], train_p.tolist(), width=0.4, label="data")
    ax.bar([i + 0.2 for i in x], sample_p.tolist(), width=0.4, label="samples")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(f"C. Unigram distribution (KL(sample || data) = {kl:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return kl


def plot_hilbert_trajectory(traj: list[torch.Tensor], path: str) -> list[float]:
    """``d_H(p_{t_k}, p_{t_{k+1}})`` averaged over batch and positions."""
    dists = []
    for k in range(len(traj) - 1):
        d = hilbert_distance(traj[k], traj[k + 1]).mean().item()
        dists.append(d)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dists)
    ax.set_xlabel("sampling step k")
    ax.set_ylabel("d_H(p_{t_k}, p_{t_{k+1}})")
    ax.set_title("F. Hilbert distance between successive iterates")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return dists


# ----- E: reconstruction -------------------------------------------------------


def reconstruction_accuracy(
    model,
    val_ids: torch.Tensor,
    cfg: dict,
    device,
    t_start: float = 0.5,
    n_steps: int = 25,
    n_batches: int = 4,
) -> float:
    """Build ``p_1`` from real text, walk to ``t_start``, then sample to ``t_max``.

    Returns mean character-level exact-match accuracy.
    """
    model.eval()
    correct = 0
    total = 0
    L = cfg["L"]
    K_ = cfg["K"]
    t_max = cfg.get("t_train_max", 0.99)
    with torch.no_grad():
        for _ in range(n_batches):
            ids = get_batch(val_ids, cfg["batch_size"], L, device)
            log_p1 = log_p1_from_ids(ids, K_, cfg["eps_smooth"])
            log_p0 = make_log_p0(
                cfg.get("source_kind", "uniform"), cfg["batch_size"], L, K_,
                cfg["eps_smooth"], device,
            )
            t0 = torch.full((cfg["batch_size"],), t_start, device=device)
            log_p_t = log_pt(log_p0, log_p1, t0)
            ts = torch.linspace(t_start, t_max, n_steps + 1)
            for k in range(n_steps):
                tn = float(ts[k])
                tx = float(ts[k + 1])
                tb = torch.full((cfg["batch_size"],), tn, device=device)
                lph = model(log_p_t, tb)
                log_p_t = advance(log_p_t, lph, tn, tx)
            pred = log_p_t.argmax(dim=-1)
            correct += int((pred == ids).sum())
            total += int(ids.numel())
    return correct / total


# ----- top-level ---------------------------------------------------------------


def run_all(
    ckpt_path: str,
    history_path: str | None = None,
    out_dir: str | None = None,
    n_samples: int = 64,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(ckpt_path, device)
    out_dir = out_dir or os.path.join(os.path.dirname(ckpt_path), "diagnostics")
    os.makedirs(out_dir, exist_ok=True)

    history_path = history_path or os.path.join(os.path.dirname(ckpt_path), "history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            h = json.load(f)
        if h.get("loss"):
            plot_loss_curve(h["loss"], os.path.join(out_dir, "A_loss.png"))
        if h.get("buckets"):
            plot_bucket_curve(h["buckets"], os.path.join(out_dir, "B_buckets.png"))

    text = load_corpus()
    all_ids = encode(text)
    train_ids, val_ids = split_train_val(all_ids)

    sample_ids, traj = sample(
        model, B=n_samples, L=cfg["L"], K=cfg["K"], device=device,
        return_trajectory=True, t_max=cfg.get("t_train_max", 0.99),
        source_kind=cfg.get("source_kind", "uniform"),
        eps_smooth=cfg["eps_smooth"],
    )
    kl = plot_unigram(sample_ids, train_ids, os.path.join(out_dir, "C_unigram.png"))
    dists = plot_hilbert_trajectory(traj, os.path.join(out_dir, "F_hilbert_traj.png"))

    print("D. Sample sequences:")
    sample_strings = []
    for i in range(min(5, n_samples)):
        s = decode(sample_ids[i])
        sample_strings.append(s)
        print(f"  {i}: {s!r}")

    recon = reconstruction_accuracy(model, val_ids, cfg, device)
    print(f"E. Reconstruction accuracy from t=0.5: {recon:.3f}")
    print(f"C. Sample-vs-data unigram KL: {kl:.4f}")

    # Bigram-distribution KL: with K = 27 there are only 729 possible bigrams
    # and text8 contains ~all of them, so set-membership coverage saturates
    # near 1.0 trivially. KL on the full bigram frequency table is the
    # meaningful version. Keep direction (sample || data) consistent with C.
    train_bigram_p = _bigram_freq_1d(train_ids[: 5_000_000], K)
    sample_bigram_p = _bigram_freq_2d(sample_ids, K)
    bigram_kl = _kl(sample_bigram_p, train_bigram_p)
    print(f"C+. Bigram-distribution KL(sample || data): {bigram_kl:.4f}")

    summary = dict(
        unigram_kl=kl,
        bigram_kl=bigram_kl,
        reconstruction_acc=recon,
        hilbert_traj=dists,
        sample_strings=sample_strings,
    )
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "hilbert_fm/runs/default/final.pt"
    run_all(ckpt)

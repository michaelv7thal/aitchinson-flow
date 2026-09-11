"""E6b (P2) — EqM-energy auditor parity on cached LLM features.

Trains an EqM-style **energy auditor** on top of frozen, cached LLM features
(see ``scripts/cache_llm_features.py``): the auditor scores a (hidden_state,
token) pair with a scalar energy ``E(h_i, x_i)`` — low for the token the LLM's
context actually supports, high for a mismatched token. Trained with a
contrastive hinge on valid/corrupted pairs (the EqM anti-collapse objective).

This is a **parity check, not a SOTA claim** (the GPT-2/Qwen auditor line is
otherwise retired — see CLAUDE.md): its value is that the EqM energy is a
drop-in for the SVGP head *and* a generator. Reports per-token and per-sequence
AUROC vs a training-free **per-token NLL** baseline read off the cached top-k
logits, so the two are directly comparable on the same corrupted set. (That
baseline was called "spilled energy" until 2026-07; it is an NLL — the real
cross-step ΔE of Minut et al. ICLR 2026 lives in
``scripts/bench_sflm_ebm._gpt2_bpe_scores``.)

Usage:
  python scripts/cache_llm_features.py --model Qwen/Qwen2.5-1.5B --4bit --out runs/llm_cache/wt2
  python scripts/train_eqm_auditor.py --cache runs/llm_cache/wt2 --epochs 5
  python scripts/train_eqm_auditor.py --smoke      # builds a tiny synthetic cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.append(str(_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _load_cache(cache_dir: Path) -> dict:
    man = json.loads((cache_dir / "features_manifest.json").read_text())
    arrs = {}
    for name, spec in man["arrays"].items():
        arrs[name] = np.array(np.memmap(cache_dir / f"{name}.dat",
                                        dtype=spec["dtype"], mode="r",
                                        shape=tuple(spec["shape"])))
    arrs["_manifest"] = man
    return arrs


class EnergyAuditor(nn.Module):
    """EqM-style implicit energy over (h_LLM, token). The token is embedded and
    fused with the detached LLM hidden state; a small MLP maps to a per-token
    feature, and the energy is the log-sum-exp readout against a learned
    codebook (the EqM ``E = -τ·logsumexp_v <φ, e_v>`` form)."""

    def __init__(self, H: int, vocab: int, d: int = 128, tau: float = 1.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d)
        self.fuse = nn.Sequential(nn.Linear(H + d, d), nn.GELU(), nn.Linear(d, d))
        self.codebook = nn.Parameter(torch.randn(vocab, d) * 0.1)
        self.tau = tau

    def phi(self, h: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        e = self.tok_emb(tok)
        return self.fuse(torch.cat([h, e], dim=-1))  # (..., d)

    def energy(self, h: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        scores = self.phi(h, tok) @ self.codebook.t()  # (..., vocab)
        return -self.tau * torch.logsumexp(scores / self.tau, dim=-1)  # (...,)


def _auroc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    p, n = pos.flatten(), neg.flatten()
    if p.numel() == 0 or n.numel() == 0:
        return float("nan")
    return float((p[:, None] > n[None, :]).float().mean()
                 + 0.5 * (p[:, None] == n[None, :]).float().mean())


def _per_position_nll(topk_logits: torch.Tensor, topk_idx: torch.Tensor,
                      tok: torch.Tensor) -> torch.Tensor:
    """Training-free NLL ≈ logsumexp(top-k logits) − logit(token) per position,
    using only the cached top-k (an approximation of the full-vocab NLL).

    RENAMED (2026-07) from ``_spilled_energy``: this is a SAME-STEP per-token NLL,
    not the CROSS-STEP spilled energy of Minut, Dewidar & Masi (ICLR 2026,
    arXiv:2602.18671). See ``scripts/bench_sflm_ebm._gpt2_bpe_scores``."""
    lse = torch.logsumexp(topk_logits, dim=-1)  # (B,T)
    match = (topk_idx == tok.unsqueeze(-1))      # (B,T,k)
    picked = torch.where(match.any(-1),
                         (topk_logits * match).sum(-1),
                         topk_logits.min(-1).values - 5.0)  # OOV → below the floor
    return lse - picked  # (B,T) high = surprising = OOD


def run(cache: dict, *, epochs: int, lr: float, d: int, seed: int,
        device: str) -> dict:
    torch.manual_seed(seed)
    man = cache["_manifest"]
    H, vocab = man["H"], int(max(man["arrays"]["topk_idx"]["shape"][-1] * 0,
                                 cache["topk_idx"].max() + 1))
    tok = torch.from_numpy(cache["token_ids"]).long()
    hid = torch.from_numpy(cache["hidden"]).float()
    lengths = torch.from_numpy(cache["lengths"]).long()
    tkl = torch.from_numpy(cache["topk_logits"]).float()
    tki = torch.from_numpy(cache["topk_idx"]).long()
    vocab = int(max(int(tok.max()), int(tki.max())) + 1)
    N, T = tok.shape
    mask = (torch.arange(T)[None, :] < lengths[:, None])  # (N,T) valid positions

    n_tr = max(1, int(0.7 * N))
    tr, ev = slice(0, n_tr), slice(n_tr, N)
    model = EnergyAuditor(H, vocab, d=d).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def corrupt(t):  # substitution negatives: random wrong tokens
        return torch.randint(0, vocab, t.shape, device=t.device)

    for ep in range(epochs):
        model.train()
        h, x, m = hid[tr].to(device), tok[tr].to(device), mask[tr].to(device)
        e_pos = model.energy(h, x)
        e_neg = model.energy(h, corrupt(x))
        margin = 1.0
        hinge = F.relu(margin + e_pos - e_neg)
        loss = (hinge * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"  ep{ep} hinge={float(loss):.4f}")

    model.eval()
    with torch.no_grad():
        h, x, m = hid[ev].to(device), tok[ev].to(device), mask[ev].to(device)
        xc = corrupt(x)
        e_clean = model.energy(h, x)   # (B,T)
        e_corr = model.energy(h, xc)
        se = _per_position_nll(tkl[ev].to(device), tki[ev].to(device), x)
        se_c = _per_position_nll(tkl[ev].to(device), tki[ev].to(device), xc)
        mb = m.bool()
        tok_auroc = _auroc(e_corr[mb], e_clean[mb])
        seq_auroc = _auroc((e_corr * m).sum(-1), (e_clean * m).sum(-1))
        se_tok_auroc = _auroc(se_c[mb], se[mb])
        se_seq_auroc = _auroc((se_c * m).sum(-1), (se * m).sum(-1))
    return {
        "auditor": {"tok_auroc": tok_auroc, "seq_auroc": seq_auroc},
        "gpt2_nll_baseline": {"tok_auroc": se_tok_auroc, "seq_auroc": se_seq_auroc},
        "n_train": n_tr, "n_eval": N - n_tr, "vocab": vocab, "H": H,
    }


def _make_synthetic_cache(out: Path, *, N=40, T=12, H=16, k=5, vocab=20) -> None:
    """Tiny synthetic cache where the 'correct' token is decodable from the
    hidden state (so a trained auditor should beat chance)."""
    rng = np.random.default_rng(0)
    W = rng.standard_normal((H, vocab)).astype("float32")
    out.mkdir(parents=True, exist_ok=True)
    hid = rng.standard_normal((N, T, H)).astype("float32")
    logits = hid @ W  # (N,T,vocab)
    tok = logits.argmax(-1).astype("int32")  # the token the hidden state supports
    tkl = np.sort(logits, -1)[..., -k:][..., ::-1].astype("float16")
    tki = np.argsort(logits, -1)[..., -k:][..., ::-1].astype("int32")
    leng = np.full((N,), T, dtype="int32")
    for name, a in [("token_ids", tok), ("lengths", leng),
                    ("topk_logits", tkl), ("topk_idx", tki),
                    ("hidden", hid.astype("float16"))]:
        mm = np.memmap(out / f"{name}.dat", dtype=a.dtype, mode="w+", shape=a.shape)
        mm[:] = a
        mm.flush()
    man = {"model": "synthetic", "n": N, "T": T, "k": k, "H": H, "arrays": {
        "token_ids": {"shape": [N, T], "dtype": "int32"},
        "lengths": {"shape": [N], "dtype": "int32"},
        "topk_logits": {"shape": [N, T, k], "dtype": "float16"},
        "topk_idx": {"shape": [N, T, k], "dtype": "int32"},
        "hidden": {"shape": [N, T, H], "dtype": "float16"},
    }}
    (out / "features_manifest.json").write_text(json.dumps(man))


def _smoke() -> None:
    import tempfile
    cdir = Path(tempfile.mkdtemp()) / "cache"
    _make_synthetic_cache(cdir)
    res = run(_load_cache(cdir), epochs=8, lr=1e-2, d=32, seed=0, device="cpu")
    a = res["auditor"]
    assert 0.0 <= a["tok_auroc"] <= 1.0001 and 0.0 <= a["seq_auroc"] <= 1.0001, res
    print(f"OK train_eqm_auditor smoke: auditor tok_auroc={a['tok_auroc']:.3f} "
          f"seq_auroc={a['seq_auroc']:.3f} | SE baseline "
          f"tok={res['gpt2_nll_baseline']['tok_auroc']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=str, default=None,
                    help="dir written by cache_llm_features.py")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
        return
    if not args.cache:
        ap.error("--cache is required unless --smoke")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = run(_load_cache(Path(args.cache)), epochs=args.epochs, lr=args.lr,
              d=args.d, seed=args.seed, device=device)
    res["cache"] = args.cache
    out = args.out or str(Path(args.cache) / "eqm_auditor_parity.json")
    Path(out).write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")
    print(f"  auditor:  tok_auroc={res['auditor']['tok_auroc']:.3f}  "
          f"seq_auroc={res['auditor']['seq_auroc']:.3f}")
    print(f"  gpt2-nll: tok_auroc={res['gpt2_nll_baseline']['tok_auroc']:.3f}  "
          f"seq_auroc={res['gpt2_nll_baseline']['seq_auroc']:.3f}")


if __name__ == "__main__":
    main()

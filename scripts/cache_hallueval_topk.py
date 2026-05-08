"""Top-K + per-position energies cache for Hilbert-FM UQ on HaluEval-QA.

Augments the existing ``data/hallueval_cache_gpt2.pt`` (which stores
hidden states + per-position NLL but *not* the LM's full distribution)
with the features the Hilbert-FM student needs:

  * ``topk_logp``  : (2n, L, K) — log-probabilities of the top-K next-token
                     predictions at every position (already shifted: at row
                     ``r``, position ``i`` holds the LM's distribution
                     conditioned on tokens ``[..., i]``, i.e. the predictor
                     of ``x_{i+1}``).
  * ``topk_idx``   : (2n, L, K) long — vocabulary ids matching ``topk_logp``.
  * ``E_logit``    : (2n, L) — ``E^ℓ_i = −logit_{i-1}[id(x_i)]``, the
                     paper's "logit energy" of the placed token.
  * ``E_marg``     : (2n, L) — ``E^m_i = −logsumexp(logits_i)``, the
                     paper's "marginal energy".
  * ``DeltaE``     : (2n, L) — paper's spilled energy ``E^ℓ − E^m``.

The existing cache supplies ``full_ids``, ``attn_mask``, ``answer_mask``,
``label``, ``pair_id``, and ``hidden_states``. We re-run the LM here only
because raw logits weren't (couldn't be) stored.

Usage::

    python scripts/cache_hallueval_topk.py \\
        --base data/hallueval_cache_gpt2.pt \\
        --out  data/hallueval_topk_gpt2.pt \\
        --K 32 --max-n 10000 --lm-batch-size 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from aitchinson_flow.data.hallueval import load_cache  # noqa: E402


@torch.no_grad()
def _forward_topk_and_energies(
    model,
    full_ids: torch.Tensor,   # (N, L) long
    attn_mask: torch.Tensor,  # (N, L) bool
    *,
    K: int,
    batch_size: int,
):
    """Return (topk_logp, topk_idx, E_logit, E_marg) on CPU.

    Logit conventions match the paper:
      * ``E_logit[r, i] = -logits[r, i-1, full_ids[r, i]]``
      * ``E_marg [r, i] = -logsumexp(logits[r, i, :])``
    so spilled-energy ``ΔE[r, i] = E_logit[r, i] - E_marg[r, i]``.
    Position 0 has no predecessor; we set ``E_logit[:, 0] = 0`` and the
    answer-span starts well past 0 in HaluEval-QA so this is harmless.
    """
    device = next(model.parameters()).device
    N, L = full_ids.shape
    out_topk_lp = torch.empty(N, L, K, dtype=torch.float32)
    out_topk_id = torch.empty(N, L, K, dtype=torch.long)
    out_eℓ = torch.zeros(N, L, dtype=torch.float32)
    out_em = torch.zeros(N, L, dtype=torch.float32)

    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        ids_b = full_ids[s:e].to(device)
        attn_b = attn_mask[s:e].to(device)
        out = model(ids_b, attention_mask=attn_b)
        logits = out.logits.float()                          # (B, L, V)
        logp = torch.log_softmax(logits, dim=-1)             # (B, L, V)

        topk_lp, topk_id = logp.topk(K, dim=-1)              # (B, L, K)
        out_topk_lp[s:e] = topk_lp.cpu()
        out_topk_id[s:e] = topk_id.cpu()

        # Marginal energy: -logsumexp(logits) per (row, position).
        em = -torch.logsumexp(logits, dim=-1)                # (B, L)
        out_em[s:e] = em.cpu()

        # Logit energy at position i = -logits[i-1, id(x_i)].
        # Pull logits[i-1, id(x_i)] via shift-and-gather; pad with 0 at i=0.
        prev = torch.cat(
            [logits.new_zeros((logits.shape[0], 1, logits.shape[-1])), logits[:, :-1, :]],
            dim=1,
        )                                                    # (B, L, V) shifted
        gathered = prev.gather(-1, ids_b.unsqueeze(-1)).squeeze(-1)   # (B, L)
        eℓ = -gathered
        eℓ[:, 0] = 0.0                                       # no prior at i=0
        out_eℓ[s:e] = eℓ.cpu()

        del out, logits, logp, em, prev, gathered
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if s % (batch_size * 20) == 0:
            print(f"  [topk] {s + batch_size:5d}/{N}")

    return out_topk_lp, out_topk_id, out_eℓ, out_em


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True,
                   help="path to existing hallueval_cache_<lm>.pt")
    p.add_argument("--out", required=True,
                   help="path to write the top-K cache")
    p.add_argument("--lm", default="gpt2")
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--max-n", type=int, default=None,
                   help="if set, only process the first 2*max-n rows")
    p.add_argument("--lm-batch-size", type=int, default=8)
    args = p.parse_args(argv)

    print(f"[topk] loading base cache {args.base}")
    base = load_cache(args.base)
    full_ids = base["full_ids"]
    attn_mask = base["attn_mask"]
    if args.max_n is not None:
        N = min(full_ids.shape[0], 2 * args.max_n)
        full_ids = full_ids[:N]
        attn_mask = attn_mask[:N]
    print(f"  N={full_ids.shape[0]} L={full_ids.shape[1]} K={args.K}")

    print(f"[topk] loading LM {args.lm}")
    from transformers import AutoModelForCausalLM
    dtype = torch.float16 if "Qwen" in args.lm else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.lm, dtype=dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"  device={device} dtype={dtype}")

    topk_lp, topk_id, e_l, e_m = _forward_topk_and_energies(
        model, full_ids, attn_mask, K=args.K, batch_size=args.lm_batch_size,
    )
    delta = e_l - e_m

    cache = dict(
        lm=args.lm,
        K=args.K,
        L=int(full_ids.shape[1]),
        N=int(full_ids.shape[0]),
        topk_logp=topk_lp,
        topk_idx=topk_id,
        E_logit=e_l,
        E_marg=e_m,
        DeltaE=delta,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out_path)
    print(f"[topk] wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Sanity: paper-style spilled energy should be non-trivial on answer-span
    # tokens; print mean over clean vs hallucinated rows.
    label = base["label"][: full_ids.shape[0]].bool()
    answer_mask = base["answer_mask"][: full_ids.shape[0]].bool()

    def _mean_over_answer(t: torch.Tensor, mask: torch.Tensor) -> float:
        m = mask.float()
        denom = m.sum(dim=-1).clamp(min=1)
        return float(((t * m).sum(dim=-1) / denom).mean())

    delta_clean = _mean_over_answer(delta[~label], answer_mask[~label])
    delta_inv = _mean_over_answer(delta[label], answer_mask[label])
    print(f"[topk] sanity: mean ΔE per answer-token  "
          f"clean={delta_clean:+.4f}  hallucinated={delta_inv:+.4f}")


if __name__ == "__main__":
    main()

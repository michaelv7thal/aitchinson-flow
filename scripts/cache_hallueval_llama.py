"""Unified HaluEval-QA cache for any HF causal LM (GPT-2, Llama, Mistral, …).

Builds **both** caches the DirichletFM auditor needs in a single LM forward
pass:

  * ``hidden_cache_path``  — schema-compatible with cache_hallueval.py:
        ``full_ids, attn_mask, hidden_states (last layer), SE_pos,
         answer_mask, label, pair_id, prompts, lm, L, H, V, n_pairs``
  * ``topk_cache_path``    — schema-compatible with cache_hallueval_topk.py:
        ``topk_logp, topk_idx, E_logit, E_marg, DeltaE, lm, L, K, N``

The DirichletFM auditor (``HalluevalDFMDataModule``) consumes both files,
so producing them together avoids running the LM twice.

Supports fp16 / bf16 / 4-bit (bitsandbytes-nf4) precision so Llama-7B fits
on a 20 GB GPU. The 4-bit path needs ``bitsandbytes`` installed.

Usage::

    # Llama-3.2-1B at bf16 (fits comfortably on 20 GB):
    python scripts/cache_hallueval_llama.py \\
        --lm meta-llama/Llama-3.2-1B \\
        --precision bf16 --max-n 10000 --L 256 --K 32 \\
        --hidden-out data/hallueval_cache_llama1b.pt \\
        --topk-out   data/hallueval_topk_llama1b.pt

    # Llama-2-7B at 4-bit nf4 (also fits, ~6 GB):
    python scripts/cache_hallueval_llama.py \\
        --lm meta-llama/Llama-2-7b-hf \\
        --precision 4bit --max-n 10000 --L 256 --K 32 --lm-batch-size 4 \\
        --hidden-out data/hallueval_cache_llama7b.pt \\
        --topk-out   data/hallueval_topk_llama7b.pt
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from aitchinson_flow.data.hallueval import (  # noqa: E402
    load_hallueval_qa,
    build_prompts,
)
from aitchinson_flow.data.wiki import _spilled_energy_per_pos  # noqa: E402


def _tokenize_pair(tokenizer, prefix: str, full: str, *, L: int, pad_id: int):
    prefix_ids = tokenizer.encode(prefix)
    full_ids = tokenizer.encode(full)
    if len(full_ids) > L:
        full_ids = full_ids[:L]
    n_full = len(full_ids)
    n_prefix = len(prefix_ids)
    ids = list(full_ids) + [pad_id] * (L - n_full)
    attn = [True] * n_full + [False] * (L - n_full)
    answer_mask = [(i >= n_prefix and i < n_full) for i in range(L)]
    return ids, attn, answer_mask


def _load_lm(name: str, precision: str):
    """Load any HF causal LM under the requested precision.

    Returns (model, tokenizer, device, dtype, V, H).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if precision == "fp16":
        dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        model.to(device)
    elif precision == "bf16":
        dtype = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        model.to(device)
    elif precision == "fp32":
        dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        model.to(device)
    elif precision == "4bit":
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "4bit precision requires bitsandbytes. Install with `pip install bitsandbytes`."
            ) from e
        dtype = torch.bfloat16
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            name, quantization_config=bnb, device_map="cuda"
        )
    else:
        raise ValueError(f"Unknown precision: {precision}")

    model.eval()
    V = model.config.vocab_size
    H = model.config.hidden_size
    return model, tokenizer, device, dtype, V, H


@torch.no_grad()
def _forward_pass(
    model,
    ids: torch.Tensor,
    attn: torch.Tensor,
    *,
    K: int,
    batch_size: int,
    H: int,
    L: int,
    fp16_storage: bool = True,
):
    """Single LM pass producing every cached field.

    Crucial memory discipline: never keep ``(B, L, V)`` logits across the
    batch boundary. We compute the per-position spilled-energy variants
    AND the top-K log-probs *inside* the batch loop and discard logits
    before the next batch. Hidden states are stored fp16 (default) to keep
    the cache file under ~30 GB even at Llama-7B's H=4096.
    """
    device = next(model.parameters()).device
    n = ids.shape[0]

    storage_dtype = torch.float16 if fp16_storage else torch.float32

    out_hidden = torch.empty(n, L, H, dtype=storage_dtype)
    out_se = torch.zeros(n, L, dtype=torch.float32)
    out_topk_lp = torch.empty(n, L, K, dtype=torch.float32)
    out_topk_id = torch.empty(n, L, K, dtype=torch.long)
    out_eL = torch.zeros(n, L, dtype=torch.float32)
    out_em = torch.zeros(n, L, dtype=torch.float32)

    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        ids_b = ids[s:e].to(device)
        attn_b = attn[s:e].to(device)
        out = model(ids_b, attention_mask=attn_b, output_hidden_states=True)
        logits = out.logits.float()  # (B, L, V)
        hidden = out.hidden_states[-1]  # (B, L, H)

        # Hidden states (fp16 storage to fit on disk for big LMs)
        out_hidden[s:e] = hidden.detach().to(dtype=storage_dtype, device="cpu")

        # Spilled energy (per-position NLL of the actual token under LM)
        out_se[s:e] = _spilled_energy_per_pos(logits, ids_b).cpu().float()

        # Top-K log-probs and indices
        log_p = torch.log_softmax(logits, dim=-1)
        topk_lp, topk_id = log_p.topk(K, dim=-1)  # (B, L, K)
        out_topk_lp[s:e] = topk_lp.detach().cpu().float()
        out_topk_id[s:e] = topk_id.detach().cpu().long()

        # Paper E_logit / E_marg / ΔE — same conventions as
        # cache_hallueval_topk.py:
        #   E_logit[r, i] = -logits[r, i-1, full_ids[r, i]]
        #   E_marg[r, i]  = -logsumexp(logits[r, i, :])
        # Position 0 has no predecessor → E_logit[:, 0] = 0.
        if L >= 2:
            picked = logits[:, : L - 1, :].gather(
                -1, ids_b[:, 1:].unsqueeze(-1)
            ).squeeze(-1)  # (B, L-1)
            eL = torch.zeros(ids_b.shape[0], L, dtype=torch.float32)
            eL[:, 1:] = -picked.detach().cpu().float()
            out_eL[s:e] = eL
        out_em[s:e] = -torch.logsumexp(logits, dim=-1).detach().cpu().float()

        del out, logits, hidden, log_p, topk_lp, topk_id
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return out_hidden, out_se, out_topk_lp, out_topk_id, out_eL, out_em


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lm", required=True,
                   help="HF model id, e.g. meta-llama/Llama-3.2-1B")
    p.add_argument(
        "--precision",
        default="bf16",
        choices=("fp32", "fp16", "bf16", "4bit"),
        help="bf16 default; 4bit needs bitsandbytes",
    )
    p.add_argument("--max-n", type=int, default=10000,
                   help="number of HaluEval-QA pairs (writes 2*max-n rows)")
    p.add_argument("--L", type=int, default=256, help="padded sequence length")
    p.add_argument("--K", type=int, default=32, help="top-K vocab slots per pos")
    p.add_argument("--include-knowledge", action="store_true",
                   help="prepend the dataset's 'knowledge' field to the prompt")
    p.add_argument("--lm-batch-size", type=int, default=8)
    p.add_argument("--hidden-out", required=True,
                   help="output path for the hidden_states cache")
    p.add_argument("--topk-out", required=True,
                   help="output path for the top-K + paper-ΔE cache")
    p.add_argument("--no-fp16-hidden", action="store_true",
                   help="store hidden states as fp32 instead of fp16 (4× disk)")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    print(
        f"[cache] lm={args.lm}  precision={args.precision}  "
        f"max_n={args.max_n}  L={args.L}  K={args.K}  "
        f"batch={args.lm_batch_size}"
    )

    print("[cache] loading HaluEval-QA …")
    rows = load_hallueval_qa(max_n=args.max_n)
    print(f"  rows: {len(rows)}")

    model, tokenizer, device, dtype, V, H = _load_lm(args.lm, args.precision)
    print(f"  V={V}  H={H}  device={device}  dtype={dtype}")

    clean_prompts, invalid_prompts = build_prompts(
        rows, include_knowledge=args.include_knowledge
    )
    prefixes = []
    for r in rows:
        prefix = ""
        if args.include_knowledge and r.get("knowledge"):
            prefix = f"Knowledge: {r['knowledge']}\n"
        prefixes.append(f"{prefix}Question: {r['question']}\nAnswer: ")

    print("[cache] tokenising …")
    pad_id = tokenizer.pad_token_id
    ids_list, attn_list, ansmask_list = [], [], []
    label_list, pair_id_list, flat_prompts = [], [], []
    for k, (prefix, c, i) in enumerate(zip(prefixes, clean_prompts, invalid_prompts)):
        ic, ac, mc = _tokenize_pair(tokenizer, prefix, c, L=args.L, pad_id=pad_id)
        ii, ai, mi = _tokenize_pair(tokenizer, prefix, i, L=args.L, pad_id=pad_id)
        ids_list.append(ic); attn_list.append(ac); ansmask_list.append(mc)
        label_list.append(False); pair_id_list.append(k); flat_prompts.append(c)
        ids_list.append(ii); attn_list.append(ai); ansmask_list.append(mi)
        label_list.append(True); pair_id_list.append(k); flat_prompts.append(i)

    full_ids = torch.tensor(ids_list, dtype=torch.long)
    attn_mask = torch.tensor(attn_list, dtype=torch.bool)
    answer_mask = torch.tensor(ansmask_list, dtype=torch.bool)
    label = torch.tensor(label_list, dtype=torch.bool)
    pair_id = torch.tensor(pair_id_list, dtype=torch.long)
    print(f"  full_ids: {tuple(full_ids.shape)}")
    print(
        f"  answer-span tokens / row "
        f"(mean): {answer_mask.float().sum(-1).mean().item():.1f}"
    )

    print(f"[cache] LM forward on {full_ids.shape[0]} sequences …")
    hidden, se_pos, topk_lp, topk_id, eL, em = _forward_pass(
        model,
        full_ids,
        attn_mask,
        K=args.K,
        batch_size=args.lm_batch_size,
        H=H,
        L=args.L,
        fp16_storage=not args.no_fp16_hidden,
    )
    se_pos = se_pos * attn_mask.float()
    print(f"  hidden_states: {tuple(hidden.shape)}  dtype={hidden.dtype}")
    print(f"  topk_logp: {tuple(topk_lp.shape)}")

    # Free LM aggressively before writing big tensors to disk.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Hidden cache (HalluevalDFMDataset reads `hidden_states`, `full_ids`,
    # `answer_mask`, `label`, `pair_id`, `n_pairs`, `L`, `H`).
    hidden_cache: dict[str, Any] = {
        "lm": args.lm,
        "n_pairs": len(rows),
        "L": args.L,
        "H": H,
        "V": V,
        "include_knowledge": args.include_knowledge,
        "full_ids": full_ids,
        "attn_mask": attn_mask,
        "hidden_states": hidden,
        "SE_pos": se_pos,
        "answer_mask": answer_mask,
        "label": label,
        "pair_id": pair_id,
        "prompts": flat_prompts,
    }
    out_h = Path(args.hidden_out)
    out_h.parent.mkdir(parents=True, exist_ok=True)
    torch.save(hidden_cache, out_h)
    print(
        f"[cache] wrote {out_h}  ({out_h.stat().st_size / 1e9:.2f} GB)"
    )

    # Topk cache (HalluevalDFMDataset reads `topk_logp`, `topk_idx`,
    # `E_logit`, `E_marg`, `DeltaE`, `K`, `L`, `N`).
    delta_e = eL - em
    topk_cache: dict[str, Any] = {
        "lm": args.lm,
        "K": args.K,
        "L": args.L,
        "N": full_ids.shape[0],
        "topk_logp": topk_lp,
        "topk_idx": topk_id,
        "E_logit": eL,
        "E_marg": em,
        "DeltaE": delta_e,
    }
    out_t = Path(args.topk_out)
    out_t.parent.mkdir(parents=True, exist_ok=True)
    torch.save(topk_cache, out_t)
    print(
        f"[cache] wrote {out_t}  ({out_t.stat().st_size / 1e6:.1f} MB)"
    )

    # Sanity: sign of mean SE on answer span (clean vs halluc).
    is_clean = ~label
    se_clean = (se_pos[is_clean] * answer_mask[is_clean].float()).sum(-1) / (
        answer_mask[is_clean].float().sum(-1).clamp_min(1)
    )
    se_inv = (se_pos[~is_clean] * answer_mask[~is_clean].float()).sum(-1) / (
        answer_mask[~is_clean].float().sum(-1).clamp_min(1)
    )
    print(
        f"[cache] sanity: mean SE per answer-token "
        f"clean={se_clean.mean().item():.3f}  "
        f"halluc={se_inv.mean().item():.3f}  "
        f"(positive sign = halluc is more surprising than clean)"
    )

    print(json.dumps({
        "lm": args.lm, "precision": args.precision, "n_pairs": len(rows),
        "L": args.L, "K": args.K, "H": H, "V": V,
        "hidden_cache": str(out_h),
        "topk_cache":   str(out_t),
        "se_clean": float(se_clean.mean()),
        "se_halluc": float(se_inv.mean()),
    }))


if __name__ == "__main__":
    main()

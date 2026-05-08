"""One-time HaluEval-QA feature cache for UQ.

Runs the LM (default GPT-2) over (question, right_answer) and
(question, hallucinated_answer) pairs and saves
``data/hallueval_cache_<lm>.pt`` for the eval script
``scripts/eval_uq.py`` to consume.

Usage:
    python scripts/cache_hallueval.py --lm gpt2 --max-n 10000 --L 256 \\
        --out data/hallueval_cache_gpt2.pt --lm-batch-size 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from aitchinson_flow.data.hallueval import (  # noqa: E402
    load_hallueval_qa, build_prompts,
)
from aitchinson_flow.data.wiki import _spilled_energy_per_pos  # noqa: E402


def _tokenize_pair(
    tokenizer,
    prompt_prefix: str,
    full_text: str,
    *,
    L: int,
    pad_token_id: int,
) -> tuple[list[int], list[bool], list[bool]]:
    """Tokenize (prompt_prefix, full_text) and return:

    * ``ids``         : padded BPE token ids of length L
    * ``attn``        : True where token is real (not pad), length L
    * ``answer_mask`` : True at positions *inside the answer span*, length L

    The answer span is everything after ``prompt_prefix`` in ``full_text``.
    Positions are clamped to ``L``; if the full text is longer than L,
    we truncate from the right (preserving the question).
    """
    prefix_ids = tokenizer.encode(prompt_prefix)
    full_ids = tokenizer.encode(full_text)
    n_full = len(full_ids)
    if n_full > L:
        full_ids = full_ids[:L]
        n_full = L
    n_prefix = len(prefix_ids)
    # answer-mask positions: those with index >= len(prefix_ids) in the
    # original tokenisation (we trust that the prefix is a token-level
    # prefix of the full sequence — true for GPT-2 BPE).
    ids = list(full_ids) + [pad_token_id] * (L - n_full)
    attn = [True] * n_full + [False] * (L - n_full)
    answer_mask = [(i >= n_prefix and i < n_full) for i in range(L)]
    return ids, attn, answer_mask


@torch.no_grad()
def _lm_forward_batch(
    model, ids: torch.Tensor, attn: torch.Tensor, *, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the LM in batches; return (logits, last-hidden) on CPU."""
    device = next(model.parameters()).device
    n = ids.shape[0]
    out_logits: list[torch.Tensor] = []
    out_hidden: list[torch.Tensor] = []
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        ids_b = ids[s:e].to(device)
        attn_b = attn[s:e].to(device)
        out = model(ids_b, attention_mask=attn_b, output_hidden_states=True)
        out_logits.append(out.logits.cpu().float())
        out_hidden.append(out.hidden_states[-1].cpu().float())
    return torch.cat(out_logits, dim=0), torch.cat(out_hidden, dim=0)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lm", default="gpt2",
                   choices=["gpt2", "gpt2-medium", "gpt2-large", "Qwen/Qwen2.5-1.5B"])
    p.add_argument("--max-n", type=int, default=10000)
    p.add_argument("--L", type=int, default=256)
    p.add_argument("--include-knowledge", action="store_true",
                   help="prepend the dataset's 'knowledge' field to the prompt")
    p.add_argument("--lm-batch-size", type=int, default=8)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args(argv)

    print(f"[cache] lm={args.lm}, max_n={args.max_n}, L={args.L}, "
          f"include_knowledge={args.include_knowledge}")

    print("[cache] loading HaluEval-QA …")
    rows = load_hallueval_qa(max_n=args.max_n)
    print(f"  rows: {len(rows)}")
    sample = rows[0]
    print(f"  sample question: {sample['question'][:120]!r}")
    print(f"  sample right:    {sample['right_answer'][:120]!r}")
    print(f"  sample halluc:   {sample['hallucinated_answer'][:120]!r}")

    print(f"[cache] loading {args.lm} …")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.lm)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    dtype = torch.float16 if "Qwen" in args.lm else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.lm, dtype=dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    V = model.config.vocab_size
    H = model.config.hidden_size
    print(f"  V={V}  H={H}  dtype={dtype}")

    # Build prompts.
    clean_prompts, invalid_prompts = build_prompts(
        rows, include_knowledge=args.include_knowledge
    )

    # The shared prefix per row, used to compute answer_mask.
    prefixes = []
    for r in rows:
        prefix = ""
        if args.include_knowledge and r["knowledge"]:
            prefix = f"Knowledge: {r['knowledge']}\n"
        prefixes.append(f"{prefix}Question: {r['question']}\nAnswer: ")

    print("[cache] tokenising and padding …")
    ids_list, attn_list, ansmask_list = [], [], []
    label_list, pair_id_list = [], []
    flat_prompts = []
    for k, (prefix, c, i) in enumerate(zip(prefixes, clean_prompts, invalid_prompts)):
        ids_c, attn_c, am_c = _tokenize_pair(tokenizer, prefix, c,
                                              L=args.L, pad_token_id=pad_id)
        ids_i, attn_i, am_i = _tokenize_pair(tokenizer, prefix, i,
                                              L=args.L, pad_token_id=pad_id)
        ids_list.append(ids_c); attn_list.append(attn_c); ansmask_list.append(am_c)
        label_list.append(False); pair_id_list.append(k); flat_prompts.append(c)
        ids_list.append(ids_i); attn_list.append(attn_i); ansmask_list.append(am_i)
        label_list.append(True); pair_id_list.append(k); flat_prompts.append(i)

    full_ids = torch.tensor(ids_list, dtype=torch.long)         # (2n, L)
    attn_mask = torch.tensor(attn_list, dtype=torch.bool)       # (2n, L)
    answer_mask = torch.tensor(ansmask_list, dtype=torch.bool)  # (2n, L)
    label = torch.tensor(label_list, dtype=torch.bool)          # (2n,)
    pair_id = torch.tensor(pair_id_list, dtype=torch.long)      # (2n,)
    print(f"  full_ids: {tuple(full_ids.shape)}")
    print(f"  answer-span tokens (mean per row): {answer_mask.float().sum(dim=-1).mean().item():.1f}")

    print(f"[cache] LM forward on {full_ids.shape[0]} sequences …")
    logits, hidden_states = _lm_forward_batch(
        model, full_ids, attn_mask, batch_size=args.lm_batch_size
    )
    print(f"  logits: {tuple(logits.shape)}  hidden_states: {tuple(hidden_states.shape)}")

    print("[cache] computing AR-shifted Spilled Energy …")
    SE_pos = _spilled_energy_per_pos(logits, full_ids)  # (2n, L)
    # Zero out SE at pad positions (no signal there).
    SE_pos = SE_pos * attn_mask.float()

    cache: dict[str, Any] = {
        "lm": args.lm,
        "n_pairs": len(rows),
        "L": args.L,
        "H": H,
        "V": V,
        "include_knowledge": args.include_knowledge,
        "full_ids": full_ids,
        "attn_mask": attn_mask,
        "hidden_states": hidden_states,
        "SE_pos": SE_pos,
        "answer_mask": answer_mask,
        "label": label,
        "pair_id": pair_id,
        "prompts": flat_prompts,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out_path)
    print(f"[cache] wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Sanity: SE on hallucinated tokens (within answer-span) should be
    # higher than on clean tokens, on average.
    is_clean = ~label                                       # (2n,)
    se_clean = SE_pos[is_clean] * answer_mask[is_clean].float()
    se_inv = SE_pos[~is_clean] * answer_mask[~is_clean].float()
    se_clean_mean = (
        se_clean.sum(dim=-1) / answer_mask[is_clean].float().sum(dim=-1).clamp(min=1)
    ).mean().item()
    se_inv_mean = (
        se_inv.sum(dim=-1) / answer_mask[~is_clean].float().sum(dim=-1).clamp(min=1)
    ).mean().item()
    print(f"[cache] sanity: mean SE per answer-token "
          f"clean={se_clean_mean:.3f} hallucinated={se_inv_mean:.3f} "
          f"(higher on hallucinated = expected sign)")

    print(json.dumps({
        "lm": args.lm, "n_pairs": len(rows), "L": args.L, "H": H, "V": V,
        "out": str(out_path),
        "se_clean_per_answer_token": se_clean_mean,
        "se_invalid_per_answer_token": se_inv_mean,
    }))


if __name__ == "__main__":
    main()

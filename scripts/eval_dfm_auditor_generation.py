"""Phase-H-style text-quality eval for the DirichletFM auditor.

Conditional generation: pick held-out clean rows, fix their h_LLM context,
integrate the marginal vector field from x ~ Dir(1) at t=1 to t=t_max,
take argmax slot per position, decode through the row's `topk_idx` table
back to GPT-2 vocab IDs, then score the resulting sequence under GPT-2.

Phase H (TRAINING_PROTOCOL.md §6) thresholds:
  * F3 PASS    : NLL ≤ 5.5
  * PARTIAL    : NLL ∈ [5.5, 7.0]
  * FAIL       : NLL > 7.0   (EqM landed at 8.89 here)
  * Random-slot baseline reference: 9.03
  * Clean reference: 4.07

Outputs ``runs/<out>/genH.json`` + a small text grid printed to stdout.
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

import aitchinson_flow.models  # noqa: E402,F401
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.hallueval_dfm import HalluevalDFMDataModule  # noqa: E402
from aitchinson_flow.models.factory import build_model  # noqa: E402


@torch.no_grad()
def score_under_gpt2(
    lm: torch.nn.Module,
    token_ids: torch.Tensor,
    answer_mask: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 4,
) -> torch.Tensor:
    """Mean per-token NLL under a pre-loaded GPT-2 over answer-mask positions.

    token_ids:   (B, L) long
    answer_mask: (B, L) bool
    Returns (B,) per-row mean NLL.

    Batched to keep peak memory reasonable on small GPUs.
    """
    nll_out: list[torch.Tensor] = []
    for s in range(0, token_ids.shape[0], batch_size):
        e = min(s + batch_size, token_ids.shape[0])
        ids = token_ids[s:e].to(device)
        out = lm(ids)
        logits = out.logits
        L = ids.shape[1]
        pred_logits = logits[:, : L - 1, :]
        target_ids = ids[:, 1:].unsqueeze(-1)
        lse = torch.logsumexp(pred_logits, dim=-1)
        picked = pred_logits.gather(-1, target_ids).squeeze(-1)
        nll_pos = lse - picked  # (b, L-1)
        mask = answer_mask[s:e, 1:].to(device)
        denom = mask.sum(dim=-1).clamp_min(1).to(nll_pos.dtype)
        nll = (nll_pos * mask).sum(dim=-1) / denom
        nll_out.append(nll.cpu())
    return torch.cat(nll_out, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="path to epoch_final.pt or best.pt")
    parser.add_argument("--out", default=None, help="dir for results (defaults to ckpt parent)")
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--nfe", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else Path(args.ckpt).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg: Config = ckpt["cfg"]
    K = ckpt.get("K", 32)
    L = ckpt.get("L", 160)

    cfg.hallueval_dfm_auditor = replace(
        cfg.hallueval_dfm_auditor,
        enabled=True,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
    )
    dm = HalluevalDFMDataModule(cfg.hallueval_dfm_auditor)

    model = build_model(cfg)
    model.set_dims(K=K, L=L)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    val_loader = dm.val_dataloader()
    if val_loader is None:
        raise RuntimeError("val_loader is None")

    # Pull the first n_samples held-out CLEAN rows.
    h_ctx_buf, mask_buf, topk_idx_buf, full_ids_buf = [], [], [], []
    n_taken = 0
    for batch in val_loader:
        keep = ~batch["label"].bool()
        if not keep.any():
            continue
        for k in ("hidden", "answer_mask", "topk_idx", "full_ids"):
            batch[k] = batch[k][keep]
        room = args.n_samples - n_taken
        h_ctx_buf.append(batch["hidden"][:room].float())
        mask_buf.append(batch["answer_mask"][:room].bool())
        topk_idx_buf.append(batch["topk_idx"][:room].long())
        full_ids_buf.append(batch["full_ids"][:room].long())
        n_taken += min(room, batch["hidden"].shape[0])
        if n_taken >= args.n_samples:
            break

    h_ctx = torch.cat(h_ctx_buf, 0)[: args.n_samples]
    answer_mask = torch.cat(mask_buf, 0)[: args.n_samples]
    topk_idx = torch.cat(topk_idx_buf, 0)[: args.n_samples]
    full_ids = torch.cat(full_ids_buf, 0)[: args.n_samples]
    print(f"[genH] held-out clean rows: {h_ctx.shape[0]}, L={h_ctx.shape[1]}")

    # Conditional sampling
    print(f"[genH] running conditional sampler nfe={args.nfe} ...")
    slot_ids = model.sample_conditional(h_ctx.to(device), nfe=args.nfe)
    slot_ids = slot_ids.cpu()  # (B, L) — slot in topk

    # Map slot → vocab id using each row's topk_idx table
    sampled_ids = topk_idx.gather(-1, slot_ids.unsqueeze(-1)).squeeze(-1)
    # Replace position 0 with the original prompt's token (the LM has nothing
    # to predict at position 0; copying the actual first token gives a fair
    # starting context for the GPT-2 NLL forward — same as Phase H).
    sampled_ids[:, 0] = full_ids[:, 0]

    # Random-slot baseline: pick a uniform random slot at each position
    rand_slot = torch.randint(0, K, slot_ids.shape)
    rand_ids = topk_idx.gather(-1, rand_slot.unsqueeze(-1)).squeeze(-1)
    rand_ids[:, 0] = full_ids[:, 0]

    # Score under GPT-2 (loaded once)
    print("[genH] loading GPT-2 ...")
    from transformers import AutoModelForCausalLM
    lm = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    lm.to(device).eval()
    print("[genH] scoring under GPT-2 ...")
    nll_sample = score_under_gpt2(lm, sampled_ids, answer_mask, device=device)
    nll_random = score_under_gpt2(lm, rand_ids, answer_mask, device=device)
    nll_clean = score_under_gpt2(lm, full_ids, answer_mask, device=device)
    del lm
    torch.cuda.empty_cache()

    # Vocab match to the actual clean token
    vocab_match_sample = (
        ((sampled_ids == full_ids) & answer_mask).sum().float()
        / answer_mask.sum().clamp_min(1).float()
    ).item()
    vocab_match_random = (
        ((rand_ids == full_ids) & answer_mask).sum().float()
        / answer_mask.sum().clamp_min(1).float()
    ).item()
    # Slot-level diversity across rows: hamming(sample_i, sample_j)
    n = slot_ids.shape[0]
    div = []
    for i in range(min(n, 16)):
        for j in range(i + 1, min(n, 16)):
            d = (slot_ids[i] != slot_ids[j]).float().mean().item()
            div.append(d)
    diversity = float(sum(div) / max(len(div), 1))

    summary = {
        "n_samples": int(h_ctx.shape[0]),
        "nfe": args.nfe,
        "mean_nll": {
            "generated": float(nll_sample.mean()),
            "random_slot": float(nll_random.mean()),
            "clean": float(nll_clean.mean()),
        },
        "vocab_match": {
            "generated": vocab_match_sample,
            "random_slot": vocab_match_random,
        },
        "slot_diversity_pairwise_hamming": diversity,
        "phaseH_threshold": {"PASS_le": 5.5, "PARTIAL_le": 7.0, "FAIL_gt": 7.0},
        "ckpt": str(args.ckpt),
    }
    with open(out_dir / "genH.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2))

    # Decode a few sample texts for the grid
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    print("\n[genH] sample texts (first 3, answer-span only):")
    for i in range(min(3, n_taken)):
        m = answer_mask[i]
        # GPT-2 answer-span tokens (clean, sampled, random)
        clean_ids_ans = full_ids[i][m].tolist()
        gen_ids_ans = sampled_ids[i][m].tolist()
        rand_ids_ans = rand_ids[i][m].tolist()
        print(f"  --- row {i} (answer span has {int(m.sum())} tokens) ---")
        print(f"    CLEAN  : {tok.decode(clean_ids_ans)!r}")
        print(f"    GEN    : {tok.decode(gen_ids_ans)!r}")
        print(f"    RANDOM : {tok.decode(rand_ids_ans)!r}")


if __name__ == "__main__":
    main()

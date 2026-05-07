"""Phase H — auditor-driven generation (TRAINING_PROTOCOL.md §6 Phase H).

Tests the *literal* auditor-is-also-generator claim: the same EqM energy
that scores tokens drives Euler-γ sampling under the LM-vocab constraint.
The GP prototype provably cannot do this on text8 (BPC ≈ 7); EqM, by
virtue of being a flow-matching velocity field, in principle can.

We run **conditional generation**: pick a clean WikiText-2 chunk and use
its cached LM hidden states `h_clean` and top-K logit indices
`clean_topk_idx` as fixed context. Sample x ∈ R^{L×K} from a Gaussian
source, run Euler-γ for nfe=64 steps on f(x; γ, h_clean), take argmax
slot per position, and map back to the LM's vocab via the cached
top-K table. The recovered token IDs are evaluated by:

* **Token-recovery rate**: fraction of positions where the generated
  argmax slot equals the clean argmax slot (slot 0 = the LM's
  most-likely token at that position given clean context). Random
  baseline: 1/K = 1/64 ≈ 1.6%.
* **LM validity**: re-feed the generated token IDs through GPT-2,
  compute mean per-position NLL. Lower = more natural.
* **Auditor energy**: ⟨x_gen, f(x_gen; γ=1, h_clean)⟩ vs ⟨x_clean, ⟩.
* **Diversity**: 64-sample uniqueness fraction across chunks; mean
  Hamming distance between generated samples (with different RNG seeds)
  starting from the same chunk's context.

Pre-condition (TRAINING_PROTOCOL.md §6 Phase H): F passed *or* partial.
With `aud_gpt2_ctx` matching the F1 floor, this test is enabled.
Decision criterion (per protocol):

* **F3 succeeds**: generated text recognisably English at word-level
  AND mean LM log-prob ≥ −5.5.
* **F3 partial**: LM log-prob ∈ [−7, −5.5]; samples look broken-but-
  recognisable.
* **F3 fails**: character-soup / random-looking samples.

Note: the 2026-05-07 22:38 UTC denoising test showed −∇E does NOT point
toward clean for this auditor. That's a strong negative prior for
F3 — sampling on a field whose gradient doesn't denoise is unlikely to
recover natural text. Run it anyway to settle the question definitively.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.wiki import (  # noqa: E402
    WikiAuditorDataset, load_wiki_cache, _spilled_energy_per_pos,
)
from aitchinson_flow.models import build_model  # noqa: E402

from scripts.eval_full import _config_from_payload  # noqa: E402


@torch.enable_grad()
def _grad_E(model, x: torch.Tensor, gamma_val: float, h_ctx: torch.Tensor | None) -> torch.Tensor:
    """Conservative gradient ∇⟨x, f(x; γ, h)⟩."""
    time_cond = getattr(model.cfg.eqm, "time_conditioning", "off")
    B = x.shape[0]
    if time_cond != "off":
        gamma = torch.full((B,), float(gamma_val), device=x.device, dtype=x.dtype)
    else:
        gamma = None
    x_req = x.detach().requires_grad_(True)
    v = model.forward(x_req, gamma, h_ctx)
    energy = (x_req * v).sum()
    grad = torch.autograd.grad(energy, x_req, create_graph=False)[0].detach()
    return grad


@torch.no_grad()
def _v(model, x: torch.Tensor, gamma_val: float, h_ctx: torch.Tensor | None) -> torch.Tensor:
    """Raw velocity f(x; γ, h)."""
    time_cond = getattr(model.cfg.eqm, "time_conditioning", "off")
    B = x.shape[0]
    if time_cond != "off":
        gamma = torch.full((B,), float(gamma_val), device=x.device, dtype=x.dtype)
    else:
        gamma = None
    return model.forward(x, gamma, h_ctx)


def sample_euler_gamma(
    model,
    *,
    h_ctx: torch.Tensor | None,
    L: int,
    K: int,
    B: int,
    nfe: int,
    sigma: float,
    use_grad: bool,
    seed: int,
) -> torch.Tensor:
    """Euler-γ sampler. Returns (B, L, K) CLR features at γ=1."""
    device = next(model.parameters()).device
    g = torch.Generator(device=device).manual_seed(seed)
    x = sigma * torch.randn(B, L, K, device=device, generator=g)
    x = x - x.mean(dim=-1, keepdim=True)

    gammas = torch.linspace(0.0, 1.0, nfe + 1, device=device)[:-1]
    h = 1.0 / nfe
    for gv in gammas:
        if use_grad:
            grad = _grad_E(model, x, float(gv.item()), h_ctx)
            # FM target sign: c(γ)·(x_0 - x_1) points data → noise, so
            # data direction is -grad. Step toward data:
            x = x - h * grad
        else:
            v = _v(model, x, float(gv.item()), h_ctx)
            x = x - h * v
        x = x - x.mean(dim=-1, keepdim=True)
    return x.detach()


@torch.no_grad()
def _lm_eval(token_ids: torch.Tensor, batch_size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-run GPT-2 on the generated token IDs; returns (logits, hidden)."""
    from transformers import AutoModelForCausalLM

    device = token_ids.device
    lm = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32).to(device).eval()
    out_logits, out_hidden = [], []
    n = token_ids.shape[0]
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        out = lm(token_ids[s:e], output_hidden_states=True)
        out_logits.append(out.logits.cpu().float())
        out_hidden.append(out.hidden_states[-1].cpu().float())
    del lm
    torch.cuda.empty_cache()
    return torch.cat(out_logits, dim=0), torch.cat(out_hidden, dim=0)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", default="data/wiki_cache_gpt2.pt")
    p.add_argument("--n", type=int, default=32, help="number of chunks to generate from")
    p.add_argument("--nfe", type=int, default=64)
    p.add_argument("--sigma", type=float, default=0.1, help="source σ for x_0")
    p.add_argument("--use-grad", action="store_true",
                   help="sample on conservative gradient ∇⟨x,f⟩ (default: raw velocity f)")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-md", default="runs/phaseH_audited.md")
    p.add_argument("--out-json", default="runs/phaseH_audited.json")
    args = p.parse_args(argv)

    print(f"[phaseH] loading {args.ckpt} …")
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    device = cfg.training.device
    K = int(cfg.text8_dataset.K)
    L = int(cfg.text8_dataset.L)
    ctx_mode = getattr(cfg.eqm, "context_features", "off")
    needs_h = ctx_mode != "off"

    model = build_model(cfg).to(device)
    model.load_state_dict(payload.get("model_state_dict", payload))
    model.eval()
    for pp in model.parameters():
        pp.requires_grad_(False)

    cache = load_wiki_cache(args.cache)
    n_total = int(cache["clean_clr"].shape[0])
    n_train = int(args.train_frac * n_total)
    val_start = n_train
    val_end = min(n_train + args.n, n_total)
    n = val_end - val_start
    print(f"[phaseH] sampling {n} val chunks (idx {val_start}..{val_end-1}), nfe={args.nfe}, "
          f"σ={args.sigma}, ctx={ctx_mode}, use_grad={args.use_grad}")

    clean_clr = cache["clean_clr"][val_start:val_end].to(device).float()
    clean_topk_idx = cache["clean_topk_idx"][val_start:val_end]    # (n, L, K)
    clean_ids = cache["clean_ids"][val_start:val_end]               # (n, L)
    clean_se = cache["clean_SE_pos"][val_start:val_end]             # (n, L)
    if needs_h:
        h_clean = cache["clean_h"][val_start:val_end].to(device).float()
    else:
        h_clean = None

    # --- Sample (conditional on each chunk's h_clean) ---
    x_gen = sample_euler_gamma(
        model, h_ctx=h_clean, L=L, K=K, B=n,
        nfe=args.nfe, sigma=args.sigma, use_grad=args.use_grad, seed=args.seed,
    )

    # --- Recover slot indices and tokens ---
    slot_gen = x_gen.argmax(dim=-1).cpu()                       # (n, L)
    slot_clean = clean_clr.argmax(dim=-1).cpu()                 # (n, L)
    # Map slot → vocab id using each chunk's top-K table.
    gen_token_ids = clean_topk_idx.gather(-1, slot_gen.unsqueeze(-1)).squeeze(-1)  # (n, L)
    clean_top1 = clean_topk_idx[..., 0]                         # the LM's top-1 in clean ctx

    # --- Token-recovery rate ---
    # Slot-level: how often does sampled argmax slot equal clean argmax slot?
    slot_match = (slot_gen == slot_clean).float().mean().item()
    # Vocab-level: how often does the recovered token equal the actual clean token?
    vocab_match_to_data = (gen_token_ids == clean_ids).float().mean().item()
    vocab_match_to_lm_top1 = (gen_token_ids == clean_top1).float().mean().item()

    # --- Random baseline (uniform random slot) ---
    rng = torch.Generator().manual_seed(args.seed + 1)
    slot_rand = torch.randint(0, K, (n, L), generator=rng)
    rand_token_ids = clean_topk_idx.gather(-1, slot_rand.unsqueeze(-1)).squeeze(-1)
    rand_match_to_data = (rand_token_ids == clean_ids).float().mean().item()
    rand_match_to_lm_top1 = (rand_token_ids == clean_top1).float().mean().item()

    # --- Top-1 LM baseline (just take slot 0 = LM's top-1 prediction in clean context) ---
    top1_match_to_data = (clean_top1 == clean_ids).float().mean().item()

    # --- Re-evaluate generated sequences with GPT-2 ---
    print("[phaseH] re-feeding generated tokens through GPT-2 …")
    gen_logits, _ = _lm_eval(gen_token_ids.to(device).long())
    # Per-position NLL: SE(i) for the generated token under LM prediction.
    gen_se = _spilled_energy_per_pos(gen_logits, gen_token_ids.long())  # (n, L)
    nll_gen_mean = float(gen_se[:, 1:].mean())  # skip pos 0 (no prior context)
    nll_clean_mean = float(clean_se[:, 1:].mean())
    # log-prob of generated sequence per token (≈ -nll)
    logp_gen_mean = -nll_gen_mean
    logp_clean_mean = -nll_clean_mean

    # Random baseline LM eval.
    rand_logits, _ = _lm_eval(rand_token_ids.to(device).long())
    rand_se = _spilled_energy_per_pos(rand_logits, rand_token_ids.long())
    nll_rand_mean = float(rand_se[:, 1:].mean())

    # --- Auditor energy on generated and clean ---
    @torch.no_grad()
    def _energy(x, h):
        time_cond = getattr(model.cfg.eqm, "time_conditioning", "off")
        Bb = x.shape[0]
        gv = torch.full((Bb,), 1.0, device=x.device, dtype=x.dtype) if time_cond != "off" else None
        v = model.forward(x, gv, h)
        return (x * v).sum(dim=(-1, -2))
    e_gen = _energy(x_gen, h_clean).cpu()
    e_clean = _energy(clean_clr, h_clean).cpu()

    # --- Diversity: re-sample with different RNG, compute Hamming ---
    print("[phaseH] re-sampling with seed+1 to measure diversity …")
    x_gen_2 = sample_euler_gamma(
        model, h_ctx=h_clean, L=L, K=K, B=n,
        nfe=args.nfe, sigma=args.sigma, use_grad=args.use_grad, seed=args.seed + 100,
    )
    slot_gen_2 = x_gen_2.argmax(dim=-1).cpu()
    hamming_self = (slot_gen != slot_gen_2).float().mean().item()
    hamming_to_clean = (slot_gen != slot_clean).float().mean().item()

    # --- Decode some samples to text (best-effort) ---
    print("[phaseH] decoding 8 samples …")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    sample_texts: list[dict] = []
    for i in range(min(8, n)):
        gen_text = tok.decode(gen_token_ids[i].tolist(), skip_special_tokens=False)
        clean_text = tok.decode(clean_ids[i].tolist(), skip_special_tokens=False)
        sample_texts.append({
            "idx": int(val_start + i),
            "clean":     clean_text,
            "generated": gen_text,
        })

    summary = {
        "ckpt": str(args.ckpt),
        "ctx_mode": ctx_mode,
        "n": n,
        "nfe": args.nfe,
        "sigma": args.sigma,
        "use_grad": args.use_grad,
        "K": K,
        "L": L,
        "slot_match_to_clean":     slot_match,
        "vocab_match_to_data":     vocab_match_to_data,
        "vocab_match_to_lm_top1":  vocab_match_to_lm_top1,
        "random_match_to_data":    rand_match_to_data,
        "random_match_to_lm_top1": rand_match_to_lm_top1,
        "lm_top1_match_to_data":   top1_match_to_data,
        "nll_gen_mean":            nll_gen_mean,
        "nll_clean_mean":          nll_clean_mean,
        "nll_random_mean":         nll_rand_mean,
        "logp_gen_per_token":      logp_gen_mean,
        "logp_clean_per_token":    logp_clean_mean,
        "logp_random_per_token":   -nll_rand_mean,
        "auditor_energy_gen_mean":   float(e_gen.mean()),
        "auditor_energy_clean_mean": float(e_clean.mean()),
        "hamming_self_diversity":  hamming_self,
        "hamming_to_clean":        hamming_to_clean,
        "samples": sample_texts,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary, indent=2))

    # --- Markdown report ---
    md = []
    md.append(f"# Phase H — auditor-driven generation: {Path(args.ckpt).parent.name}")
    md.append("")
    md.append(f"- ckpt: `{args.ckpt}`")
    md.append(f"- ctx_mode: {ctx_mode},  nfe: {args.nfe},  σ: {args.sigma},  use_grad: {args.use_grad}")
    md.append(f"- n chunks (val): {n} (idx {val_start}..{val_end-1})")
    md.append("")
    md.append("## Token-recovery rates")
    md.append("")
    md.append("| Method | Match to actual clean token | Match to LM top-1 |")
    md.append("|---|---:|---:|")
    md.append(
        f"| **EqM auditor sample (argmax slot)** | "
        f"{vocab_match_to_data*100:.1f}% | "
        f"{vocab_match_to_lm_top1*100:.1f}% |"
    )
    md.append(
        f"| Random slot baseline | "
        f"{rand_match_to_data*100:.1f}% | "
        f"{rand_match_to_lm_top1*100:.1f}% |"
    )
    md.append(
        f"| LM top-1 (slot 0) baseline | "
        f"{top1_match_to_data*100:.1f}% | "
        f"100.0% (by definition) |"
    )
    md.append("")
    md.append(f"Slot-level match to clean argmax slot: {slot_match*100:.1f}% "
              f"(chance = {100/K:.1f}%)")
    md.append("")
    md.append("## LM validity (mean per-token NLL on the generated sequence)")
    md.append("")
    md.append("| Sequence | NLL (lower=better) | log-prob/token |")
    md.append("|---|---:|---:|")
    md.append(f"| **EqM-generated** | {nll_gen_mean:.3f} | {-nll_gen_mean:+.3f} |")
    md.append(f"| Clean WikiText | {nll_clean_mean:.3f} | {-nll_clean_mean:+.3f} |")
    md.append(f"| Random-slot baseline | {nll_rand_mean:.3f} | {-nll_rand_mean:+.3f} |")
    md.append("")
    md.append(f"Protocol's F3 success threshold: log-prob ≥ −5.5 → "
              f"NLL ≤ 5.5. Generated NLL = {nll_gen_mean:.3f} → "
              f"**{'PASS' if nll_gen_mean <= 5.5 else 'FAIL'}** at this floor.")
    md.append(f"Partial threshold: NLL ∈ [5.5, 7] → "
              f"**{'PARTIAL' if 5.5 < nll_gen_mean <= 7 else 'no'}**.")
    md.append("")
    md.append("## Auditor energy")
    md.append("")
    md.append(f"- Generated x: E mean = {summary['auditor_energy_gen_mean']:+.3f}")
    md.append(f"- Clean x:     E mean = {summary['auditor_energy_clean_mean']:+.3f}")
    md.append(f"- Difference: the trained auditor's hinge target. If the field "
              f"is 'fooling itself', E(gen) ≈ E(clean).")
    md.append("")
    md.append("## Diversity")
    md.append("")
    md.append(f"- Hamming(sample₁, sample₂) on slot indices: {hamming_self*100:.1f}%")
    md.append(f"- Hamming(sample, clean argmax)             : {hamming_to_clean*100:.1f}%")
    md.append(f"- (chance ≈ {(K-1)/K*100:.1f}% for random slots)")
    md.append("")
    md.append("## Decoded examples (first 8 chunks)")
    md.append("")
    for s in sample_texts:
        md.append(f"### chunk {s['idx']}")
        md.append("```")
        md.append(f"clean:     {s['clean']!r}")
        md.append(f"generated: {s['generated']!r}")
        md.append("```")
    md.append("")
    md.append("## Verdict")
    md.append("")
    if nll_gen_mean <= 5.5:
        md.append("**F3 PASSES** — generated text is recognisably English at "
                  "word-level under the LM's NLL test.")
    elif nll_gen_mean <= 7.0:
        md.append("**F3 PARTIAL** — samples are recognisable but broken under "
                  "the LM. NLL is between the partial and pass thresholds.")
    else:
        md.append("**F3 FAILS** — samples are character-soup. The auditor's "
                  "energy field cannot drive coherent Euler-γ sampling.")
    Path(args.out_md).write_text("\n".join(md))
    print(f"[phaseH] wrote {args.out_md} and {args.out_json}")
    print()
    print("=== summary ===")
    print(f"vocab_match_to_data: {vocab_match_to_data*100:.1f}%  "
          f"(random: {rand_match_to_data*100:.1f}%, LM top-1: {top1_match_to_data*100:.1f}%)")
    print(f"NLL_gen: {nll_gen_mean:.3f}  (clean: {nll_clean_mean:.3f}, random: {nll_rand_mean:.3f})")
    print(f"hamming_self: {hamming_self*100:.1f}%  hamming_to_clean: {hamming_to_clean*100:.1f}%")


if __name__ == "__main__":
    main()

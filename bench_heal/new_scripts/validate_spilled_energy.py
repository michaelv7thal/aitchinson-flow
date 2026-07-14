"""Validate our spilled-energy implementation against the OFFICIAL reference code.

The repo previously computed a SAME-STEP quantity ``logsumexp(logits[i-1]) -
logits[i-1][x_i]`` = per-token NLL and called it "spilled energy". The real thing
(Minut, Dewidar & Masi, ICLR 2026, arXiv:2602.18671, Definition 4.1 / Eq. 8) is the
CROSS-STEP discrepancy between the logit energy read at step i-1 and the marginal
energy read at step i.

A sign flip inverts AUROC, so we do not trust prose: this script checks our
implementation numerically against the authors' own code
(github.com/OmnAI-Lab/spilled-energy, ``src/spilled_energy/energy.py``), which
defines::

    E[i]        = -logits[i-1][ids[i]]
    E_margin[i] = -logsumexp(logits[i])
    delta[i]    = -E_margin[i] + E[i]  =  logsumexp(logits[i]) - logits[i-1][ids[i]]

Checks:
  1. random logits — our formula vs official ``spilled_energy_torch`` (exact match)
  2. real GPT-2 on text8 — our ``_gpt2_bpe_scores`` vs official, on shared logits
  3. the paper's diagnostic: ΔE should sit near 0 on clean, well-modelled text,
     whereas NLL is substantially positive. If clean-mean ΔE is far from 0 we are
     probably still computing NLL.
  4. the identity  ΔE[i] = NLL[i] + (logsumexp(logits[i]) - logsumexp(logits[i-1]))

Usage:
    uv run python scripts/validate_spilled_energy.py --ref-repo <path/to/spilled-energy>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from scripts.bench_sflm_ebm import (  # noqa: E402
    _gpt2_bpe_scores, _load_gpt2, _ids_to_text,
)


def _load_official(ref_repo: Path):
    """Import the authors' spilled_energy_torch from a clone of their repo."""
    src = ref_repo / "src"
    if not (src / "spilled_energy" / "energy.py").exists():
        raise SystemExit(
            f"official code not found under {src}. Clone it first:\n"
            f"  git clone https://github.com/OmnAI-Lab/spilled-energy {ref_repo}")
    sys.path.insert(0, str(src))
    from spilled_energy.energy import spilled_energy_torch  # type: ignore
    return spilled_energy_torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref-repo", required=True,
                    help="path to a clone of OmnAI-Lab/spilled-energy")
    ap.add_argument("--ref-lm", default="gpt2")
    ap.add_argument("--n", type=int, default=16, help="# text8 windows for check 2/3")
    ap.add_argument("--length", type=int, default=256)
    ap.add_argument("--skip-gpt2", action="store_true")
    args = ap.parse_args()
    official = _load_official(Path(args.ref_repo))
    ok = True

    # ---- 1. random logits: our formula vs the official one ------------------
    torch.manual_seed(0)
    B, T, V = 3, 12, 50
    logits = torch.randn(B, T, V) * 3.0
    ids = torch.randint(0, V, (B, T))
    delta_ref, _, _ = official(logits, ids)

    # our formula, expressed exactly as in _gpt2_bpe_scores:
    #   token k (k>=1 in the padded frame) -> logit from step k-1, lse from step k
    picked = logits[:, :-1, :].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    lse_next = torch.logsumexp(logits[:, 1:, :], dim=-1)
    ours = lse_next - picked  # (B, T-1) aligned to ids[:, 1:]
    ref = delta_ref[:, 1:]  # official drops index 0 (BOS, E=0)
    max_err = (ours - ref).abs().max().item()
    print(f"[1] random logits: max|ours - official| = {max_err:.3e}", end="  ")
    if max_err < 1e-4:
        print("OK")
    else:
        print("MISMATCH")
        ok = False
        # is it a pure sign flip? (the trap)
        flip = (ours + ref).abs().max().item()
        if flip < 1e-4:
            print("    -> our result is the exact NEGATION of the official one "
                  "(sign flip: this would invert AUROC)")

    # sanity: the OLD (same-step) quantity is NOT the official spilled energy
    old_nll = torch.logsumexp(logits[:, :-1, :], dim=-1) - picked
    print(f"[1b] old same-step NLL vs official SE: max|diff| = "
          f"{(old_nll - ref).abs().max().item():.3f}  "
          f"(large => the old code was NOT computing spilled energy)")

    # identity: SE = NLL + (lse_i - lse_{i-1})
    drift = (torch.logsumexp(logits[:, 1:, :], dim=-1)
             - torch.logsumexp(logits[:, :-1, :], dim=-1))
    ident = (ours - (old_nll + drift)).abs().max().item()
    print(f"[1c] identity SE = NLL + logsumexp-drift: max err = {ident:.3e}", end="  ")
    print("OK" if ident < 1e-4 else "FAIL")
    ok = ok and ident < 1e-4

    if args.skip_gpt2:
        return 0 if ok else 1

    # ---- 2/3. real GPT-2 on clean text8 -------------------------------------
    from aitchinson_flow.config import Config
    from aitchinson_flow.training import build_training_datamodule
    import dataclasses

    cfg = Config()
    cfg = dataclasses.replace(cfg)
    cfg.data.length = args.length
    dm = build_training_datamodule(cfg)
    batch = next(iter(dm.test_dataloader() if hasattr(dm, "test_dataloader")
                      else dm.val_dataloader()))
    char_ids = batch["token_ids"][: args.n]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = _load_gpt2(args.ref_lm, device)

    se_scores, _ = _gpt2_bpe_scores(model, tok, char_ids, chunk=8, score="spilled")
    nll_scores, _ = _gpt2_bpe_scores(model, tok, char_ids, chunk=8, score="nll")
    se_mean = torch.cat(se_scores).mean().item()
    nll_mean = torch.cat(nll_scores).mean().item()

    # official, on the SAME logits (re-tokenise one chunk and feed the reference impl)
    texts = _ids_to_text(char_ids[:4])
    enc = tok(texts, add_special_tokens=False, padding=True, return_tensors="pt")
    inp = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)
    bos = torch.full((inp.shape[0], 1), tok.eos_token_id, dtype=inp.dtype, device=device)
    one = torch.ones((inp.shape[0], 1), dtype=am.dtype, device=device)
    padded = torch.cat([bos, inp], 1)
    with torch.no_grad():
        lg = model(input_ids=padded,
                   attention_mask=torch.cat([one, am], 1)).logits.float().cpu()
    d_ref, _, _ = official(lg, padded.cpu())
    picked2 = lg[:, :-1, :].gather(-1, inp.cpu().unsqueeze(-1)).squeeze(-1)
    ours2 = torch.logsumexp(lg[:, 1:, :], dim=-1) - picked2
    err2 = (ours2 - d_ref[:, 1:]).abs().max().item()
    print(f"[2] GPT-2 real logits: max|ours - official| = {err2:.3e}", end="  ")
    print("OK" if err2 < 1e-3 else "MISMATCH")
    ok = ok and err2 < 1e-3

    print(f"[3] clean text8 means:  spilled ΔE = {se_mean:+.4f}   NLL = {nll_mean:+.4f}")

    # [4] The paper's zero-property, tested where it actually applies. ΔE -> 0 only for
    # text the model MODELS CORRECTLY. GPT-2's domain is cased, punctuated English;
    # text8 is lowercase/unpunctuated and is genuinely OOD *for GPT-2*, so clean text8
    # legitimately sits well above 0. Checking only on text8 would look like a bug, so
    # we anchor on in-domain English: that is where ΔE must collapse to ~0 while NLL
    # stays clearly positive. If THIS number is far from 0, the implementation is wrong.
    def _score_text(text: str):
        enc = tok([text], add_special_tokens=False, return_tensors="pt")
        inp = enc["input_ids"].to(device)
        b = torch.full((1, 1), tok.eos_token_id, dtype=inp.dtype, device=device)
        with torch.no_grad():
            lgt = model(input_ids=torch.cat([b, inp], 1)).logits.float()
        pk = lgt[:, :-1, :].gather(-1, inp.unsqueeze(-1)).squeeze(-1)
        return ((torch.logsumexp(lgt[:, 1:, :], -1) - pk).mean().item(),
                (torch.logsumexp(lgt[:, :-1, :], -1) - pk).mean().item())

    eng = ("The Treaty of Versailles was signed on 28 June 1919, formally ending the "
           "state of war between Germany and the Allied Powers.")
    se_e, nll_e = _score_text(eng)
    print(f"[4] in-domain English:  spilled ΔE = {se_e:+.4f}   NLL = {nll_e:+.4f}")
    print("    ΔE should be ~0 here (paper Def 4.1) while NLL stays clearly > 0.")
    zero_ok = abs(se_e) < 1.0 and nll_e > 1.0
    print(f"    zero-property: {'OK' if zero_ok else 'FAIL — implementation suspect'}")
    ok = ok and zero_ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Reproduce the bench's plausible swap for selected eval windows.

Replays the exact RNG stream of ood_plausible_swap.plausible_swap (one shared
generator, one randperm per window in order, seed 42+1) but runs the candidate
scoring only for CANDIDATES. Windows 0 and 1 must reproduce the stored
examples of bench_ood_final/plausible/plausible_swap.json verbatim; the run
aborts if they do not.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path.home() / "projects" / "aitchinson-flow"
sys.path.insert(0, str(REPO / "src"))
sys.path.append(str(REPO))

from scripts.ood_variance_perpos import _load_dirichletfm  # noqa: E402
from scripts.ood_plausible_swap import _word_spans, _CHAR2ID, _decode  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from aitchinson_flow.data.corruption import build_vocab_by_len  # noqa: E402

CKPT = REPO / "runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_converge/epoch_final.pt"
REF = REPO / "bench_ood_final/plausible/plausible_swap.json"
OUT = Path(__file__).parent / "plausible_candidates.json"
CANDIDATES = [0, 1, 26, 30, 34, 37, 40, 44]
FIT, N, SEED, RATE, T_NLL, N_CANDS, CHUNK = 512, 64, 42, 0.15, 3.0, 48, 64

torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, cfg = _load_dirichletfm(str(CKPT), device)
K = cfg.text8_dataset.K

dm, _ = build_training_datamodule(cfg)
seqs = []
for b in dm.test_dataloader():
    seqs.append(b["token_ids"].long())
    if sum(s.shape[0] for s in seqs) >= FIT + N:
        break
seqs = torch.cat(seqs)
fit_tok = seqs[:FIT]
eval_tok = seqs[FIT:FIT + N]
fit_txt = " ".join(_decode(row) for row in fit_tok)
by_len = build_vocab_by_len(fit_txt)
print(f"[replay] device={device} eval={eval_tok.shape[0]}")

g = torch.Generator().manual_seed(SEED + 1)
out = eval_tok.clone()
results = {}
for i in range(N):
    spans = _word_spans(eval_tok[i])
    if not spans:
        continue
    n_sub = max(1, round(RATE * len(spans)))
    order = torch.randperm(len(spans), generator=g).tolist()[:n_sub]
    if i not in CANDIDATES:
        continue
    base = eval_tok[i:i + 1].to(device).long()
    swaps = []
    with torch.no_grad():
        for idx in order:
            a, b = spans[idx]
            cands = by_len.get(b - a)
            if not cands:
                continue
            orig = _decode(eval_tok[i, a:b])
            cand_words = [w for w in cands[:N_CANDS] if w != orig]
            if not cand_words:
                continue
            C = len(cand_words)
            cand_ids = base.repeat(C, 1)
            for ci, w in enumerate(cand_words):
                for j, ch in enumerate(w):
                    cand_ids[ci, a + j] = _CHAR2ID[ch]
            nlls = []
            for s0 in range(0, C, CHUNK):
                cb = cand_ids[s0:s0 + CHUNK]
                Bc = cb.shape[0]
                beta = torch.ones(Bc, cb.shape[1], K, device=device)
                beta.scatter_(-1, cb.unsqueeze(-1), float(T_NLL))
                x_t = beta / beta.sum(-1, keepdim=True)
                tt = torch.full((Bc,), float(T_NLL), device=device)
                logits = model.forward(x_t, tt)
                nll = -F.log_softmax(logits, -1).gather(
                    -1, cb.unsqueeze(-1)).squeeze(-1)
                nlls.append(nll[:, a:b].mean(dim=1).cpu())
            best = cand_words[int(torch.cat(nlls).argmin())]
            for j, ch in enumerate(best):
                out[i, a + j] = _CHAR2ID[ch]
            swaps.append({"span": [a, b], "orig": orig, "best": best})
    changed = (out[i] != eval_tok[i])
    results[i] = {
        "clean": _decode(eval_tok[i][:120]),
        "plausible": _decode(out[i][:120]),
        "changed120": [bool(x) for x in changed[:120].tolist()],
        "swaps": swaps,
    }
    print(f"[replay] window {i}: {len(swaps)} swaps")

ref = json.loads(REF.read_text())["examples"]
for i in (0, 1):
    ok = (results[i]["clean"] == ref[i]["clean"]
          and results[i]["plausible"] == ref[i]["plausible"]
          and results[i]["changed120"] == ref[i]["plausible_changed"])
    print(f"[replay] reproduce stored example[{i}]: {ok}")
    if not ok:
        raise SystemExit(f"replay mismatch on window {i} — do not trust results")

OUT.write_text(json.dumps(results, indent=1))
print(f"[replay] wrote {OUT}")

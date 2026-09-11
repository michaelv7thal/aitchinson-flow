"""Reproduce the healing bench's false-info demo corruption for all 64 windows.

Replays corrupt_false_info on the demo split exactly as scripts/heal_dirichlet.py
builds it for bench_heal_final/*_falseinfo.json: test loader, fit=256 / cal=256 /
demo=64 windows, vocab from the fit windows, corruption seed 42+100+0=142 (the
seed-0 realization the stored `examples` quote). Only the corruption is replayed;
no model forward pass is needed, so the checkpoint is read for its config alone.

Windows 0-3 must reproduce the stored `examples` of
bench_heal_final/nll_falseinfo.json verbatim (clean, corrupted, and the
true_corrupt mask); the run aborts without writing if they do not. Writes
clean/corrupted text plus the per-window word swaps to
bench_heal_final/falseinfo_demo_replay.json.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path.home() / "projects" / "aitchinson-flow"
sys.path.insert(0, str(REPO / "src"))
sys.path.append(str(REPO))

from scripts.eval_full import _config_from_payload  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from aitchinson_flow.data.corruption import (  # noqa: E402
    build_vocab_by_len,
    corrupt_false_info,
)

CKPT = REPO / "runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_converge/epoch_final.pt"
REF = REPO / "bench_heal_final/nll_falseinfo.json"
OUT = REPO / "bench_heal_final/falseinfo_demo_replay.json"
FIT, N_DEMO, SEED, RATE = 256, 64, 42, 0.15

_ALPH = "abcdefghijklmnopqrstuvwxyz "  # text8 K=27 decode (a-z + space)


def _decode(ids) -> str:
    return "".join(_ALPH[int(i)] for i in ids)


def main() -> int:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    payload = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    dm, _ = build_training_datamodule(cfg)

    need = 2 * FIT + N_DEMO
    seqs = []
    for b in dm.test_dataloader():
        seqs.append(b["token_ids"].long())
        if sum(s.shape[0] for s in seqs) >= need:
            break
    seqs = torch.cat(seqs)
    if seqs.shape[0] < need:
        raise SystemExit(f"test loader yielded {seqs.shape[0]} < {need} seqs")
    fit_tok = seqs[:FIT]
    demo = seqs[2 * FIT : 2 * FIT + N_DEMO]

    by_len = build_vocab_by_len(" ".join(_decode(r) for r in fit_tok))
    dc = corrupt_false_info(demo.clone(), RATE, by_len, seed=SEED + 100 + 0)

    ref = json.loads(REF.read_text())["examples"]
    for b in range(len(ref)):
        if _decode(demo[b]) != ref[b]["clean"]:
            raise SystemExit(f"ABORT: clean window {b} does not match stored example")
        if _decode(dc[b]) != ref[b]["corrupted"]:
            raise SystemExit(f"ABORT: corrupted window {b} does not match stored example")
        mask = [bool(x) for x in (dc[b] != demo[b]).tolist()]
        if mask != ref[b]["true_corrupt"]:
            raise SystemExit(f"ABORT: true_corrupt mask {b} does not match stored example")
    print(f"[replay] windows 0-{len(ref)-1} reproduce the stored examples bit-exactly")

    windows = []
    for i in range(N_DEMO):
        c, k = _decode(demo[i]), _decode(dc[i])
        swaps = [
            {"pos": m.start(), "clean": m.group(), "false": k[m.start() : m.end()]}
            for m in re.finditer(r"[a-z]+", c)
            if k[m.start() : m.end()] != m.group()
        ]
        windows.append({"window": i, "clean": c, "corrupted": k, "swaps": swaps})

    OUT.write_text(json.dumps({
        "ckpt": str(CKPT.relative_to(REPO)),
        "source": "scripts/replay_falseinfo_demo.py",
        "note": ("demo falseinfo corruption of bench_heal_final, seed-0 realization "
                 "(seed 42+100+0); windows 0-3 verified against nll_falseinfo.json examples"),
        "fit_seqs": FIT, "n_demo": N_DEMO, "seed": SEED, "corrupt_rate": RATE,
        "windows": windows,
    }, indent=2))
    print(f"Wrote {OUT} ({sum(len(w['swaps']) for w in windows)} swaps in {N_DEMO} windows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

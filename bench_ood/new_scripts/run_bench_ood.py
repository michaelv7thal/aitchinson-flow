"""Reproducible OOD-detection benchmark orchestrator.

Runs the four detectors (NLL, BLR, BGMM, GPT2-spilled-energy) on the SAME test
sequences + corruption ladder, plus the latent-split geometry check and the
random-vs-plausible swap experiment, each at its documented per-detector path-time.
Every arm's exact CLI + provenance (git SHA, ckpt path+md5, GPU, seed, N, fit-seqs,
rates, timestamps, per-arm status/wall-time) is recorded in
``bench_ood/manifest.json`` so the whole sweep is reproducible from one file.

Continues past a failed arm (records it) so one OOM doesn't sink the suite.

    uv run python scripts/run_bench_ood.py \
        --ckpt runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM/epoch_best.pt
    uv run python scripts/run_bench_ood.py --ckpt ... --smoke   # tiny, fast
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from scripts._bench_common import (  # noqa: E402
    git_sha, ckpt_md5, gpu_name, now_iso, write_manifest, code_fingerprint,
)

PY = [sys.executable]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt",
                    default="runs/sflm_bench_a100_20g_L256_d1280L14_full/"
                            "DirichletFM/epoch_best.pt")
    ap.add_argument("--out-dir", default="bench_ood")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--fit-seqs", type=int, default=512)
    ap.add_argument("--rates", default="0.05,0.1,0.15,0.2,0.25,0.3,0.5,0.7,1.0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--schemes", default="replace,shuffle,both,falseinfo")
    ap.add_argument("--only", default="",
                    help="comma list of arm names to run (default: all)")
    ap.add_argument("--resume", action="store_true",
                    help="skip arms already recorded 'done' in the manifest WITH an "
                         "identical cmd (so an interrupted overnight run picks up where "
                         "it stopped, but stale outputs from older code are re-run)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny/fast config to validate the whole pipeline")
    args = ap.parse_args()

    if args.smoke:
        args.n, args.fit_seqs, args.rates = 16, 64, "0.15,0.5"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ck, split, n, fit, rates, seed = (
        args.ckpt, args.split, str(args.n), str(args.fit_seqs), args.rates,
        str(args.seed))
    common = ["--ckpt", ck, "--split", split, "--n", n, "--fit-seqs", fit,
              "--rates", rates, "--seed", seed]
    bgmm_iter = "200" if args.smoke else "1000"

    # per-detector best path-time (documented in the plan / CLAUDE.md)
    detector_t = {"NLL": {"t_nll": 3.0, "t_var": 7.5},
                  "BLR": {"t_eval": 4.5, "var_t_eval": 7.5},
                  "BGMM": {"t_eval": 7.5},
                  "GPT2_SE": {"score": "spilled", "char_attrib": "boundary"},
                  "GPT2_NLL": {"score": "nll", "char_attrib": "boundary"}}

    def sub(name):
        return str(out / name)

    arms = [
        ("nll", f"{sub('nll')}/denoiser_nll_sweep.json",
         PY + ["scripts/ood_denoiser_nll.py", *common, "--schemes", args.schemes,
               "--t-nll", "3.0", "--t-var", "7.5", "--no-plot",
               "--out", f"{sub('nll')}/denoiser_nll_sweep.json"]),
        ("blr", f"{sub('blr')}/bayes_linear_sweep.json",
         PY + ["scripts/ood_bayes_linear.py", *common, "--schemes", args.schemes,
               "--train-schemes", "replace", "--no-plot",
               "--out", f"{sub('blr')}/bayes_linear_sweep.json"]),
        # the "better detector": BLR trained on a synthetic adversarial MIX of
        # corruption schemes (replace+shuffle+falseinfo+both), full-dim, at t=7.5.
        ("blr_adv", f"{sub('blr_adv')}/bayes_linear_adv_sweep.json",
         PY + ["scripts/ood_bayes_linear.py", *common, "--schemes", args.schemes,
               "--train-schemes", "replace,shuffle,falseinfo,both",
               "--t-eval", "7.5", "--pca-dim", "0", "--no-plot",
               "--out", f"{sub('blr_adv')}/bayes_linear_adv_sweep.json"]),
        # controlled comparison: BLR trained on falseinfo negatives ONLY (the
        # focused head; tests whether the easy replace/shuffle negatives in the mix
        # dilute the subtle false-info direction).
        # t=4.5 to mirror the latent-split probe (full_dim_linear per-token 0.947);
        # this is the direct held-out validation of "just use the full-dim linear".
        ("blr_fi", f"{sub('blr_fi')}/bayes_linear_fi_sweep.json",
         PY + ["scripts/ood_bayes_linear.py", *common, "--schemes", args.schemes,
               "--train-schemes", "falseinfo",
               "--t-eval", "4.5", "--pca-dim", "0", "--no-plot",
               "--out", f"{sub('blr_fi')}/bayes_linear_fi_sweep.json"]),
        ("bgmm", f"{sub('bgmm')}/bgmm_perpos_sweep.json",
         PY + ["scripts/ood_bgmm_perpos.py", *common, "--schemes", args.schemes,
               "--t-evals", "7.5", "--example-t", "7.5", "--max-components", "20",
               "--covariance-type", "full", "--pca-dim", "64", "--max-iter", bgmm_iter,
               "--out", f"{sub('bgmm')}/bgmm_perpos_sweep.json"]),
        # External-LM baseline, the PAPER's method: cross-step spilled energy
        # (Minut et al., ICLR 2026). Strong at sequence level; by construction it
        # mixes a step-(i-1) logit with a step-i logsumexp, so it does NOT localize.
        ("gpt2_se", f"{sub('gpt2_se')}/gpt2_spilled_energy_sweep.json",
         PY + ["scripts/ood_gpt2_spilled_energy.py", *common, "--schemes", args.schemes,
               "--score", "spilled", "--char-attrib", "boundary", "--no-plot",
               "--out", f"{sub('gpt2_se')}/gpt2_spilled_energy_sweep.json"]),
        # Same external LM, same slices — but scored with the SAME-STEP per-token NLL.
        # This is the honest per-token comparator (and is what the repo previously
        # reported under the name "spilled energy"). Beating the real SE at per-token
        # localization would be a straw man: SE is not a localizer. Report both.
        ("gpt2_nll", f"{sub('gpt2_nll')}/gpt2_nll_sweep.json",
         PY + ["scripts/ood_gpt2_spilled_energy.py", *common, "--schemes", args.schemes,
               "--score", "nll", "--char-attrib", "boundary", "--no-plot",
               "--out", f"{sub('gpt2_nll')}/gpt2_nll_sweep.json"]),
        ("latent_split", f"{sub('latent_split')}/latent_split.json",
         PY + ["scripts/plot_latent_split.py", "--ckpt", ck, "--split", split,
               "--out-dir", sub("latent_split"), "--schemes",
               "replace,shuffle,falseinfo", "--token-scheme", "falseinfo",
               "--fit-seqs", "192", "--n", n, "--rate", "0.3", "--seed", seed]),
        ("plausible", f"{sub('plausible')}/plausible_swap.json",
         PY + ["scripts/ood_plausible_swap.py", "--ckpt", ck, "--split", split,
               "--fit-seqs", fit, "--n", "16" if args.smoke else "64", "--rate", "0.15",
               "--t-eval", "7.5", "--t-nll", "3.0", "--blr-t-eval", "4.5",
               "--n-cands", "16" if args.smoke else "48", "--bgmm-max-iter", bgmm_iter,
               "--seed", seed, "--out", f"{sub('plausible')}/plausible_swap.json"]),
    ]
    only = {s for s in args.only.split(",") if s.strip()}
    if only:
        arms = [a for a in arms if a[0] in only]

    # merge with an existing manifest so a partial re-run (--only) keeps the other
    # arms' records instead of clobbering them.
    prev_arms = {}
    mp = out / "manifest.json"
    if mp.exists():
        try:
            prev_arms = {a["name"]: a for a in json.loads(mp.read_text()).get("arms", [])}
        except (OSError, ValueError):
            prev_arms = {}
    # --resume: skip an arm ONLY if it is recorded `done` with an IDENTICAL cmd AND was
    # produced by the SAME CODE. The cmd alone is not enough — editing a detector changes
    # what the arm computes while its CLI stays byte-identical, so a cmd-only check would
    # silently keep stale results. `code_fp` invalidates them automatically.
    code_fp = code_fingerprint()
    skipped = []
    if args.resume:
        run_arms = []
        for nm, o, cmd in arms:
            p = prev_arms.get(nm)
            if (p and p.get("status") == "done" and p.get("cmd") == cmd
                    and p.get("code_fp") == code_fp and Path(o).exists()):
                skipped.append(nm)
            else:
                run_arms.append((nm, o, cmd))
        stale = [nm for nm, _, _ in arms if nm not in skipped
                 and (prev_arms.get(nm) or {}).get("status") == "done"]
        arms = run_arms
        if skipped:
            print(f"### [resume] skipping {len(skipped)} completed arm(s): "
                  f"{', '.join(skipped)}")
        if stale:
            print(f"### [resume] RE-RUNNING {len(stale)} arm(s) marked done but produced "
                  f"by different code/cmd: {', '.join(stale)}")
        if not arms:
            print("### [resume] nothing left to run — all arms already done.")

    for nm, o, cmd in arms:  # (re)set the arms we're about to run to pending
        prev_arms[nm] = {"name": nm, "out": o, "cmd": cmd, "status": "pending",
                         "code_fp": code_fp}
    record = {
        "benchmark": "ood_detection", "git_sha": git_sha(), "created": now_iso(),
        "gpu": gpu_name(), "ckpt": ck, "ckpt_md5": ckpt_md5(ck),
        "common": {"split": split, "n": args.n, "fit_seqs": args.fit_seqs,
                   "rates": rates, "seed": args.seed, "schemes": args.schemes,
                   "smoke": args.smoke},
        "detector_t": detector_t,
        "arms": list(prev_arms.values()),
    }
    by_name = {a["name"]: a for a in record["arms"]}
    write_manifest(out, record)

    for nm, o, cmd in arms:
        Path(o).parent.mkdir(parents=True, exist_ok=True)
        log = Path(o).parent / f"{nm}.log"
        print(f"### [{now_iso()}] START {nm} -> {o}")
        t0 = time.time()
        with open(log, "w") as lf:
            rc = subprocess.run(cmd, cwd=str(_ROOT), stdout=lf, stderr=subprocess.STDOUT,
                                env={**_env()}).returncode
        dt = round(time.time() - t0, 1)
        ok = rc == 0 and Path(o).exists()
        by_name[nm].update(status="done" if ok else "failed",
                           wall_s=dt, returncode=rc, log=str(log))
        write_manifest(out, record)  # checkpoint after each arm
        print(f"### [{now_iso()}] {'OK   ' if ok else 'FAIL '} {nm} ({dt}s, rc={rc})")

    done = sum(1 for nm, _, _ in arms if by_name[nm]["status"] == "done")
    print(f"\n###### bench_ood: {done}/{len(arms)} (re)run arms OK -> {out}/manifest.json")
    return 0


def _env():
    import os
    e = dict(os.environ)
    e.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return e


if __name__ == "__main__":
    raise SystemExit(main())

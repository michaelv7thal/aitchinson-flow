"""AE-scaling sweep runner — for each cell in sweeps/ae_scaling.yaml:

    1. Train the autoencoder      → runs/<name>/ae/epoch_final.pt
    2. Train an EqMAE flow on it  → runs/<name>/eqm/epoch_final.pt + eval.json
    3. Run recovery_check.py      → runs/<name>/eqm/recovery.json

Idempotent: each step skips if its target artefact already exists, so
re-running after an interruption resumes where it left off.

Run via:
    env -u PYTHONPATH .venv/bin/python scripts/run_ae_scaling.py \\
        --sweep sweeps/ae_scaling.yaml --runs-root runs

To re-do just one cell: --only <name>. To re-run everything from scratch
on a cell, delete the runs/<name>/ directory.

The EqMAE backbone is held constant at the comp_ baseline size
(d_model=1024, num_layers=8) so the AE width/depth is the only varying
flow-relevant quantity. Recovery uses the same alpha grid as the comp_
sweep (n=256, steps=200, alphas 0.05..1.00).
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")


def _run(cmd: list[str], log: Path) -> int:
    """Run a subprocess with PYTHONPATH stripped, append stdout+stderr to log."""
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[exec] {' '.join(shlex.quote(c) for c in cmd)} → {log}")
    with log.open("a") as fh:
        fh.write(f"\n--- {' '.join(shlex.quote(c) for c in cmd)} ---\n")
        fh.flush()
        env = {k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"}
        rc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        print(f"[error] rc={rc}; see {log}")
    return rc


def _train_ae(cell: dict[str, Any], cell_dir: Path, log: Path) -> int:
    ae_dir = cell_dir / "ae"
    ckpt = ae_dir / "epoch_final.pt"
    if ckpt.exists():
        print(f"[skip ae] {cell['name']} (ckpt exists)")
        return 0
    ae = cell["ae"]
    train = cell["ae_train"]
    cmd = [
        PY, "scripts/train_autoencoder.py",
        "--out", str(ae_dir),
        "--d-model", str(ae["d_model"]),
        "--num-layers", str(ae["num_layers"]),
        "--nhead", str(ae["nhead"]),
        "--d-latent", str(ae["d_latent"]),
        "--denoising-schedule", str(ae.get("denoising_schedule", "relative_uniform")),
        "--denoising-sigma", str(ae["denoising_sigma"]),
        "--latent-l2", str(ae["latent_l2"]),
        "--epochs", str(train["epochs"]),
        "--windows", str(train["windows"]),
        "--batch-size", str(train["batch_size"]),
        "--lr", str(train["lr"]),
        "--seed", "42",
        "--mode", str(ae.get("mode", "ae")),
    ]
    if "vae_beta" in ae:
        cmd.extend(["--vae-beta", str(ae["vae_beta"])])
    if "vae_beta_warmup_epochs" in ae:
        cmd.extend(["--vae-beta-warmup-epochs", str(ae["vae_beta_warmup_epochs"])])
    if "L" in train:
        cmd.extend(["--seq-len", str(train["L"])])
    if train.get("variable_length"):
        cmd.append("--variable-length")
        cmd.extend(["--L-min", str(train.get("L_min", 40))])
        cmd.extend(["--L-max", str(train.get("L_max", 128))])
    return _run(cmd, log)


def _measure_latent_sigma(ae_ckpt: Path, cell: dict[str, Any]) -> float:
    """Run the trained AE on a held-out batch and return the per-dim std of z.

    Used to set EqMAE's source_sigma so the FM source N(0, σ²I) matches the
    encoded x1 scale. Without this, x0 is dominated by signal at all γ and
    the regression sees no real noise.
    """
    cache_path = ae_ckpt.parent / "latent_stats.json"
    if cache_path.exists():
        return float(json.loads(cache_path.read_text())["per_dim_std"])

    import torch
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    import aitchinson_flow.models  # noqa: F401 — populate registry
    from aitchinson_flow.config import Config
    from aitchinson_flow.models import build_model
    from aitchinson_flow.training import build_training_datamodule

    payload = torch.load(ae_ckpt, map_location="cpu", weights_only=False)
    cfg = Config()
    # Rebuild the AE config from the saved cell spec (more reliable than
    # parsing the saved cfg dict which may lack autoencoder section in older
    # ckpts).
    from dataclasses import replace
    ae = cell["ae"]
    ae_kwargs = dict(
        d_model=int(ae["d_model"]),
        num_layers=int(ae["num_layers"]),
        nhead=int(ae["nhead"]),
        d_latent=int(ae["d_latent"]),
        denoising_schedule=str(ae.get("denoising_schedule", "relative_uniform")),
        denoising_sigma=float(ae["denoising_sigma"]),
        latent_l2=float(ae["latent_l2"]),
        activation="gelu",
        tie_embeddings=True,
        mode=str(ae.get("mode", "ae")),
    )
    if "vae_beta" in ae:
        ae_kwargs["vae_beta"] = float(ae["vae_beta"])
    cfg.autoencoder = replace(cfg.autoencoder, **ae_kwargs)
    cfg.training = replace(cfg.training, model_name="TextAE")
    eqm_train = cell.get("eqm_train") or cell.get("ae_train") or {}
    if "L" in eqm_train:
        L = int(eqm_train["L"])
        cfg.training = replace(cfg.training, L=L)
        cfg.text8_dataset = replace(cfg.text8_dataset, L=L)
    model = build_model(cfg).to(cfg.training.device)
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()
    dm, _ = build_training_datamodule(cfg)
    val_ids = dm.splits.val.long()[:512].to(cfg.training.device)
    with torch.no_grad():
        z = model.encode(val_ids)
    # Per-dim std measured across (B, L, d) — same number a Gaussian source
    # N(0, σ²I) would produce per-coordinate.
    per_dim_std = float(z.flatten().std().item())
    cache_path.write_text(json.dumps({
        "per_dim_std": per_dim_std,
        "z_norm_mean": float(z.norm(dim=-1).mean().item()),
        "z_shape": list(z.shape),
    }, indent=2))
    return per_dim_std


def _write_eqm_sweep_yaml(cell: dict[str, Any], cell_dir: Path, ae_ckpt: Path) -> Path:
    """Generate a one-cell sweep YAML so we can call run_sweep.py and reuse
    its eval pipeline (writes eval.json with KL_uni/KL_bi/H_ratio/...)."""
    name = f"{cell['name']}_eqm"
    ae = cell["ae"]
    source_sigma = _measure_latent_sigma(ae_ckpt, cell)
    print(f"[autoscale] {cell['name']}: source_sigma = {source_sigma:.4f}")
    eqm_train = cell.get("eqm_train", {})
    overrides: dict[str, Any] = {
        "training.model_name": "EqMAE",
        "training.epochs": int(eqm_train.get("epochs", 5)),
        "training.seed": int(cell.get("eqm_ae_seed", 42)),
        "text8_dataset.max_train_windows": int(eqm_train.get("windows", 10000)),
        # EqMAE training-side overrides — defaults preserve v1 behaviour.
        "training.lr": float(eqm_train.get("lr", 3e-4)),
        "training.scheduler_warmup_epochs": int(eqm_train.get("scheduler_warmup_epochs", 0)),
        # AE arch must match the trained ckpt
        "autoencoder.d_model": int(ae["d_model"]),
        "autoencoder.num_layers": int(ae["num_layers"]),
        "autoencoder.nhead": int(ae["nhead"]),
        "autoencoder.d_latent": int(ae["d_latent"]),
        "autoencoder.denoising_schedule": str(ae.get("denoising_schedule", "relative_uniform")),
        "autoencoder.denoising_sigma": float(ae["denoising_sigma"]),
        "autoencoder.latent_l2": float(ae["latent_l2"]),
        "autoencoder.activation": "gelu",
        "autoencoder.tie_embeddings": True,
        "autoencoder.mode": str(ae.get("mode", "ae")),
        "eqm_ae.ae_ckpt_path": str(ae_ckpt),
        # EqM backbone matches the comp_ baselines
        "eqm.lambda_ce": 0.5,
        "eqm.gamma_power": 1.0,
        # Per-cell source_sigma so FM source N(0, σ²I) matches the
        # encoded x1's per-dim std — fixes the "x0 is negligible at
        # all γ" failure mode.
        "eqm.source_sigma": float(source_sigma),
        "loss.mode": "mse",
    }
    if "vae_beta" in ae:
        overrides["autoencoder.vae_beta"] = float(ae["vae_beta"])
    if "vae_beta_warmup_epochs" in ae:
        overrides["autoencoder.vae_beta_warmup_epochs"] = int(ae["vae_beta_warmup_epochs"])
    if "L" in eqm_train:
        L = int(eqm_train["L"])
        overrides["training.L"] = L
        overrides["text8_dataset.L"] = L
    if "batch_size" in eqm_train:
        B = int(eqm_train["batch_size"])
        overrides["training.B"] = B
        overrides["text8_dataset.batch_size"] = B
    if eqm_train.get("variable_length"):
        L_max = int(eqm_train.get("L_max", 128))
        overrides["text8_dataset.variable_length"] = True
        overrides["text8_dataset.L_min"] = int(eqm_train.get("L_min", 40))
        overrides["text8_dataset.L_max"] = L_max
        # Pos emb sized to L_max
        overrides["training.L"] = L_max
        overrides["text8_dataset.L"] = L_max
    # Per-cell EqM overrides (e.g. lambda_bigram_joint) — flattened into
    # eqm.<key> keys. Lets a sweep cell opt into auxiliary heads without
    # editing this script.
    for k, v in (cell.get("eqm") or {}).items():
        overrides[f"eqm.{k}"] = v
    spec = [{"name": name, "overrides": overrides}]
    yaml_path = cell_dir / "_eqm_sweep.yaml"
    yaml_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return yaml_path


def _train_eqm_ae(cell: dict[str, Any], cell_dir: Path, log: Path) -> int:
    ae_ckpt = cell_dir / "ae" / "epoch_final.pt"
    if not ae_ckpt.exists():
        print(f"[error] AE ckpt missing for {cell['name']}: {ae_ckpt}")
        return 1
    eqm_dir = cell_dir / "eqm"
    eval_path = eqm_dir / "eval.json"
    if eval_path.exists():
        print(f"[skip eqm] {cell['name']} (eval.json exists)")
        return 0
    yaml_path = _write_eqm_sweep_yaml(cell, cell_dir, ae_ckpt)
    # run_sweep.py writes to runs-root/<name>/, so point runs-root at cell_dir
    # and the sweep cell name becomes the subdir. To get cell_dir/eqm/, name
    # the sweep cell "eqm".
    spec = yaml.safe_load(yaml_path.read_text())
    spec[0]["name"] = "eqm"
    yaml_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    cmd = [
        PY, "scripts/run_sweep.py",
        "--sweep", str(yaml_path),
        "--runs-root", str(cell_dir),
    ]
    return _run(cmd, log)


def _recovery(cell: dict[str, Any], cell_dir: Path, log: Path) -> int:
    eqm_dir = cell_dir / "eqm"
    ckpt = eqm_dir / "epoch_final.pt"
    out = eqm_dir / "recovery.json"
    if out.exists():
        print(f"[skip recovery] {cell['name']} (recovery.json exists)")
        return 0
    if not ckpt.exists():
        print(f"[error] EqMAE ckpt missing for {cell['name']}: {ckpt}")
        return 1
    cmd = [
        PY, "scripts/recovery_check.py",
        "--ckpt", str(ckpt),
        # Denser grid in the informative band (pt_acc transitions smoothly
        # from ~0.95 to ~0.40 across α ∈ [0.20, 0.50]) plus three coarser
        # points to cover the destruction regime. Skips α<0.15 because the
        # CLR/latent argmax is robust to per-dim noise that small — pt_acc
        # is trivially 1.0 there.
        "--alphas", "0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80,1.00",
        "--n", "256", "--steps", "200",
        "--out", str(out),
    ]
    return _run(cmd, log)


def _process_cell(cell: dict[str, Any], runs_root: Path) -> dict[str, int]:
    name = cell["name"]
    cell_dir = runs_root / name
    cell_dir.mkdir(parents=True, exist_ok=True)
    log = cell_dir / "_runlog.log"
    print(f"\n=== {name} ===")
    rc_ae = _train_ae(cell, cell_dir, log)
    if rc_ae != 0:
        return {"ae": rc_ae, "eqm": -1, "recovery": -1}
    rc_eqm = _train_eqm_ae(cell, cell_dir, log)
    if rc_eqm != 0:
        return {"ae": rc_ae, "eqm": rc_eqm, "recovery": -1}
    rc_rec = _recovery(cell, cell_dir, log)
    return {"ae": rc_ae, "eqm": rc_eqm, "recovery": rc_rec}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", required=True)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--only", default=None, help="comma-separated subset of names")
    args = p.parse_args()

    cells = yaml.safe_load(Path(args.sweep).read_text())
    if not isinstance(cells, list):
        print("sweep file must be a list", file=sys.stderr)
        return 2
    only = set(args.only.split(",")) if args.only else None
    if only:
        cells = [c for c in cells if c["name"] in only]
        if not cells:
            print(f"no matching cells for --only={args.only}", file=sys.stderr)
            return 2

    runs_root = Path(args.runs_root)
    summary = []
    for cell in cells:
        rcs = _process_cell(cell, runs_root)
        summary.append({"name": cell["name"], **rcs})

    print("\n=== sweep summary ===")
    for s in summary:
        print(json.dumps(s))
    failed = [s for s in summary if any(v != 0 for v in (s["ae"], s["eqm"], s["recovery"]))]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

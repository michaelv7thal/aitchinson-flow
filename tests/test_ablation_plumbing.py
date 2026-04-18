"""Tests for benchmark ablation plumbing (Plan point 5).

Covers:
- ``apply_transformer_scale`` honors ablation overrides (``velocity_loss``,
  ``transform_mode``, ``label_smoothing``, ``model_name``) without forcing
  backbone-scale args.
- ``TextAuditTask`` surfaces ``ablation_tags`` in its result dict and exposes
  Stage 1 (``auroc_energy``) vs Stage 2 (``auroc_auditor``) AUROCs side by
  side so the Plan-point-5 score comparison can be tabulated from a single
  run.
"""

from __future__ import annotations

import numpy as np
import torch

from aitchinson_flow.config import Config
from benchmarks.runner import _ABLATION_KEYS, apply_transformer_scale
from benchmarks.tasks.text_audit import TextAuditTask


class TestApplyTransformerScaleAblations:
    def test_velocity_loss_only_override(self) -> None:
        cfg = Config()
        cfg.training.velocity_loss = "soft_hilbert"
        out = apply_transformer_scale(cfg, velocity_loss="clr_mse")
        assert out.training.velocity_loss == "clr_mse"
        assert cfg.training.velocity_loss == "soft_hilbert", "must not mutate input"

    def test_transform_mode_and_label_smoothing(self) -> None:
        cfg = Config()
        out = apply_transformer_scale(cfg, transform_mode="clr", label_smoothing=0.05)
        assert out.hf_dataset.transform_mode == "clr"
        assert out.hf_dataset.label_smoothing == 0.05

    def test_model_name_override(self) -> None:
        cfg = Config()
        out = apply_transformer_scale(cfg, model_name="bayesian_auditor_stage1")
        assert out.training.model_name == "bayesian_auditor_stage1"

    def test_ablation_keys_match_signature(self) -> None:
        # Defensive: keep the runner allowlist in sync with the kwargs above.
        assert _ABLATION_KEYS == {
            "velocity_loss",
            "transform_mode",
            "label_smoothing",
            "model_name",
        }

    def test_no_backbone_scale_kwargs_required(self) -> None:
        cfg = Config()
        d0 = cfg.transformer.d_model
        out = apply_transformer_scale(cfg, velocity_loss="hard_hilbert")
        assert out.transformer.d_model == d0


class _FakeAuditor(torch.nn.Module):
    """Minimal AuditorModel-compatible stub.

    Returns deterministic, well-separated scores so AUROC contracts can be
    asserted without training a real model.
    """

    def __init__(self) -> None:
        super().__init__()
        self._tag = torch.nn.Parameter(torch.zeros(1))

    def to(self, *_a, **_k):  # noqa: D401
        return self

    def eval(self):  # noqa: D401
        return self

    def audit(self, batch):  # noqa: ANN001
        return {"audit_loss": torch.tensor(0.0)}

    def score_per_sample(self, log_x):  # noqa: ANN001
        # mean over (L, D) — invalid sequences will have larger magnitudes
        return log_x.reshape(log_x.shape[0], -1).mean(dim=-1)

    def energy_score(self, log_x):  # noqa: ANN001
        # Negative magnitude — `_auditor_energy_score` will sign-flip to anomaly.
        return -log_x.reshape(log_x.shape[0], -1).pow(2).mean(dim=-1)


class _FakeDM:
    def __init__(self, batches):
        self._batches = batches

    def test_dataloader(self):
        return self._batches

    def val_dataloader(self):
        return None

    def train_dataloader(self):
        return self._batches


def _make_batches(n_batches: int = 2, B: int = 4, L: int = 6, D: int = 5) -> list[dict]:
    rng = torch.Generator().manual_seed(0)
    batches: list[dict] = []
    for _ in range(n_batches):
        valid = 0.01 * torch.randn(B, L, D, generator=rng)
        invalid = 0.5 + 0.01 * torch.randn(B, L, D, generator=rng)
        batches.append({"log_x": valid, "log_x_invalid": invalid})
    return batches


class TestAblationTagsInResult:
    def test_tags_reflect_config(self) -> None:
        cfg = Config()
        cfg.training.device = torch.device("cpu")
        cfg.training.velocity_loss = "clr_mse"
        cfg.training.model_name = "bayesian_auditor_stage1"
        cfg.hf_dataset.transform_mode = "clr"
        cfg.hf_dataset.label_smoothing = 0.1
        cfg.dataset.K = 5
        cfg.dataset.L = 6
        cfg.benchmark.compute_spilled_energy = False
        cfg.benchmark.corrupt_rate_sweep = None

        dm = _FakeDM(_make_batches())
        task = TextAuditTask()
        result = task.run(_FakeAuditor(), dm, cfg)

        assert "ablation_tags" in result
        tags = result["ablation_tags"]
        assert tags["velocity_loss"] == "clr_mse"
        assert tags["transform_mode"] == "clr"
        assert tags["model_name"] == "bayesian_auditor_stage1"
        assert tags["label_smoothing"] == 0.1
        assert tags["K"] == 5
        assert tags["L"] == 6
        # CLR mode → feature_dim == K
        assert tags["feature_dim"] == 5

    def test_energy_and_auditor_aurocs_both_present(self) -> None:
        cfg = Config()
        cfg.training.device = torch.device("cpu")
        cfg.dataset.K = 5
        cfg.dataset.L = 6
        cfg.benchmark.compute_spilled_energy = False
        cfg.benchmark.corrupt_rate_sweep = None

        dm = _FakeDM(_make_batches())
        result = TextAuditTask().run(_FakeAuditor(), dm, cfg)

        assert "auroc_auditor" in result
        assert "auroc_energy" in result
        # Both should be in [0, 1] when defined
        for k in ("auroc_auditor", "auroc_energy"):
            v = result[k]
            assert np.isnan(v) or 0.0 <= v <= 1.0

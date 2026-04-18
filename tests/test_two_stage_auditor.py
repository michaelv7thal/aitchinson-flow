"""Smoke tests for the two-stage Bayesian auditor split.

Covers:
    * Stage 1 (`bayesian_auditor_stage1`): one training step, batch contract,
      `residual_score` shape, `audit` wiring.
    * Stage 2 (`bayesian_auditor_stage2`): strict freeze invariants, one
      training-step gradient update on GP params only, contrastive loss keys.
    * Composition: `compose_auditor_from_stages` fuses Stage 1 backbone +
      Stage 2 GP head into a `BayesianAuditor` and supports `audit` +
      `forward` (`_velocity` via GP gradient).
    * Registry lookup works for both stage names.
"""

from __future__ import annotations

import pytest
import torch

from aitchinson_flow.config import Config
from aitchinson_flow.models import (
    BayesianAuditor,
    BayesianAuditorStage1,
    BayesianAuditorStage2,
    compose_auditor_from_stages,
)
from aitchinson_flow.models.base import TRAINING_LOSS_KEY
from aitchinson_flow.models.factory import build_model, registered_model_names


def _tiny_cfg(*, K: int = 5, L: int = 6, B: int = 3) -> Config:
    cfg = Config()
    cfg.dataset.K = K
    cfg.dataset.L = L
    cfg.transformer.d_model = 16
    cfg.transformer.nhead = 2
    cfg.transformer.num_layers = 1
    cfg.transformer.d_latent = 8
    cfg.gp.num_inducing = 8
    cfg.training.B = B
    cfg.training.device = torch.device("cpu")
    cfg.training.velocity_loss = "soft_hilbert"
    return cfg


def _valid_batch(
    cfg: Config, *, with_invalid: bool = False, seed: int = 0
) -> dict[str, torch.Tensor]:
    """Deterministic tiny batch for unit tests.

    A fixed seed is essential: unseeded ``torch.randn`` makes every numerical
    assertion in the test suite potentially flaky, since the GP predictive
    distribution and contrastive margin both depend on the input values.
    """
    B, L, D = cfg.training.B, cfg.dataset.L, max(1, cfg.dataset.K - 1)
    gen = torch.Generator().manual_seed(seed)
    batch: dict[str, torch.Tensor] = {"log_x": torch.randn(B, L, D, generator=gen)}
    if with_invalid:
        batch["log_x_invalid"] = torch.randn(B, L, D, generator=gen) * 1.5 + 0.2
    return batch


class TestRegistryNames:
    def test_stage1_and_stage2_registered(self) -> None:
        names = registered_model_names()
        assert "bayesian_auditor_stage1" in names
        assert "bayesian_auditor_stage2" in names

    def test_build_stage1_via_registry(self) -> None:
        cfg = _tiny_cfg()
        cfg.training.model_name = "bayesian_auditor_stage1"
        model = build_model(cfg)
        assert isinstance(model, BayesianAuditorStage1)

    def test_build_stage2_via_registry(self) -> None:
        cfg = _tiny_cfg()
        cfg.training.model_name = "bayesian_auditor_stage2"
        model = build_model(cfg)
        assert isinstance(model, BayesianAuditorStage2)


class TestStage1:
    def test_requires_hilbert_velocity_loss(self) -> None:
        cfg = _tiny_cfg()
        cfg.training.velocity_loss = "mse"
        try:
            BayesianAuditorStage1(cfg)
        except ValueError as e:
            assert "Hilbert" in str(e)
        else:
            raise AssertionError("Stage 1 must reject non-Hilbert velocity losses")

    @pytest.mark.parametrize(
        "velocity_loss", ["soft_hilbert", "hard_hilbert", "clr_mse", "ilr_mse"]
    )
    def test_training_step_produces_loss_and_metrics(self, velocity_loss: str) -> None:
        """Every member of ``_HILBERT_FAMILY`` must drive a valid Stage 1 step.

        Regression for M11: previously only ``soft_hilbert`` was tested, so a
        backward-pass regression in any of the other three velocity losses
        would slip through CI.
        """
        cfg = _tiny_cfg()
        cfg.training.velocity_loss = velocity_loss
        model = BayesianAuditorStage1(cfg)
        batch = _valid_batch(cfg)
        out = model.training_step(batch, step=0)
        assert "loss" in out
        assert out["loss"].requires_grad
        assert "flow_loss" in out
        assert "velocity_loss" in out
        out["loss"].backward()

    def test_one_optimizer_step_updates_backbone(self) -> None:
        cfg = _tiny_cfg()
        model = BayesianAuditorStage1(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        out = model.training_step(_valid_batch(cfg), step=0)
        out["loss"].backward()
        opt.step()
        after = model.state_dict()
        changed = [k for k in before if not torch.equal(before[k], after[k])]
        assert any(k.startswith("backbone.") for k in changed)
        assert any(k.startswith("velocity_head.") for k in changed)

    def test_residual_score_shape(self) -> None:
        cfg = _tiny_cfg()
        model = BayesianAuditorStage1(cfg)
        batch = _valid_batch(cfg)
        score = model.residual_score(batch["log_x"])
        assert score.shape == (cfg.training.B,)
        assert torch.isfinite(score).all()

    def test_energy_score_matches_soft_hilbert_definition(self) -> None:
        """``energy_score = -d_H(f(x), x)`` averaged over tokens."""
        from aitchinson_flow.geometry import nielsen_soft_hilbert_distance

        cfg = _tiny_cfg()
        model = BayesianAuditorStage1(cfg)
        model.eval()
        log_x = _valid_batch(cfg)["log_x"]
        f = model.forward(log_x)
        f_c = f - f.mean(dim=-1, keepdim=True)
        x_c = log_x - log_x.mean(dim=-1, keepdim=True)
        d = nielsen_soft_hilbert_distance(
            f_c, x_c, alpha=cfg.training.soft_hilbert_alpha
        )
        expected = -d.mean(dim=-1)
        got = model.energy_score(log_x)
        assert got.shape == (cfg.training.B,)
        assert torch.allclose(got, expected, atol=1e-6)

    def test_ood_score_is_negated_energy(self) -> None:
        cfg = _tiny_cfg()
        model = BayesianAuditorStage1(cfg)
        log_x = _valid_batch(cfg)["log_x"]
        e = model.energy_score(log_x)
        ood = model.ood_score(log_x)
        assert ood.shape == (cfg.training.B,)
        assert torch.allclose(ood, -e, atol=1e-6)
        # Anomaly convention: ood_score must be non-negative
        assert (ood >= 0).all(), "ood_score = d_H must be non-negative"

    def test_score_per_sample_uses_geometric_energy_not_velocity_norm(self) -> None:
        """Plan point 3: ``score_per_sample`` is wired to ``ood_score`` (energy),
        NOT the legacy velocity-norm residual."""
        cfg = _tiny_cfg()
        model = BayesianAuditorStage1(cfg)
        log_x = _valid_batch(cfg)["log_x"]
        sps = model.score_per_sample(log_x)
        ood = model.ood_score(log_x)
        resid = model.residual_score(log_x)
        assert torch.allclose(sps, ood, atol=1e-6), (
            "score_per_sample must equal ood_score (Stage 1 geometric energy)"
        )
        # The energy and the velocity-norm residual must NOT be the same
        # quantity (otherwise the benchmark fallback to residual is silent).
        assert not torch.allclose(sps, resid, atol=1e-6), (
            "score_per_sample must not silently fall back to the velocity-norm residual"
        )

    def test_audit_path_uses_log_x(self) -> None:
        cfg = _tiny_cfg()
        model = BayesianAuditorStage1(cfg)
        out = model.audit(_valid_batch(cfg))
        assert "loss" in out

    def test_c_gamma_linear(self) -> None:
        cfg = _tiny_cfg()
        cfg.equilibrium.eqm_decay_strategy = "linear"
        model = BayesianAuditorStage1(cfg)
        gamma = torch.tensor([0.0, 0.25, 0.75])
        got = model._c_gamma(gamma).squeeze(-1).squeeze(-1)
        want = 1.0 - gamma
        assert torch.allclose(got, want)

    def test_c_gamma_truncated(self) -> None:
        cfg = _tiny_cfg()
        cfg.equilibrium.eqm_decay_strategy = "truncated"
        cfg.equilibrium.eqm_decay_a = 0.4
        model = BayesianAuditorStage1(cfg)
        gamma = torch.tensor([0.2, 0.4, 0.7])
        got = model._c_gamma(gamma).squeeze(-1).squeeze(-1)
        want = torch.tensor([1.0, 1.0, (1.0 - 0.7) / (1.0 - 0.4)])
        assert torch.allclose(got, want)

    def test_c_gamma_piecewise(self) -> None:
        cfg = _tiny_cfg()
        cfg.equilibrium.eqm_decay_strategy = "piecewise"
        cfg.equilibrium.eqm_decay_a = 0.5
        cfg.equilibrium.eqm_decay_b = 2.0
        model = BayesianAuditorStage1(cfg)
        gamma = torch.tensor([0.25, 0.8])
        got = model._c_gamma(gamma).squeeze(-1).squeeze(-1)
        left = 2.0 - ((2.0 - 1.0) / 0.5) * 0.25
        right = (1.0 - 0.8) / (1.0 - 0.5)
        want = torch.tensor([left, right])
        assert torch.allclose(got, want)

    def test_eqm_target_uses_log_x0_minus_log_x1_sign(self) -> None:
        """``u_tgt = c(gamma) * (log_x0 - log_x1)`` (data → noise direction).

        We seed `_uniform_log_x0` to a known constant, set ``c(gamma) = 1`` via
        the linear schedule + gradient multiplier, and verify the loss matches
        the Hilbert-family loss against ``log_x0 - log_x1`` (NOT
        ``log_x1 - log_x0``).
        """
        from aitchinson_flow.loss import build_velocity_loss
        from aitchinson_flow.models.bayesian_auditor_stage1 import _uniform_log_x0
        import aitchinson_flow.models.bayesian_auditor_stage1 as stage1_mod

        # Slightly larger K/L than the default _tiny_cfg so the Hilbert
        # distance between expected/wrong sign targets is comfortably above
        # the 1e-5 atol used for comparison (M12 fix: no more conditional
        # assertion).
        cfg = _tiny_cfg(K=9, L=8, B=4)
        cfg.equilibrium.eqm_decay_strategy = "linear"
        cfg.equilibrium.eqm_gradient_lambda = 1.0
        model = BayesianAuditorStage1(cfg)
        model.eval()

        torch.manual_seed(0)
        log_x1 = torch.randn(cfg.training.B, cfg.dataset.L, max(1, cfg.dataset.K - 1))

        gamma_const = torch.zeros(cfg.training.B)
        original_uniform = _uniform_log_x0
        original_rand = torch.rand

        def _patched_rand(*args, **kwargs):  # type: ignore[no-untyped-def]
            return gamma_const

        log_x0 = original_uniform(*log_x1.shape, log_x1.device, log_x1.dtype)
        log_x_gamma = log_x0
        v_pred = model.forward(log_x_gamma)

        torch.rand = _patched_rand
        try:
            loss = model._eqm_hilbert_loss(log_x1)[TRAINING_LOSS_KEY]
        finally:
            torch.rand = original_rand

        loss_fn = build_velocity_loss(
            "soft_hilbert", soft_hilbert_alpha=cfg.training.soft_hilbert_alpha
        )
        u_tgt_correct = log_x0 - log_x1
        u_tgt_wrong = log_x1 - log_x0
        expected = loss_fn(v_pred, u_tgt_correct).detach()
        wrong = loss_fn(v_pred, u_tgt_wrong).detach()

        assert torch.allclose(loss.detach(), expected, atol=1e-5), (
            f"Stage 1 EqM target sign drift: got {loss.item():.6e}, "
            f"expected (log_x0 - log_x1) = {expected.item():.6e}"
        )
        # The test itself is only meaningful if expected and wrong are
        # numerically distinct — otherwise the following assertion would
        # pass trivially regardless of the sign convention.
        assert not torch.allclose(expected, wrong, atol=1e-5), (
            "sign-convention test requires numerically distinct expected vs "
            "wrong losses; tune the test cfg (K, L, seed) to guarantee "
            f"separation (got expected={expected.item():.6e}, wrong={wrong.item():.6e})"
        )
        assert not torch.allclose(loss.detach(), wrong, atol=1e-5), (
            "Stage 1 EqM loss matches the wrong-sign target — sign convention "
            "bug in ``_eqm_hilbert_loss``"
        )

        del stage1_mod  # keep linter quiet about unused import

    def test_integrate_subtracts_predicted_velocity(self) -> None:
        """``integrate`` must apply ``x ← x - v(x) * dt`` to flow noise → data.

        With ``u_tgt = log_x0 - log_x1`` the learned velocity points data → noise,
        so the inference flow direction is the **negation** of v.
        """
        cfg = _tiny_cfg()
        model = BayesianAuditorStage1(cfg)
        model.eval()
        gen = torch.Generator().manual_seed(0)
        log_x = torch.randn(
            cfg.training.B,
            cfg.dataset.L,
            max(1, cfg.dataset.K - 1),
            generator=gen,
        )
        # Center into V_K so the post-step ``x - x.mean(-1)`` projection is a
        # no-op and the expected equality is exact (without centering, the
        # test's mean-drift cleanup inside ``integrate`` would make the strict
        # allclose deliberately fail).
        log_x = log_x - log_x.mean(dim=-1, keepdim=True)
        v = model.forward(log_x)
        out = model.integrate(log_x, steps=1, dt=0.1)
        expected = log_x - v * 0.1
        assert torch.allclose(out, expected, atol=1e-5), (
            "integrate must subtract v(x) * dt to match the (data → noise) sign convention"
        )

    def test_c_gamma_gradient_multiplier(self) -> None:
        cfg = _tiny_cfg()
        cfg.equilibrium.eqm_decay_strategy = "linear"
        cfg.equilibrium.eqm_gradient_lambda = 2.0
        model = BayesianAuditorStage1(cfg)
        gamma = torch.tensor([0.2, 0.6])
        got = model._c_gamma(gamma).squeeze(-1).squeeze(-1)
        want = (1.0 - gamma) / 2.0
        assert torch.allclose(got, want)


class TestStage2Freezing:
    def test_backbone_frozen_latent_head_and_gp_trainable(self) -> None:
        cfg = _tiny_cfg()
        model = BayesianAuditorStage2(cfg)
        assert all(not p.requires_grad for p in model.backbone.parameters())
        assert any(p.requires_grad for p in model.latent_head.parameters())
        assert any(p.requires_grad for p in model.gp.parameters())

    def test_train_keeps_backbone_in_eval_but_latent_head_active(self) -> None:
        cfg = _tiny_cfg()
        model = BayesianAuditorStage2(cfg)
        model.train()
        assert not model.backbone.training
        assert model.latent_head.training

    def test_trainable_parameters_excludes_backbone_only(self) -> None:
        cfg = _tiny_cfg()
        model = BayesianAuditorStage2(cfg)
        trainable_ids = {id(p) for p in model.trainable_parameters()}
        for p in model.backbone.parameters():
            assert id(p) not in trainable_ids
        for p in model.latent_head.parameters():
            assert id(p) in trainable_ids
        for p in model.gp.parameters():
            assert id(p) in trainable_ids

    def test_one_optimizer_step_updates_latent_head_and_gp_not_backbone(self) -> None:
        cfg = _tiny_cfg()
        model = BayesianAuditorStage2(cfg)
        opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-2)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        out = model.training_step(_valid_batch(cfg), step=0)
        out["loss"].backward()
        opt.step()
        after = model.state_dict()
        changed = [k for k in before if not torch.equal(before[k], after[k])]
        assert changed, "expected at least the GP / latent_head to update"
        for k in changed:
            assert not k.startswith("backbone."), f"backbone key changed: {k}"
        assert any(k.startswith("latent_head.") for k in changed), (
            "expected latent_head parameters to receive gradient updates"
        )
        assert any(k.startswith("gp.") for k in changed), (
            "expected GP parameters to receive gradient updates"
        )

    def test_backbone_receives_no_gradient(self) -> None:
        cfg = _tiny_cfg()
        model = BayesianAuditorStage2(cfg)
        out = model.training_step(_valid_batch(cfg), step=0)
        out["loss"].backward()
        for p in model.backbone.parameters():
            assert p.grad is None or torch.all(p.grad == 0)
        for p in model.latent_head.parameters():
            if p.requires_grad:
                assert p.grad is not None, (
                    "latent_head must be in the autograd graph (gradient not None)"
                )

    def test_latent_head_receives_gradient_on_first_step(self) -> None:
        """Token-level NLL gives non-zero d(loss)/dz even at init.

        Unlike the contrastive setup, the NLL depends on the GP epistemic
        variance at each token location, which is a non-trivial function of
        ``z`` even when the variational mean is zero. So gradient should flow
        into ``latent_head`` on the very first training step.
        """
        cfg = _tiny_cfg()
        model = BayesianAuditorStage2(cfg)
        out = model.training_step(_valid_batch(cfg), step=0)
        out["loss"].backward()
        any_lh_grad = any(
            p.requires_grad and p.grad is not None and torch.any(p.grad != 0)
            for p in model.latent_head.parameters()
        )
        assert any_lh_grad, (
            "expected latent_head to receive a non-zero gradient on the first "
            "step under the token-level NLL objective"
        )

    def test_build_optimizer_uses_trainable_parameters(self) -> None:
        from aitchinson_flow.training.optim import build_optimizer

        cfg = _tiny_cfg()
        model = BayesianAuditorStage2(cfg)
        opt = build_optimizer(model, cfg)
        opt_param_ids = {id(p) for group in opt.param_groups for p in group["params"]}
        for p in model.backbone.parameters():
            assert id(p) not in opt_param_ids, "frozen backbone params must not be in optimizer"
        for p in model.latent_head.parameters():
            assert id(p) in opt_param_ids
        for p in model.gp.parameters():
            assert id(p) in opt_param_ids

    def test_training_step_accepts_valid_only_batch(self) -> None:
        cfg = _tiny_cfg()
        model = BayesianAuditorStage2(cfg)
        out = model.training_step(_valid_batch(cfg), step=0)
        assert set(out.keys()) == {
            "loss",
            "nll",
            "contrastive",
            "anchor",
            "kl",
            "noise_var",
        }
        assert out["loss"].requires_grad
        assert out["loss"].ndim == 0

    def test_gp_hyperparameters_use_zero_weight_decay(self) -> None:
        """M5: GP kernel/inducing params must sit in a ``weight_decay=0`` group.

        Decaying inducing locations or kernel log-scales toward zero is
        semantically incorrect and empirically destabilising.
        """
        from aitchinson_flow.training.optim import (
            _GP_HYPERPARAM_ATTRS,
            build_optimizer,
        )

        cfg = _tiny_cfg()
        cfg.training.weight_decay = 0.1
        model = BayesianAuditorStage2(cfg)
        opt = build_optimizer(model, cfg)

        gp_param_ids = {
            id(getattr(model.gp, attr))
            for attr in _GP_HYPERPARAM_ATTRS
            if hasattr(model.gp, attr)
        }
        assert gp_param_ids, "expected GP module to expose at least one hyperparameter"

        for group in opt.param_groups:
            gp_in_group = any(id(p) in gp_param_ids for p in group["params"])
            if gp_in_group:
                assert group["weight_decay"] == 0.0, (
                    "GP hyperparameters must live in a weight_decay=0 group, "
                    f"got weight_decay={group['weight_decay']}"
                )

        non_gp = [
            p
            for p in model.trainable_parameters()
            if id(p) not in gp_param_ids
        ]
        if non_gp:
            decay_values = {
                group["weight_decay"]
                for group in opt.param_groups
                for p in group["params"]
                if id(p) not in gp_param_ids
            }
            assert cfg.training.weight_decay in decay_values, (
                f"expected non-GP params to use weight_decay={cfg.training.weight_decay}, "
                f"got {decay_values}"
            )

    def test_gp_noise_is_learnable(self) -> None:
        cfg = _tiny_cfg()
        model = BayesianAuditorStage2(cfg)
        assert model.gp.log_noise_var.requires_grad
        trainable_ids = {id(p) for p in model.trainable_parameters()}
        assert id(model.gp.log_noise_var) in trainable_ids

    def test_score_per_sample_shape(self) -> None:
        cfg = _tiny_cfg()
        model = BayesianAuditorStage2(cfg)
        batch = _valid_batch(cfg)
        score = model.score_per_sample(batch["log_x"])
        assert score.shape == (cfg.training.B,)

    def test_per_token_uq_returns_three_fields(self) -> None:
        cfg = _tiny_cfg()
        model = BayesianAuditorStage2(cfg)
        batch = _valid_batch(cfg)
        energy, variance, noise = model.per_token_uq(batch["log_x"])
        assert energy.shape == (cfg.training.B, cfg.dataset.L)
        assert variance.shape == (cfg.training.B, cfg.dataset.L)
        assert isinstance(noise, float)


class TestComposition:
    def test_compose_transfers_backbone_and_gp(self) -> None:
        cfg = _tiny_cfg()
        s1 = BayesianAuditorStage1(cfg)
        s2 = BayesianAuditorStage2(cfg)
        # Make stage dicts distinguishable
        with torch.no_grad():
            for p in s1.backbone.parameters():
                p.add_(0.01)
            for p in s2.gp.parameters():
                p.add_(0.02)
            for p in s2.latent_head.parameters():
                p.add_(0.03)

        composed = compose_auditor_from_stages(
            cfg, stage1_state=s1.state_dict(), stage2_state=s2.state_dict()
        )
        assert isinstance(composed, BayesianAuditor)

        # Stage 1 backbone weights should be present
        for k, v in s1.state_dict().items():
            if k.startswith("backbone."):
                assert torch.equal(composed.state_dict()[k], v)

        # Stage 2 GP + latent_head weights should be present
        for k, v in s2.state_dict().items():
            if k.startswith("gp.") or k.startswith("latent_head."):
                assert torch.equal(composed.state_dict()[k], v)

    def test_compose_drops_velocity_head_keys_safely(self) -> None:
        cfg = _tiny_cfg()
        s1 = BayesianAuditorStage1(cfg)
        composed = compose_auditor_from_stages(cfg, stage1_state=s1.state_dict())
        # BayesianAuditor has no velocity_head; composition must silently drop those keys
        assert not any(k.startswith("velocity_head.") for k in composed.state_dict())

    def test_compose_ablation_stage2_only_falls_back_to_stage2_backbone(self) -> None:
        cfg = _tiny_cfg()
        s2 = BayesianAuditorStage2(cfg)
        with torch.no_grad():
            for p in s2.backbone.parameters():
                p.add_(0.05)
        composed = compose_auditor_from_stages(cfg, stage2_state=s2.state_dict())
        for k, v in s2.state_dict().items():
            if k.startswith("backbone."):
                assert torch.equal(composed.state_dict()[k], v)

    def test_composed_auditor_can_audit_and_velocity(self) -> None:
        cfg = _tiny_cfg()
        s1 = BayesianAuditorStage1(cfg)
        s2 = BayesianAuditorStage2(cfg)
        composed = compose_auditor_from_stages(
            cfg, stage1_state=s1.state_dict(), stage2_state=s2.state_dict()
        )
        composed.eval()
        batch = _valid_batch(cfg, with_invalid=True)
        out = composed.audit(batch)
        assert "loss" in out
        # Combined inference: GP-gradient velocity
        v = composed(batch["log_x"])
        assert v.shape == batch["log_x"].shape
        # OOD variance score
        score = composed.score_per_sample(batch["log_x"])
        assert score.shape == (cfg.training.B,)
        # Residual (flow-based) score also available on composed model
        resid = composed.residual_score(batch["log_x"])
        assert resid.shape == (cfg.training.B,)


class TestComposedTrainingStep:
    """Exercise the fused ``BayesianAuditor`` training path (C7 fix).

    The composed model's ``training_step``/``_auditor_loss`` were previously
    untested — bugs in the shared flow+GP+contrastive objective would have
    slipped through despite full coverage of Stage 1 and Stage 2 individually.
    """

    def test_training_step_produces_loss_keys(self) -> None:
        cfg = _tiny_cfg()
        s1 = BayesianAuditorStage1(cfg)
        s2 = BayesianAuditorStage2(cfg)
        composed = compose_auditor_from_stages(
            cfg, stage1_state=s1.state_dict(), stage2_state=s2.state_dict()
        )
        batch = _valid_batch(cfg, with_invalid=True, seed=7)
        out = composed.training_step(batch, step=0)
        expected_keys = {"loss", "flow_loss", "mean_loss", "kl", "noise_var"}
        assert expected_keys.issubset(out.keys()), (
            f"expected {expected_keys} in {sorted(out.keys())}"
        )
        assert out["loss"].requires_grad
        assert out["loss"].ndim == 0
        assert torch.isfinite(out["loss"]).all()

    def test_training_step_backward_reaches_all_parameter_groups(self) -> None:
        """Backbone, latent_head, and GP should all receive non-zero gradients.

        Regression test: composition must not silently disconnect any branch
        of the auditor computation graph.

        Note: at default initialisation both ``gp.mean_linear.weight`` and
        ``gp.var_mean`` are zero, so the GP predictive mean is identically
        zero regardless of ``z``. That breaks the gradient path from the
        energy / contrastive terms back through the backbone and latent_head.
        We perturb ``var_mean`` so the predictive mean depends non-trivially
        on ``z`` (which is what happens after a single optimizer step in real
        training anyway).
        """
        cfg = _tiny_cfg()
        s1 = BayesianAuditorStage1(cfg)
        s2 = BayesianAuditorStage2(cfg)
        composed = compose_auditor_from_stages(
            cfg, stage1_state=s1.state_dict(), stage2_state=s2.state_dict()
        )
        with torch.no_grad():
            composed.gp.var_mean.copy_(
                torch.randn_like(composed.gp.var_mean) * 0.1
            )
            composed.gp.mean_linear.weight.copy_(
                torch.randn_like(composed.gp.mean_linear.weight) * 0.1
            )

        batch = _valid_batch(cfg, with_invalid=True, seed=11)
        out = composed.training_step(batch, step=0)
        out["loss"].backward()

        def any_nonzero_grad(module: torch.nn.Module) -> bool:
            return any(
                p.grad is not None and torch.any(p.grad != 0)
                for p in module.parameters()
                if p.requires_grad
            )

        assert any_nonzero_grad(composed.backbone), "backbone missed gradient"
        assert any_nonzero_grad(composed.latent_head), "latent_head missed gradient"
        assert any_nonzero_grad(composed.gp), "gp missed gradient"

    def test_training_step_missing_invalid_raises(self) -> None:
        cfg = _tiny_cfg()
        composed = compose_auditor_from_stages(cfg)
        batch = _valid_batch(cfg, with_invalid=False, seed=3)
        try:
            composed.training_step(batch, step=0)
        except KeyError as e:
            assert "log_x_invalid" in str(e)
        else:
            raise AssertionError(
                "BayesianAuditor.training_step must raise KeyError when "
                "batch['log_x_invalid'] is absent"
            )

    def test_kl_normalizer_plumbs_through_to_loss(self) -> None:
        """Setting ``_kl_normalizer`` rescales the KL contribution to total loss."""
        cfg = _tiny_cfg()
        # Crank lambda_kl + perturb var_mean so KL is non-trivial at init; the
        # default ``1e-4`` weight combined with a near-zero KL at init would
        # otherwise make the two losses equal to machine precision.
        cfg.gp.lambda_kl = 1.0
        composed = compose_auditor_from_stages(cfg)
        with torch.no_grad():
            composed.gp.var_mean.copy_(
                torch.randn_like(composed.gp.var_mean) * 0.5
            )
        composed.eval()
        batch = _valid_batch(cfg, with_invalid=True, seed=5)

        composed._kl_normalizer = float(cfg.training.B)
        torch.manual_seed(0)
        out_small = composed._auditor_loss(
            batch["log_x"], batch["log_x_invalid"], create_graph=False
        )

        composed._kl_normalizer = float(cfg.training.B * 100)
        torch.manual_seed(0)
        out_large = composed._auditor_loss(
            batch["log_x"], batch["log_x_invalid"], create_graph=False
        )

        # kl tensor is identical; its contribution to total differs by the ratio of normalizers.
        assert torch.allclose(out_small["kl"], out_large["kl"], atol=1e-6)
        assert out_small["kl"].item() > 0, (
            "test sanity check: KL must be strictly positive to exercise normalization"
        )
        # Total must differ: lambda_kl * kl * (1/N_small - 1/N_large) should be nonzero.
        assert not torch.allclose(
            out_small["loss"].detach(), out_large["loss"].detach(), atol=1e-4
        ), "KL normalizer change must shift the total loss"


class TestResidualScoreBenchmarkHook:
    """`residual_score` is picked up by the TextAuditTask helper for flow-vs-GP AUROC comparison."""

    def test_helper_reads_stage1_residual(self) -> None:
        from benchmarks.tasks.text_audit import _auditor_residual_score

        cfg = _tiny_cfg()
        model = BayesianAuditorStage1(cfg)
        score = _auditor_residual_score(model, _valid_batch(cfg)["log_x"])
        assert score is not None
        assert score.shape == (cfg.training.B,)

    def test_helper_reads_composed_residual(self) -> None:
        from benchmarks.tasks.text_audit import _auditor_residual_score

        cfg = _tiny_cfg()
        s1 = BayesianAuditorStage1(cfg)
        s2 = BayesianAuditorStage2(cfg)
        composed = compose_auditor_from_stages(
            cfg, stage1_state=s1.state_dict(), stage2_state=s2.state_dict()
        )
        composed.eval()
        score = _auditor_residual_score(composed, _valid_batch(cfg)["log_x"])
        assert score is not None
        assert score.shape == (cfg.training.B,)

    def test_helper_returns_none_when_missing(self) -> None:
        from benchmarks.tasks.text_audit import _auditor_residual_score
        import torch.nn as nn

        class _Empty(nn.Module):
            pass

        assert _auditor_residual_score(_Empty(), torch.zeros(1, 1, 1)) is None

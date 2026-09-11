"""TextAutoencoder — contextual denoising AE for text8 windows.

Pretraining target for ``EqMAE``. The recipe:

  encoder: token_embed + pos_embed + N-layer Transformer encoder + Linear → z
  decoder: Linear + pos_embed + N-layer Transformer encoder + Linear → logits
  loss:    CE(decoder(encoder(tokens) + σ·N(0, I)), tokens)
           + small L2 on z to keep latent norms bounded

The denoising noise is what makes the latent space useful as a flow-matching
target: the decoder learns to be robust to small perturbations of z, so a
sampler that lands "near" a data point still decodes to valid text. The
latent L2 regulariser keeps ‖z‖ bounded so the source noise σ for the
downstream EqM model can match the data scale without tuning.

Trains via the standard runner; ``training_step`` returns the LossDict shape
the loop expects. Has no ``sample`` method — disable the unigram-KL probe
when training (``training.sample_eval_every: null``).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from aitchinson_flow.config import Config
from aitchinson_flow.models import LossDict, TRAINING_LOSS_KEY, register


def _make_transformer(
    d_model: int,
    nhead: int,
    num_layers: int,
    dropout: float,
    activation: str = "gelu",
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=4 * d_model,
        dropout=dropout,
        activation=activation,
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(
        encoder_layer=layer,
        num_layers=num_layers,
        norm=nn.LayerNorm(d_model),
        enable_nested_tensor=False,
    )


class TextAutoencoder(nn.Module):
    """Token-sequence → latent → token-sequence contextual AE."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        ae = cfg.autoencoder
        K = cfg.text8_dataset.K
        L = cfg.text8_dataset.L
        d_model = ae.d_model
        d_latent = ae.d_latent

        self.K = K
        self.L = L
        self.d_latent = d_latent

        self.token_embed = nn.Embedding(K, d_model)
        self.enc_pos = nn.Embedding(L, d_model)
        self.enc_t = _make_transformer(
            d_model, ae.nhead, ae.num_layers, ae.dropout, ae.activation
        )
        self.mode = ae.mode  # "ae" | "vae"
        if self.mode == "vae":
            # VAE: parallel heads for posterior mean and log-std-dev.
            # logsig parameterises σ directly (not log(σ²)) so the KL
            # closed form is symmetric and easier to read.
            self.mu_head = nn.Linear(d_model, d_latent)
            self.logsig_head = nn.Linear(d_model, d_latent)
            self.to_latent = None  # type: ignore[assignment]
        elif self.mode == "ae":
            self.to_latent = nn.Linear(d_model, d_latent)
        else:
            raise ValueError(f"unknown autoencoder.mode={self.mode!r}")

        self.from_latent = nn.Linear(d_latent, d_model)
        self.dec_pos = nn.Embedding(L, d_model)
        self.dec_t = _make_transformer(
            d_model, ae.nhead, ae.num_layers, ae.dropout, ae.activation
        )
        self.head = nn.Linear(d_model, K, bias=True)
        if ae.tie_embeddings:
            # Standard tying pattern: assign the encoder embedding tensor as
            # the decoder head's weight. Both views share storage, so the
            # encoder gradient also updates the head and vice versa.
            self.head.weight = self.token_embed.weight

    # ----- encode / decode ------------------------------------------------- #

    def _encode_features(
        self, token_ids: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """(B, L) → (B, L, d_model) — shared encoder body for AE and VAE.

        pad_mask: optional (B, L) bool, True at padding positions. Threaded
        into the transformer's src_key_padding_mask so padded positions
        don't contaminate the attention.
        """
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device)
        h = self.token_embed(token_ids.long()) + self.enc_pos(pos).unsqueeze(0)
        h = self.enc_t(h, src_key_padding_mask=pad_mask)
        return h

    def encode(
        self,
        token_ids: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(B, L) long → (B, L, d_latent).

        For AE mode: returns z = to_latent(h).
        For VAE mode: returns μ (the posterior mean — deterministic).

        ``pad_mask`` (optional, (B, L) bool) is threaded into the encoder
        transformer's attention.

        No SDPBackend.MATH constraint — the AE's loss is first-order only,
        so PyTorch is free to dispatch to Flash/efficient attention here.
        """
        h = self._encode_features(token_ids, pad_mask=pad_mask)
        if self.mode == "vae":
            return self.mu_head(h)
        return self.to_latent(h)

    def encode_sample(
        self,
        token_ids: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Stochastic encode: returns a fresh draw from q(z|x) every call.

        For AE mode: same as encode() (no stochasticity to sample from).
        For VAE mode: z = μ + σ·ε, ε ~ N(0, I).
        """
        if self.mode != "vae":
            return self.encode(token_ids, pad_mask=pad_mask)
        h = self._encode_features(token_ids, pad_mask=pad_mask)
        mu = self.mu_head(h)
        logsig = self.logsig_head(h)
        sig = torch.exp(logsig)
        eps = torch.randn_like(mu)
        return mu + sig * eps

    def encode_dist(
        self, token_ids: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (μ, logσ) for VAE mode. Raises for AE mode."""
        if self.mode != "vae":
            raise ValueError("encode_dist is VAE-only")
        h = self._encode_features(token_ids, pad_mask=pad_mask)
        return self.mu_head(h), self.logsig_head(h)

    def decode(
        self, z: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """(B, L, d_latent) → (B, L, K) logits."""
        B, L, _ = z.shape
        pos = torch.arange(L, device=z.device)
        h = self.from_latent(z) + self.dec_pos(pos).unsqueeze(0)
        h = self.dec_t(h, src_key_padding_mask=pad_mask)
        return self.head(h)

    def decode_to_logprobs(
        self, z: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return F.log_softmax(self.decode(z, pad_mask=pad_mask), dim=-1)

    # ----- training step --------------------------------------------------- #

    def training_step(self, batch: Any, step: int) -> LossDict:
        del step
        return self._ae_loss(
            batch["token_ids"], training=True, pad_mask=batch.get("pad_mask")
        )

    def eval_step(self, batch: Any) -> LossDict:
        return self._ae_loss(
            batch["token_ids"], training=False, pad_mask=batch.get("pad_mask")
        )

    def _ae_loss(
        self,
        token_ids: torch.Tensor,
        *,
        training: bool,
        pad_mask: torch.Tensor | None = None,
    ) -> LossDict:
        ae = self.cfg.autoencoder
        if self.mode == "vae":
            return self._vae_loss(token_ids, training=training, pad_mask=pad_mask)
        z = self.encode(token_ids, pad_mask=pad_mask)
        # Multi-scale denoising trains the decoder to be robust to a *range*
        # of per-dim residuals — exactly the regime the downstream EqM
        # sampler leaves it in (close to but not exactly at the encoded z).
        sig_eff = torch.zeros((), device=z.device, dtype=z.dtype)
        if training and ae.denoising_sigma > 0.0:
            B = z.shape[0]
            if ae.denoising_schedule == "fixed":
                sig_per = z.new_full((B, 1, 1), float(ae.denoising_sigma))
            elif ae.denoising_schedule == "relative_uniform":
                # σ_b = U(0, denoising_sigma) * std(z_batch). Scale-invariant
                # noise ratio across AE sizes. Per-sample σ so each batch
                # contains the full clean→noisy spectrum.
                with torch.no_grad():
                    z_std = z.detach().std()
                u = torch.rand((B, 1, 1), device=z.device, dtype=z.dtype)
                sig_per = u * float(ae.denoising_sigma) * z_std
            else:
                raise ValueError(
                    f"unknown autoencoder.denoising_schedule={ae.denoising_schedule!r}"
                )
            z_in = z + sig_per * torch.randn_like(z)
            sig_eff = sig_per.detach().mean()
        else:
            z_in = z
        logits = self.decode(z_in, pad_mask=pad_mask)
        if pad_mask is None:
            ce = F.cross_entropy(
                logits.reshape(-1, self.K), token_ids.reshape(-1).long()
            )
            lat_l2 = z.pow(2).mean()
        else:
            # Mask CE and latent_l2 to valid positions only. Padding positions
            # contain garbage tokens (space-pad) and arbitrary z's, so
            # averaging them in would distort the loss.
            valid = (~pad_mask).reshape(-1).float()
            ce_per = F.cross_entropy(
                logits.reshape(-1, self.K),
                token_ids.reshape(-1).long(),
                reduction="none",
            )
            ce = (ce_per * valid).sum() / valid.sum().clamp(min=1.0)
            valid_f = (~pad_mask).float().unsqueeze(-1)
            lat_l2 = (z.pow(2) * valid_f).sum() / (valid_f.sum() * z.shape[-1]).clamp(min=1.0)
        total = ce + ae.latent_l2 * lat_l2
        # Diagnostics: token-level argmax acc on the clean reconstruction
        # (so we can see "is the AE actually a near-identity in expectation").
        with torch.no_grad():
            clean_logits = self.decode(z, pad_mask=pad_mask)
            if pad_mask is None:
                acc = (clean_logits.argmax(dim=-1) == token_ids.long()).float().mean()
                z_norm = z.norm(dim=-1).mean()
            else:
                valid_b = ~pad_mask  # (B, L) bool
                correct = (clean_logits.argmax(dim=-1) == token_ids.long()) & valid_b
                acc = correct.float().sum() / valid_b.float().sum().clamp(min=1.0)
                z_norm = (z.norm(dim=-1) * valid_b.float()).sum() / valid_b.float().sum().clamp(min=1.0)
        return {
            TRAINING_LOSS_KEY: total,
            "ce": ce.detach(),
            "lat_l2": lat_l2.detach(),
            "z_norm": z_norm,
            "tok_acc": acc,
            "sig_eff": sig_eff,
        }

    def _vae_loss(
        self,
        token_ids: torch.Tensor,
        *,
        training: bool,
        pad_mask: torch.Tensor | None = None,
    ) -> LossDict:
        """VAE objective: CE on reconstruction + β·KL(q(z|x) || N(0, I)).

        β-warmup: linearly anneal β from 0 to ``vae_beta`` over the first
        ``vae_beta_warmup_epochs`` epochs to avoid posterior collapse —
        the model should learn a useful encoder before being penalised
        for diverging from the unit Gaussian prior.

        The KL drives the marginal q(z) toward N(0, I), which is what the
        downstream EqM source distribution is — making unconditional
        sampling tractable as "draw z ~ N(0, I) → decode" with optional
        EBM refinement.
        """
        ae = self.cfg.autoencoder
        h = self._encode_features(token_ids, pad_mask=pad_mask)
        mu = self.mu_head(h)
        logsig = self.logsig_head(h)
        sig = torch.exp(logsig)
        if training:
            eps = torch.randn_like(mu)
            z = mu + sig * eps
        else:
            z = mu
        logits = self.decode(z, pad_mask=pad_mask)
        kl_per = 0.5 * (mu.pow(2) + sig.pow(2) - 2.0 * logsig - 1.0)
        if pad_mask is None:
            ce = F.cross_entropy(
                logits.reshape(-1, self.K), token_ids.reshape(-1).long()
            )
            kl = kl_per.mean()
            mu_l2 = mu.pow(2).mean()
        else:
            valid = (~pad_mask).reshape(-1).float()
            ce_per = F.cross_entropy(
                logits.reshape(-1, self.K),
                token_ids.reshape(-1).long(),
                reduction="none",
            )
            ce = (ce_per * valid).sum() / valid.sum().clamp(min=1.0)
            valid_f = (~pad_mask).float().unsqueeze(-1)
            denom = (valid_f.sum() * mu.shape[-1]).clamp(min=1.0)
            kl = (kl_per * valid_f).sum() / denom
            mu_l2 = (mu.pow(2) * valid_f).sum() / denom
        # β-warmup
        warmup = max(1, int(ae.vae_beta_warmup_epochs))
        epoch = float(getattr(self, "_warmup_epoch_seen", 0))
        beta_eff = float(ae.vae_beta) * min(1.0, (epoch + 1.0) / warmup)
        total = ce + beta_eff * kl + ae.latent_l2 * mu_l2
        with torch.no_grad():
            clean_logits = self.decode(mu, pad_mask=pad_mask)
            if pad_mask is None:
                acc = (clean_logits.argmax(dim=-1) == token_ids.long()).float().mean()
                z_norm = mu.norm(dim=-1).mean()
                sig_mean = sig.mean()
            else:
                valid_b = ~pad_mask
                correct = (clean_logits.argmax(dim=-1) == token_ids.long()) & valid_b
                acc = correct.float().sum() / valid_b.float().sum().clamp(min=1.0)
                z_norm = (mu.norm(dim=-1) * valid_b.float()).sum() / valid_b.float().sum().clamp(min=1.0)
                sig_mean = (sig.mean(dim=-1) * valid_b.float()).sum() / valid_b.float().sum().clamp(min=1.0)
        return {
            TRAINING_LOSS_KEY: total,
            "ce": ce.detach(),
            "kl": kl.detach(),
            "beta_eff": torch.tensor(beta_eff),
            "mu_l2": mu_l2.detach(),
            "z_norm": z_norm,
            "sig_mean": sig_mean,
            "tok_acc": acc,
        }


@register("TextAE")
def build_text_ae(cfg: Config) -> TextAutoencoder:
    return TextAutoencoder(cfg)

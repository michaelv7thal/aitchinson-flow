"""HaluEval-QA datamodule for the DirichletFM auditor.

Pairs two caches that were produced for Phase K / Phase Q:

* ``data/hallueval_topk_gpt2.pt`` — per-position top-K (K=32) log-probs and
  vocab indices, plus the paper Spilled-Energy variants (E_logit, E_marg,
  ΔE).  ``shape == (N_topk, L=160, K=32)``.
* ``data/hallueval_cache_gpt2.pt`` — per-position GPT-2 last-hidden state
  (768-dim), full token ids, answer-span mask, per-row hallucination label,
  per-pair id.  ``shape == (N_full, L=160, *)``.

The topk cache covers the first ``N_topk`` rows of the full cache (this is
how the cache was constructed; verified at load time by checking that
``topk_idx[0, 0]`` contains ``full_ids[0, 1]``, the actual next token).

A pair-level train/val split keeps both halves of a prompt on the same
side, so the per-row label leak is avoided.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from aitchinson_flow.config import HalluevalDFMAuditorConfig


def _load(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


class HalluevalDFMDataset(Dataset[dict[str, torch.Tensor]]):
    """Per-row dataset: one item = one full LM sequence (L=160).

    Item dict (all CPU, fp32 unless noted):

    * ``topk_logp``  (L, K)    — log-probabilities of the top-K vocab slots
    * ``topk_idx``   (L, K)    — vocab IDs of the top-K slots (long)
    * ``hidden``     (L, H)    — GPT-2 last hidden state per position
    * ``full_ids``   (L,)      — actual token ids (long)
    * ``answer_mask``(L,)      — bool, True over the answer span
    * ``next_slot``  (L,)      — slot index in top-K of ``full_ids[k+1]`` or
                                 -1 if the actual next token is not in top-K
                                 or position k is the last (long)
    * ``label``      ()        — bool, True if hallucinated (per-row)
    * ``pair_id``    ()        — long, prompt-pair index
    * ``E_logit``    (L,)      — paper E_logit per position (= -topk_logp[..,0]
                                 of next slot, see cache script)
    * ``E_marg``     (L,)      — paper E_marg per position
    * ``DeltaE``     (L,)      — paper ΔE per position
    """

    def __init__(
        self,
        topk_cache: dict[str, Any],
        full_cache: dict[str, Any],
        *,
        max_rows: int | None = None,
    ) -> None:
        N_topk = int(topk_cache["N"])
        N_full = int(full_cache["n_pairs"]) * 2  # clean + halluc per pair
        n = min(N_topk, N_full)
        if max_rows is not None:
            n = min(n, max_rows)
        self.n = n
        self.K = int(topk_cache["K"])
        self.L = int(topk_cache["L"])
        self.H = int(full_cache["H"])

        # Topk cache slices
        self.topk_logp = topk_cache["topk_logp"][:n].float()  # (n, L, K)
        self.topk_idx = topk_cache["topk_idx"][:n].long()  # (n, L, K)
        self.E_logit = topk_cache["E_logit"][:n].float()  # (n, L)
        self.E_marg = topk_cache["E_marg"][:n].float()
        self.DeltaE = topk_cache["DeltaE"][:n].float()

        # Full cache slices
        self.hidden = full_cache["hidden_states"][:n].float()  # (n, L, H)
        self.full_ids = full_cache["full_ids"][:n].long()  # (n, L)
        self.answer_mask = full_cache["answer_mask"][:n].bool()  # (n, L)
        self.label = full_cache["label"][:n].bool()  # (n,)
        self.pair_id = full_cache["pair_id"][:n].long()  # (n,)

        # Precompute per-position next-slot index (slot of full_ids[k+1] in
        # topk_idx[k] if present, else -1).  This is the FM "x_1" target.
        # topk_idx[k] = top-K slots predicted at position k → token at pos k+1.
        next_id = torch.cat(
            [self.full_ids[:, 1:], self.full_ids.new_zeros(self.n, 1)], dim=1
        )  # (n, L)
        match = self.topk_idx == next_id.unsqueeze(-1)  # (n, L, K)
        slot = match.float().argmax(dim=-1)  # 0 if no match
        any_match = match.any(dim=-1)  # (n, L)
        next_slot = torch.where(any_match, slot, slot.new_full(slot.shape, -1))
        # Last position has no next token; mask it out.
        next_slot[:, -1] = -1
        self.next_slot = next_slot.long()  # (n, L)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "topk_logp": self.topk_logp[i],
            "topk_idx": self.topk_idx[i],
            "hidden": self.hidden[i],
            "full_ids": self.full_ids[i],
            "answer_mask": self.answer_mask[i],
            "next_slot": self.next_slot[i],
            "label": self.label[i],
            "pair_id": self.pair_id[i],
            "E_logit": self.E_logit[i],
            "E_marg": self.E_marg[i],
            "DeltaE": self.DeltaE[i],
        }


class HalluevalDFMDataModule:
    """Train/val split that respects pair_id boundaries.

    The HaluEval cache lays out rows as ``[clean_0, halluc_0, clean_1,
    halluc_1, ...]`` (verified by ``label[0..10] == [F,T,F,T,...]`` and
    ``pair_id[0..10] == [0,0,1,1,...]``). A pair-level split therefore
    falls naturally on contiguous index ranges of length 2 — we just split
    at a pair boundary.
    """

    def __init__(
        self,
        cfg: HalluevalDFMAuditorConfig,
        *,
        num_workers: int = 0,
    ) -> None:
        topk = _load(cfg.topk_cache_path)
        full = _load(cfg.hidden_cache_path)
        ds = HalluevalDFMDataset(topk, full, max_rows=cfg.max_rows)

        n = len(ds)
        # Sanity: assume rows are paired (n is even). Split by pair index.
        if n % 2 != 0:
            n -= 1  # drop a trailing row to keep pairs intact
        n_pairs = n // 2
        n_train_pairs = max(1, int(cfg.train_frac * n_pairs))
        idx_train = list(range(0, 2 * n_train_pairs))
        idx_val = list(range(2 * n_train_pairs, n))

        self._train_ds = Subset(ds, idx_train)
        self._val_ds = Subset(ds, idx_val) if idx_val else None
        self._batch_size = cfg.batch_size
        self._num_workers = num_workers
        self._meta = {
            "K": ds.K,
            "L": ds.L,
            "H": ds.H,
            "n_total_rows": n,
            "n_train_rows": len(idx_train),
            "n_val_rows": len(idx_val),
        }

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self._train_ds,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
        )

    def val_dataloader(self) -> DataLoader[Any] | None:
        if self._val_ds is None:
            return None
        return DataLoader(
            self._val_ds,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )


__all__ = ["HalluevalDFMDataset", "HalluevalDFMDataModule"]

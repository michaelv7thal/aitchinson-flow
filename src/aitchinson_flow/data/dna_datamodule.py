"""DNA nucleotide sequence datamodule with train/val/test splits + corruption.

Supports two data sources:

1. **Synthetic** (default): generates random nucleotide sequences with configurable
   GC content. No external download required; suitable for smoke tests and
   infrastructure validation.
2. **HuggingFace** (``cfg.dna_dataset.hf_path`` set): loads genomic sequences from
   a HF dataset and chunks them into length-L windows.

Invalid (OOD) sequences are generated via two modes, alternated per batch:

* **Point mutation** — randomly substitute ~``corrupt_rate`` of bases with a
  different base. Models single-nucleotide polymorphisms.
* **Frameshift insertion** — insert random bases at random positions (trimming
  to keep length L). Models insertion-based frameshifts.

The batch keys ``log_x`` and ``log_x_invalid`` are ``(B, L, K-1)`` ILR
Aitchison coordinates, matching the Stage 1 and Stage 2 model interfaces.
``logits`` (one-hot pseudo-logits) are included to enable spilled-energy
computation in benchmark tasks.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from aitchinson_flow.config import Config
from aitchinson_flow.data.transforms.discrete import token_ids_to_features
from aitchinson_flow.training.datamodule import DataModule


DNA_ALPHABET = "ACGT"
DNA_ALPHABET_N = "ACGTN"

CHAR2ID_4: dict[str, int] = {c: i for i, c in enumerate(DNA_ALPHABET)}
CHAR2ID_5: dict[str, int] = {c: i for i, c in enumerate(DNA_ALPHABET_N)}
VOCAB_SIZE_4: int = 4
VOCAB_SIZE_5: int = 5


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def _generate_synthetic_dna(
    n: int,
    L: int,
    *,
    gc_content: float = 0.5,
    vocab_size: int = 4,
    seed: int = 42,
) -> Tensor:
    """Generate ``n`` random DNA sequences of length ``L``.

    Base probabilities: A=(1-gc)/2, C=gc/2, G=gc/2, T=(1-gc)/2.
    When ``vocab_size=5`` the N-base is added with probability 0.01 (re-normalized).
    """
    gen = torch.Generator().manual_seed(seed)
    at = (1.0 - gc_content) / 2.0
    gc = gc_content / 2.0
    probs = torch.tensor([at, gc, gc, at], dtype=torch.float32)
    if vocab_size == 5:
        probs = torch.cat([probs, torch.tensor([0.01])])
        probs = probs / probs.sum()

    ids = torch.multinomial(
        probs.unsqueeze(0).expand(n * L, -1),
        num_samples=1,
        replacement=True,
        generator=gen,
    )
    return ids.view(n, L)


def _load_dna_from_hf(
    hf_path: str,
    hf_name: str | None,
    split: str,
    text_column: str,
    L: int,
    vocab_size: int,
    *,
    trust_remote_code: bool = False,
) -> Tensor:
    """Load genomic sequences from a HuggingFace dataset and chunk into windows."""
    from datasets import load_dataset  # noqa: PLC0415

    char2id = CHAR2ID_5 if vocab_size == 5 else CHAR2ID_4
    ds = load_dataset(hf_path, hf_name, split=split, trust_remote_code=trust_remote_code)

    all_ids: list[int] = []
    for row in ds:
        seq = str(row[text_column]).upper()
        for c in seq:
            if c in char2id:
                all_ids.append(char2id[c])

    if len(all_ids) < L:
        raise ValueError(
            f"DNA HF dataset at {hf_path!r} is too short for L={L}: "
            f"only {len(all_ids)} usable bases after filtering."
        )

    n = len(all_ids) // L
    t = torch.tensor(all_ids[: n * L], dtype=torch.long)
    return t.view(n, L)


# ---------------------------------------------------------------------------
# Corruption functions
# ---------------------------------------------------------------------------


def corrupt_dna_point_mutation(
    token_ids: Tensor,
    *,
    vocab_size: int,
    corrupt_rate: float = 0.15,
    seed: int | None = None,
) -> Tensor:
    """Substitute a random fraction of bases with a different base.

    Mirrors :func:`~aitchinson_flow.data.corruption.corrupt_token_ids` but is
    tuned for small DNA alphabets: guarantees the replacement is a *different*
    base rather than a no-op.
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    mask = torch.rand(token_ids.shape, generator=gen) < corrupt_rate
    replacements = torch.randint(0, vocab_size, token_ids.shape, generator=gen)
    same = replacements == token_ids
    replacements[same] = (replacements[same] + 1) % vocab_size
    return torch.where(mask, replacements, token_ids)


def corrupt_dna_frameshift(
    token_ids: Tensor,
    *,
    vocab_size: int,
    corrupt_rate: float = 0.15,
    seed: int | None = None,
) -> Tensor:
    """Insert random bases at random positions, trimming to preserve length L.

    Simulates frameshift insertion mutations. Each sequence in the batch
    receives ``ceil(corrupt_rate * L)`` independent single-base insertions.
    After insertion the sequence is truncated back to L.
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    bsz, L = token_ids.shape
    n_insert = max(1, int(math.ceil(corrupt_rate * L)))
    out = token_ids.clone()

    for b in range(bsz):
        positions = torch.randperm(L, generator=gen)[:n_insert].sort().values
        seq: list[int] = out[b].tolist()
        for offset, pos in enumerate(positions.tolist()):
            insert_base = int(torch.randint(0, vocab_size, (1,), generator=gen).item())
            seq.insert(pos + offset, insert_base)
        out[b] = torch.tensor(seq[:L], dtype=torch.long)

    return out


# ---------------------------------------------------------------------------
# Dataset + collate
# ---------------------------------------------------------------------------


class DNAWindowDataset(Dataset[dict[str, Tensor]]):
    """Map-style dataset of length-L nucleotide windows → ``{log_x, token_ids}``."""

    def __init__(
        self,
        windows: Tensor,
        *,
        K: int,
        eps: float,
        label_smoothing: float = 0.0,
        transform_mode: str = "ilr",
    ) -> None:
        self._windows = windows
        self._K = K
        self._eps = eps
        self._label_smoothing = label_smoothing
        self._transform_mode = transform_mode

    def __len__(self) -> int:
        return self._windows.shape[0]

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        ids = self._windows[idx]
        log_x = token_ids_to_features(
            ids,
            K=self._K,
            eps=self._eps,
            label_smoothing=self._label_smoothing,
            transform_mode=self._transform_mode,
        )
        return {"log_x": log_x, "token_ids": ids}


class _DNACorruptingCollate:
    """Collate DNA windows and append ``log_x_invalid`` / ``logits`` fields.

    Alternates between point-mutation and frameshift-insertion corruption
    across successive calls so the Stage 2 GP sees both mutation types
    during training.
    """

    def __init__(
        self,
        *,
        K: int,
        vocab_size: int,
        corrupt_rate: float,
        eps: float,
        seed: int,
        label_smoothing: float = 0.0,
        transform_mode: str = "ilr",
    ) -> None:
        self._K = K
        self._vocab = vocab_size
        self._rate = corrupt_rate
        self._eps = eps
        self._seed = seed
        self._label_smoothing = label_smoothing
        self._transform_mode = transform_mode
        self._n_calls = 0

    def _token_ids_to_logits(self, token_ids: Tensor) -> Tensor:
        bsz, seq_len = token_ids.shape
        logits = torch.full((bsz, seq_len, self._vocab), math.log(self._eps))
        logits.scatter_(dim=-1, index=token_ids.unsqueeze(-1), value=0.0)
        return logits

    def __call__(self, samples: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        log_x = torch.stack([s["log_x"] for s in samples], dim=0)
        token_ids = torch.stack([s["token_ids"] for s in samples], dim=0)
        batch: dict[str, Tensor] = {"log_x": log_x, "token_ids": token_ids}
        batch["logits"] = self._token_ids_to_logits(token_ids)

        seed = self._seed + self._n_calls
        self._n_calls += 1

        # Alternate mutation type per batch for diverse invalid coverage.
        if self._n_calls % 2 == 0:
            bad_ids = corrupt_dna_frameshift(
                token_ids, vocab_size=self._vocab, corrupt_rate=self._rate, seed=seed
            )
        else:
            bad_ids = corrupt_dna_point_mutation(
                token_ids, vocab_size=self._vocab, corrupt_rate=self._rate, seed=seed
            )

        rows = [
            token_ids_to_features(
                row,
                K=self._K,
                eps=self._eps,
                label_smoothing=self._label_smoothing,
                transform_mode=self._transform_mode,
            )
            for row in bad_ids
        ]
        batch["token_ids_invalid"] = bad_ids
        batch["log_x_invalid"] = torch.stack(rows, dim=0)
        batch["logits_invalid"] = self._token_ids_to_logits(bad_ids)
        return batch


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class DNADataModule(DataModule):
    """Nucleotide sequence datamodule with real train/val/test loaders.

    When ``cfg.dna_dataset.hf_path`` is set, sequences are loaded from the
    specified HuggingFace dataset. Otherwise, synthetic sequences are generated
    using the configured ``gc_content`` and ``synthetic_seed``.

    Expects ``cfg.dataset.K`` to match the alphabet size:
    - ``K=4`` when ``cfg.dna_dataset.use_n_base=False`` (ACGT)
    - ``K=5`` when ``cfg.dna_dataset.use_n_base=True`` (ACGTN)
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        dcfg = cfg.dna_dataset
        L = cfg.dataset.L
        K = cfg.dataset.K
        vocab_size = VOCAB_SIZE_5 if dcfg.use_n_base else VOCAB_SIZE_4

        if K != vocab_size:
            raise ValueError(
                f"DNADataModule expects cfg.dataset.K == {vocab_size} "
                f"(use_n_base={dcfg.use_n_base}), got K={K}. "
                f"Set cfg.dataset.K=4 for ACGT or K=5 for ACGTN."
            )

        if dcfg.hf_path is not None:
            train_windows = _load_dna_from_hf(
                dcfg.hf_path,
                dcfg.hf_name,
                dcfg.split_train,
                dcfg.text_column,
                L,
                vocab_size,
                trust_remote_code=dcfg.trust_remote_code,
            )
            val_split = dcfg.split_val or dcfg.split_train
            val_windows = _load_dna_from_hf(
                dcfg.hf_path,
                dcfg.hf_name,
                val_split,
                dcfg.text_column,
                L,
                vocab_size,
                trust_remote_code=dcfg.trust_remote_code,
            )
            test_windows = val_windows
        else:
            n_train = dcfg.max_train_windows or 50_000
            n_eval = dcfg.max_eval_windows or 10_000
            train_windows = _generate_synthetic_dna(
                n_train, L, gc_content=dcfg.gc_content, vocab_size=vocab_size, seed=dcfg.synthetic_seed
            )
            val_windows = _generate_synthetic_dna(
                n_eval, L, gc_content=dcfg.gc_content, vocab_size=vocab_size, seed=dcfg.synthetic_seed + 1
            )
            test_windows = _generate_synthetic_dna(
                n_eval, L, gc_content=dcfg.gc_content, vocab_size=vocab_size, seed=dcfg.synthetic_seed + 2
            )

        if dcfg.max_train_windows is not None:
            train_windows = train_windows[: dcfg.max_train_windows]
        if dcfg.max_eval_windows is not None:
            val_windows = val_windows[: dcfg.max_eval_windows]
            test_windows = test_windows[: dcfg.max_eval_windows]

        eps = cfg.hf_dataset.log_simplex_eps
        ls = cfg.hf_dataset.label_smoothing
        tm = cfg.hf_dataset.transform_mode

        self._train_ds = DNAWindowDataset(train_windows, K=K, eps=eps, label_smoothing=ls, transform_mode=tm)
        self._val_ds = DNAWindowDataset(val_windows, K=K, eps=eps, label_smoothing=ls, transform_mode=tm)
        self._test_ds = DNAWindowDataset(test_windows, K=K, eps=eps, label_smoothing=ls, transform_mode=tm)

        self._train_collate = _DNACorruptingCollate(
            K=K,
            vocab_size=vocab_size,
            corrupt_rate=dcfg.train_corrupt_rate,
            eps=eps,
            seed=dcfg.corruption_seed,
            label_smoothing=ls,
            transform_mode=tm,
        )
        self._eval_collate = _DNACorruptingCollate(
            K=K,
            vocab_size=vocab_size,
            corrupt_rate=dcfg.eval_corrupt_rate,
            eps=eps,
            seed=dcfg.corruption_seed + 10_000,
            label_smoothing=ls,
            transform_mode=tm,
        )

    def _loader(self, ds: Dataset[Any], *, shuffle: bool, collate: Any) -> DataLoader[Any]:
        return DataLoader(
            ds,
            batch_size=self._cfg.training.B,
            shuffle=shuffle,
            num_workers=self._cfg.training.num_workers,
            collate_fn=collate,
            pin_memory=self._cfg.training.device.type == "cuda",
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return self._loader(self._train_ds, shuffle=True, collate=self._train_collate)

    def val_dataloader(self) -> DataLoader[Any] | None:
        return self._loader(self._val_ds, shuffle=False, collate=self._eval_collate)

    def test_dataloader(self) -> DataLoader[Any] | None:
        return self._loader(self._test_ds, shuffle=False, collate=self._eval_collate)

    def num_train_samples(self) -> int | None:
        return len(self._train_ds)


__all__ = [
    "DNA_ALPHABET",
    "DNA_ALPHABET_N",
    "CHAR2ID_4",
    "CHAR2ID_5",
    "VOCAB_SIZE_4",
    "VOCAB_SIZE_5",
    "corrupt_dna_point_mutation",
    "corrupt_dna_frameshift",
    "DNAWindowDataset",
    "DNADataModule",
]

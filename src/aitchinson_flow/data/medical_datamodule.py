"""Clinical text datamodule with char-level encoding, train/val/test splits, and corruption.

Supports two data sources:

1. **Synthetic** (default): generates clinical notes from drug/condition/dose
   templates. No external download required; suitable for smoke tests and
   infrastructure validation.
2. **HuggingFace** (``cfg.medical_dataset.hf_path`` set): loads clinical text
   from an HF dataset and chunks it into length-L windows.

The character vocabulary is a 47-character clinical ASCII set (K=47):

    a-z, 0-9, space, .,;:-/()+%

All text is lowercased before encoding. Characters outside the vocabulary
are silently dropped.

Invalid (OOD) sequences are generated via three modes, cycled per batch:

* **Char swap** — swap random adjacent character pairs (simulates typos /
  drug-name misspellings at the character level).
* **Unit corrupt** — replace common dosage unit substrings (``mg``, ``mcg``,
  ``ml``, ``g``) with an incorrect unit.
* **Random replace** — replace a fraction of characters with random vocab
  characters.

Batch keys ``log_x`` and ``log_x_invalid`` are ``(B, L, K-1)`` ILR Aitchison
coordinates. ``logits`` (one-hot pseudo-logits) are included to enable
spilled-energy computation in benchmark tasks.
"""

from __future__ import annotations

import math
import random
import re
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from aitchinson_flow.config import Config
from aitchinson_flow.data.transforms.discrete import token_ids_to_features
from aitchinson_flow.training.datamodule import DataModule


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

_CLINICAL_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 .,;:-/()+%"
CHAR2ID: dict[str, int] = {c: i for i, c in enumerate(_CLINICAL_CHARS)}
ID2CHAR: dict[int, str] = {i: c for c, i in CHAR2ID.items()}
VOCAB_SIZE: int = len(_CLINICAL_CHARS)  # 47

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

_DRUGS = [
    "aspirin", "ibuprofen", "metformin", "lisinopril", "atorvastatin",
    "omeprazole", "amlodipine", "metoprolol", "losartan", "gabapentin",
    "sertraline", "fluoxetine", "prednisone", "furosemide", "tramadol",
    "amoxicillin", "azithromycin", "ciprofloxacin", "doxycycline", "warfarin",
    "clopidogrel", "albuterol", "levothyroxine", "simvastatin", "ramipril",
]
_CONDITIONS = [
    "hypertension", "type 2 diabetes", "rheumatoid arthritis", "pneumonia",
    "asthma", "major depression", "anxiety disorder", "migraine", "insomnia",
    "hypothyroidism", "iron deficiency anemia", "gastritis", "acute bronchitis",
    "sinusitis", "atrial fibrillation", "heart failure", "chronic kidney disease",
]
_DOSES = ["5", "10", "20", "25", "40", "50", "100", "200", "500", "1000"]
_UNITS = ["mg", "mcg", "ml", "g", "units", "mg/ml"]
_ROUTES = ["oral", "iv", "topical", "inhaled", "sublingual"]
_FREQS = ["once daily", "twice daily", "three times daily", "as needed", "weekly"]
_TEMPLATES = [
    "patient presents with {condition}. prescribed {drug} {dose}{unit} {freq}.",
    "diagnosis: {condition}. medication: {drug} {dose}{unit} by {route}.",
    "{drug} {dose}{unit} for {condition}. {freq} administration.",
    "chief complaint: {condition}. treatment: {drug} {dose}{unit}.",
    "patient with known {condition} on {drug} {dose}{unit} {freq}.",
    "allergies: none. vitals stable. plan: start {drug} {dose}{unit} for {condition}.",
    "follow-up visit for {condition}. continue {drug} {dose}{unit} {freq}.",
    "clinical note: {condition} managed with {drug} {dose}{unit} {route}.",
]


def _fill_template(template: str, rng: random.Random) -> str:
    return template.format(
        condition=rng.choice(_CONDITIONS),
        drug=rng.choice(_DRUGS),
        dose=rng.choice(_DOSES),
        unit=rng.choice(_UNITS),
        route=rng.choice(_ROUTES),
        freq=rng.choice(_FREQS),
    )


def _text_to_ids(text: str) -> list[int]:
    """Lowercase, filter to clinical vocab, return token id list."""
    return [CHAR2ID[c] for c in text.lower() if c in CHAR2ID]


def _generate_synthetic_medical(n: int, L: int, *, seed: int = 99) -> Tensor:
    """Generate ``n`` synthetic clinical note windows of length ``L``."""
    rng = random.Random(seed)
    all_ids: list[int] = []

    while len(all_ids) < n * L + L:
        template = rng.choice(_TEMPLATES)
        note = _fill_template(template, rng)
        all_ids.extend(_text_to_ids(note))
        all_ids.append(CHAR2ID[" "])  # separator

    total = n * L
    ids = all_ids[:total]
    t = torch.tensor(ids, dtype=torch.long)
    return t.view(n, L)


def _load_medical_from_hf(
    hf_path: str,
    hf_name: str | None,
    split: str,
    text_column: str,
    L: int,
    *,
    trust_remote_code: bool = False,
) -> Tensor:
    """Load clinical text from a HuggingFace dataset and chunk into windows."""
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(hf_path, hf_name, split=split, trust_remote_code=trust_remote_code)

    all_ids: list[int] = []
    for row in ds:
        text = str(row[text_column])
        all_ids.extend(_text_to_ids(text))

    if len(all_ids) < L:
        raise ValueError(
            f"Medical HF dataset at {hf_path!r} is too short for L={L}: "
            f"only {len(all_ids)} usable chars after filtering."
        )

    n = len(all_ids) // L
    t = torch.tensor(all_ids[: n * L], dtype=torch.long)
    return t.view(n, L)


# ---------------------------------------------------------------------------
# Corruption functions
# ---------------------------------------------------------------------------

_UNIT_PATTERN = re.compile(r"(mg/ml|mcg|mg|ml|units|g)")
_UNITS_LIST = ["mg", "mcg", "ml", "g", "units", "mg/ml"]


def corrupt_medical_char_swap(
    token_ids: Tensor,
    *,
    corrupt_rate: float = 0.15,
    seed: int | None = None,
) -> Tensor:
    """Swap random adjacent character pairs within each sequence.

    Simulates drug-name misspellings and transcription errors. For each
    sequence, ``ceil(corrupt_rate * L)`` swap sites are chosen; each site
    exchanges position ``i`` and ``i+1``.
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    bsz, L = token_ids.shape
    n_swap = max(1, int(math.ceil(corrupt_rate * L)))
    out = token_ids.clone()

    for b in range(bsz):
        # Sample swap sites (exclude last position)
        sites = torch.randperm(L - 1, generator=gen)[:n_swap]
        for i in sites.tolist():
            out[b, i], out[b, i + 1] = out[b, i + 1].item(), out[b, i].item()

    return out


def corrupt_medical_unit(
    token_ids: Tensor,
    *,
    corrupt_rate: float = 0.15,
    seed: int | None = None,
) -> Tensor:
    """Replace dosage unit substrings with an incorrect unit.

    Decodes each sequence to a string, finds unit tokens (mg, mcg, ml, g,
    units, mg/ml), replaces them with a randomly chosen different unit, then
    re-encodes. Positions without a unit pattern fall back to random char
    replacement at the given ``corrupt_rate``.
    """
    rng = random.Random(seed)

    bsz, L = token_ids.shape
    out = token_ids.clone()

    for b in range(bsz):
        text = "".join(ID2CHAR.get(int(id_), " ") for id_ in out[b].tolist())
        matches = list(_UNIT_PATTERN.finditer(text))
        if matches:
            for m in matches:
                old_unit = m.group()
                alternatives = [u for u in _UNITS_LIST if u != old_unit]
                new_unit = rng.choice(alternatives)
                start, end = m.start(), m.end()
                # Overwrite the unit chars in token_ids (pad/trim to same length)
                new_ids = _text_to_ids(new_unit)
                span = end - start
                new_ids = (new_ids + [CHAR2ID[" "]] * span)[:span]
                for offset, nid in enumerate(new_ids):
                    if start + offset < L:
                        out[b, start + offset] = nid
        else:
            # No units found: fall back to random replace
            mask = torch.rand(L) < corrupt_rate
            replacements = torch.randint(0, VOCAB_SIZE, (L,))
            out[b] = torch.where(mask, replacements, out[b])

    return out


def corrupt_medical_random(
    token_ids: Tensor,
    *,
    corrupt_rate: float = 0.15,
    seed: int | None = None,
) -> Tensor:
    """Replace a random fraction of chars with random clinical vocab chars."""
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    mask = torch.rand(token_ids.shape, generator=gen) < corrupt_rate
    replacements = torch.randint(0, VOCAB_SIZE, token_ids.shape, generator=gen)
    same = replacements == token_ids
    replacements[same] = (replacements[same] + 1) % VOCAB_SIZE
    return torch.where(mask, replacements, token_ids)


# ---------------------------------------------------------------------------
# Dataset + collate
# ---------------------------------------------------------------------------


class MedicalWindowDataset(Dataset[dict[str, Tensor]]):
    """Map-style dataset of length-L clinical text windows → ``{log_x, token_ids}``."""

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


class _MedicalCorruptingCollate:
    """Collate clinical text windows and append ``log_x_invalid`` / ``logits``.

    Cycles through three corruption modes (char-swap, unit-corrupt, random-replace)
    across successive calls so the GP sees a diverse range of clinical errors.
    """

    _MODES = ("char_swap", "unit_corrupt", "random_replace")

    def __init__(
        self,
        *,
        K: int,
        corrupt_rate: float,
        eps: float,
        seed: int,
        label_smoothing: float = 0.0,
        transform_mode: str = "ilr",
    ) -> None:
        self._K = K
        self._rate = corrupt_rate
        self._eps = eps
        self._seed = seed
        self._label_smoothing = label_smoothing
        self._transform_mode = transform_mode
        self._n_calls = 0

    def _token_ids_to_logits(self, token_ids: Tensor) -> Tensor:
        bsz, seq_len = token_ids.shape
        logits = torch.full((bsz, seq_len, VOCAB_SIZE), math.log(self._eps))
        logits.scatter_(dim=-1, index=token_ids.unsqueeze(-1), value=0.0)
        return logits

    def __call__(self, samples: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        log_x = torch.stack([s["log_x"] for s in samples], dim=0)
        token_ids = torch.stack([s["token_ids"] for s in samples], dim=0)
        batch: dict[str, Tensor] = {"log_x": log_x, "token_ids": token_ids}
        batch["logits"] = self._token_ids_to_logits(token_ids)

        seed = self._seed + self._n_calls
        mode = self._MODES[self._n_calls % len(self._MODES)]
        self._n_calls += 1

        if mode == "char_swap":
            bad_ids = corrupt_medical_char_swap(token_ids, corrupt_rate=self._rate, seed=seed)
        elif mode == "unit_corrupt":
            bad_ids = corrupt_medical_unit(token_ids, corrupt_rate=self._rate, seed=seed)
        else:
            bad_ids = corrupt_medical_random(token_ids, corrupt_rate=self._rate, seed=seed)

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


class MedicalDataModule(DataModule):
    """Clinical text datamodule with real train/val/test loaders.

    When ``cfg.medical_dataset.hf_path`` is set, clinical text is loaded from
    the specified HuggingFace dataset. Otherwise, synthetic notes are generated
    from clinical templates.

    Expects ``cfg.dataset.K == VOCAB_SIZE`` (47).
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        mcfg = cfg.medical_dataset
        L = cfg.dataset.L
        K = cfg.dataset.K

        if K != VOCAB_SIZE:
            raise ValueError(
                f"MedicalDataModule expects cfg.dataset.K == {VOCAB_SIZE} "
                f"(clinical ASCII vocab), got K={K}."
            )

        if mcfg.hf_path is not None:
            train_windows = _load_medical_from_hf(
                mcfg.hf_path, mcfg.hf_name, mcfg.split_train,
                mcfg.text_column, L, trust_remote_code=mcfg.trust_remote_code,
            )
            val_split = mcfg.split_val or mcfg.split_train
            val_windows = _load_medical_from_hf(
                mcfg.hf_path, mcfg.hf_name, val_split,
                mcfg.text_column, L, trust_remote_code=mcfg.trust_remote_code,
            )
            test_windows = val_windows
        else:
            n_train = mcfg.max_train_windows or 10_000
            n_eval = mcfg.max_eval_windows or 2_000
            train_windows = _generate_synthetic_medical(n_train, L, seed=mcfg.synthetic_seed)
            val_windows = _generate_synthetic_medical(n_eval, L, seed=mcfg.synthetic_seed + 1)
            test_windows = _generate_synthetic_medical(n_eval, L, seed=mcfg.synthetic_seed + 2)

        if mcfg.max_train_windows is not None:
            train_windows = train_windows[: mcfg.max_train_windows]
        if mcfg.max_eval_windows is not None:
            val_windows = val_windows[: mcfg.max_eval_windows]
            test_windows = test_windows[: mcfg.max_eval_windows]

        eps = cfg.hf_dataset.log_simplex_eps
        ls = cfg.hf_dataset.label_smoothing
        tm = cfg.hf_dataset.transform_mode

        self._train_ds = MedicalWindowDataset(train_windows, K=K, eps=eps, label_smoothing=ls, transform_mode=tm)
        self._val_ds = MedicalWindowDataset(val_windows, K=K, eps=eps, label_smoothing=ls, transform_mode=tm)
        self._test_ds = MedicalWindowDataset(test_windows, K=K, eps=eps, label_smoothing=ls, transform_mode=tm)

        self._train_collate = _MedicalCorruptingCollate(
            K=K,
            corrupt_rate=mcfg.train_corrupt_rate,
            eps=eps,
            seed=mcfg.corruption_seed,
            label_smoothing=ls,
            transform_mode=tm,
        )
        self._eval_collate = _MedicalCorruptingCollate(
            K=K,
            corrupt_rate=mcfg.eval_corrupt_rate,
            eps=eps,
            seed=mcfg.corruption_seed + 10_000,
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
    "CHAR2ID",
    "ID2CHAR",
    "VOCAB_SIZE",
    "corrupt_medical_char_swap",
    "corrupt_medical_unit",
    "corrupt_medical_random",
    "MedicalWindowDataset",
    "MedicalDataModule",
]

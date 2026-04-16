import torch

from aitchinson_flow.data.text8_datamodule import (
    CHAR2ID,
    chunk_text8_to_ids,
    _load_text8_chars,
)
from benchmarks.corruption import build_invalid_batch

data = _load_text8_chars(split="train")
ids = chunk_text8_to_ids(data, L=40)
token_ids = ids[:8].clone()
vocab_size = 27
eps = 1e-8
logits = torch.full((token_ids.shape[0], token_ids.shape[1], vocab_size), torch.log(torch.tensor(eps)))
logits.scatter_(dim=-1, index=token_ids.unsqueeze(-1), value=0.0)

batch = {"token_ids": token_ids, "logits": logits}
build_invalid_batch(
    batch,
    K=vocab_size,
    corrupt_rate=0.30,
    order_mix_rate=0.30,
    order_mix_prob=0.5,
    eps=eps,
    seed=0,
)

id2char = {idx: ch for ch, idx in CHAR2ID.items()}


def decode(seq: torch.Tensor) -> str:
    return "".join(id2char[int(t)] for t in seq)


for i in range(batch["token_ids"].shape[0]):
    clean = decode(batch["token_ids"][i])
    corrupt = decode(batch["token_ids_invalid"][i])
    print(f"[{i}] clean   : {clean}")
    print(f"[{i}] corrupt : {corrupt}")
    print("-" * 80)

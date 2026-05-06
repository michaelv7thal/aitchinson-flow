"""Quick evaluation: load a checkpoint, sample, report unigram/bigram KL + sample text.

Usage: python scripts/quick_eval.py [path_to_checkpoint]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F

import aitchinson_flow.models  # noqa: F401 — populate registry
from aitchinson_flow.config import Config
from aitchinson_flow.data.char_window_dataset import CHAR2ID, VOCAB_SIZE
from aitchinson_flow.models import build_model
from aitchinson_flow.training import build_training_datamodule

ALPHABET = "".join(sorted(CHAR2ID, key=CHAR2ID.__getitem__))
K = VOCAB_SIZE


def ids_to_text(ids: torch.Tensor) -> list[str]:
    return ["".join(ALPHABET[int(i)] for i in row) for row in ids.cpu()]


def unigram_kl(gen_ids: torch.Tensor, ref_ids: torch.Tensor) -> tuple[float, float, float]:
    gen = torch.zeros(K).scatter_add_(0, gen_ids.reshape(-1), torch.ones_like(gen_ids.reshape(-1), dtype=torch.float))
    ref = torch.zeros(K).scatter_add_(0, ref_ids.reshape(-1), torch.ones_like(ref_ids.reshape(-1), dtype=torch.float))
    gen = (gen + 1e-9) / (gen.sum() + K * 1e-9)
    ref = (ref + 1e-9) / (ref.sum() + K * 1e-9)
    kl = float((gen * (gen.log() - ref.log())).sum())
    H_gen = float(-(gen * gen.log()).sum())
    H_ref = float(-(ref * ref.log()).sum())
    return kl, H_gen, H_ref


def bigram_kl(gen_ids: torch.Tensor, ref_ids: torch.Tensor) -> float:
    def counts(ids: torch.Tensor) -> torch.Tensor:
        c = torch.zeros(K, K)
        flat = ids.reshape(-1, ids.shape[-1])
        for row in flat:
            for a, b in zip(row[:-1], row[1:]):
                c[int(a), int(b)] += 1
        return c
    gc = counts(gen_ids)
    rc = counts(ref_ids)
    gp = (gc + 1e-6) / (gc.sum() + K * K * 1e-6)
    rp = (rc + 1e-6) / (rc.sum() + K * K * 1e-6)
    return float((gp * (gp.log() - rp.log())).sum())


def main() -> None:
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/epoch_final.pt"
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    print(f"checkpoint={ckpt_path}  n_samples={n_samples}  steps={n_steps}")

    cfg = Config()
    device = cfg.training.device
    model = build_model(cfg).to(device)

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()
    print(f"loaded epoch={payload.get('epoch', '?')}  global_step={payload.get('global_step', '?')}")

    dm, _ = build_training_datamodule(cfg)
    train_ids = dm.splits.train.long()
    val_ids = dm.splits.val.long()
    L = cfg.text8_dataset.L

    # Sample.
    with torch.no_grad():
        x = model.sample(n_samples, L, max_steps=n_steps)
        log_probs = model.decode_to_logprobs(x)
    gen_ids = log_probs.argmax(-1).cpu()

    kl_u, H_gen, H_ref = unigram_kl(gen_ids, train_ids)
    kl_b = bigram_kl(gen_ids, train_ids)
    print(f"\n=== Argmax decode ({n_samples} samples) ===")
    print(f"unigram_kl={kl_u:.4f}  H_gen={H_gen:.3f}  H_ref={H_ref:.3f}  ratio={H_gen/H_ref:.3f}")
    print(f"bigram_kl ={kl_b:.4f}")

    # Top-k decode for variety.
    k_topk = 3
    topv, topi = log_probs.topk(k_topk, dim=-1)
    sampled_topk = torch.gather(topi, -1, torch.distributions.Categorical(logits=topv).sample().unsqueeze(-1)).squeeze(-1).cpu()
    kl_u2, H_gen2, _ = unigram_kl(sampled_topk, train_ids)
    kl_b2 = bigram_kl(sampled_topk, train_ids)
    print(f"\n=== Top-{k_topk} decode ===")
    print(f"unigram_kl={kl_u2:.4f}  H_gen={H_gen2:.3f}  ratio={H_gen2/H_ref:.3f}")
    print(f"bigram_kl ={kl_b2:.4f}")

    # Gradient norms — proxy for "on-manifold-ness".
    val_features = dm._val_ds._features[: min(64, len(dm._val_ds))].to(device)
    grad_gt = model.position_uncertainty(val_features).mean().item()
    grad_gen = model.position_uncertainty(x[:64]).mean().item()
    print(f"\n=== Gradient norm (lower = on energy minimum) ===")
    print(f"||grad|| at GT  : {grad_gt:.4f}")
    print(f"||grad|| at gen : {grad_gen:.4f}")

    # Print some samples.
    print(f"\n=== 8 argmax samples ===")
    for s in ids_to_text(gen_ids[:8]):
        print(f"  '{s}'")
    print(f"\n=== 8 top-{k_topk} samples ===")
    for s in ids_to_text(sampled_topk[:8]):
        print(f"  '{s}'")


if __name__ == "__main__":
    main()

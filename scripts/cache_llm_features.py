"""E6a — cache LLM features (token_ids, top-k logits, last hidden state) to disk.

Precomputes per-example features under a causal LM (default Qwen2.5-1.5B, 4-bit
nf4 when CUDA + bitsandbytes are available) over a WikiText-2 subset and
memory-maps them to disk, so the LLM is **never loaded again** during the
auditor-parity training (E6b) — the cached tensors fit any GPU trivially.

Output layout under ``--out``:
  token_ids.dat   (N, T) int32      padded token ids
  lengths.dat     (N,)   int32      true length per example
  topk_logits.dat (N, T, k) float16 top-k next-token logits
  topk_idx.dat    (N, T, k) int32   their vocab indices
  hidden.dat      (N, T, H) float16 last hidden state
  features_manifest.json            shapes, dtypes, dtype-string, model, k

Usage:
  python scripts/cache_llm_features.py --model Qwen/Qwen2.5-1.5B --n 2000 --4bit --out runs/llm_cache/wikitext2
  python scripts/cache_llm_features.py --smoke      # mocked tiny model, no download
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.append(str(_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def _write_memmaps(out: Path, examples: list[dict], *, T: int, k: int, H: int,
                   model_name: str) -> dict:
    """examples: list of {token_ids:(t,), topk_logits:(t,k), topk_idx:(t,k),
    hidden:(t,H)} (variable t<=T). Pads to T and writes the memmaps + manifest."""
    out.mkdir(parents=True, exist_ok=True)
    N = len(examples)
    arrs = {
        "token_ids": (np.memmap(out / "token_ids.dat", dtype="int32", mode="w+", shape=(N, T)), 0),
        "lengths":   (np.memmap(out / "lengths.dat",   dtype="int32", mode="w+", shape=(N,)), 0),
        "topk_logits": (np.memmap(out / "topk_logits.dat", dtype="float16", mode="w+", shape=(N, T, k)), 0.0),
        "topk_idx":  (np.memmap(out / "topk_idx.dat",  dtype="int32", mode="w+", shape=(N, T, k)), 0),
        "hidden":    (np.memmap(out / "hidden.dat",    dtype="float16", mode="w+", shape=(N, T, H)), 0.0),
    }
    for mm, fill in arrs.values():
        mm[:] = fill
    tid, leng, tkl, tki, hid = (arrs[k2][0] for k2 in
                                ("token_ids", "lengths", "topk_logits", "topk_idx", "hidden"))
    for i, ex in enumerate(examples):
        t = int(ex["token_ids"].shape[0])
        leng[i] = t
        tid[i, :t] = ex["token_ids"].astype("int32")
        tkl[i, :t] = ex["topk_logits"].astype("float16")
        tki[i, :t] = ex["topk_idx"].astype("int32")
        hid[i, :t] = ex["hidden"].astype("float16")
    for mm, _ in arrs.values():
        mm.flush()
    manifest = {
        "model": model_name, "n": N, "T": T, "k": k, "H": H,
        "arrays": {
            "token_ids": {"shape": [N, T], "dtype": "int32"},
            "lengths": {"shape": [N], "dtype": "int32"},
            "topk_logits": {"shape": [N, T, k], "dtype": "float16"},
            "topk_idx": {"shape": [N, T, k], "dtype": "int32"},
            "hidden": {"shape": [N, T, H], "dtype": "float16"},
        },
    }
    (out / "features_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _texts(dataset: str, n: int) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset(dataset, "wikitext-2-raw-v1", split="train") \
        if "wikitext" in dataset else load_dataset(dataset, split="train")
    out = [t for t in ds["text"] if t and len(t.strip()) > 32][:n]
    return out


def _load_lm(model_name: str, use_4bit: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw: dict = {"output_hidden_states": True}
    if use_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        kw["device_map"] = "auto"
    else:
        kw["torch_dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw).eval()
    return model, tok


@torch.no_grad()
def cache(model, tok, texts: list[str], out: Path, *, T: int, k: int,
         model_name: str) -> dict:
    device = next(model.parameters()).device
    H = model.config.hidden_size
    examples = []
    for text in texts:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=T)
        ids = enc["input_ids"].to(device)
        o = model(ids, output_hidden_states=True)
        logits = o.logits[0].float()           # (t, V)
        hidden = o.hidden_states[-1][0].float()  # (t, H)
        kk = min(k, logits.shape[-1])
        tl, ti = torch.topk(logits, kk, dim=-1)
        examples.append({
            "token_ids": ids[0].cpu().numpy(),
            "topk_logits": tl.cpu().numpy(),
            "topk_idx": ti.cpu().numpy(),
            "hidden": hidden.cpu().numpy(),
        })
    return _write_memmaps(out, examples, T=T, k=k, H=H, model_name=model_name)


class _MockLM:
    """Tiny stand-in with the minimal HF-causal-LM surface used by cache()."""
    class _Cfg:
        hidden_size = 16
        vocab_size = 50

    def __init__(self):
        self.config = self._Cfg()

    def parameters(self):
        yield torch.zeros(1)

    def __call__(self, ids, output_hidden_states=True):
        t = ids.shape[1]

        class _Out:
            pass
        o = _Out()
        o.logits = torch.randn(1, t, self._Cfg.vocab_size)
        o.hidden_states = [torch.randn(1, t, self._Cfg.hidden_size)]
        return o

    def eval(self):
        return self


class _MockTok:
    def __call__(self, text, return_tensors=None, truncation=True, max_length=32):
        t = min(max_length, max(4, len(text.split())))
        return {"input_ids": torch.randint(0, 50, (1, t))}


def _smoke() -> None:
    import tempfile
    out = Path(tempfile.mkdtemp()) / "cache"
    texts = ["the quick brown fox jumps over the lazy dog",
             "machine learning models cache features to disk",
             "all spaces and structure"]
    man = cache(_MockLM().eval(), _MockTok(), texts, out, T=16, k=5,
                model_name="mock")
    # reload + sanity
    tid = np.memmap(out / "token_ids.dat", dtype="int32", mode="r",
                    shape=tuple(man["arrays"]["token_ids"]["shape"]))
    hid = np.memmap(out / "hidden.dat", dtype="float16", mode="r",
                    shape=tuple(man["arrays"]["hidden"]["shape"]))
    assert (out / "features_manifest.json").exists()
    assert tid.shape == (3, 16) and hid.shape == (3, 16, 16), (tid.shape, hid.shape)
    print(f"OK cache_llm_features smoke: wrote+reloaded memmaps "
          f"token_ids{tid.shape} hidden{hid.shape} k={man['k']} at {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--dataset", type=str, default="wikitext")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--out", type=str, default="runs/llm_cache/wikitext2")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
        return
    model, tok = _load_lm(args.model, args.four_bit)
    texts = _texts(args.dataset, args.n)
    man = cache(model, tok, texts, Path(args.out), T=args.max_len,
                k=args.topk, model_name=args.model)
    print(f"cached {man['n']} examples → {args.out} (T={man['T']}, k={man['k']}, H={man['H']})")


if __name__ == "__main__":
    main()

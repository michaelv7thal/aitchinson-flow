from __future__ import annotations

from aitchinson_flow.models import LossDict


def detach_means(outputs: LossDict) -> dict[str, float]:
    """Turn a single-step LossDict into loggable scalars."""
    out: dict[str, float] = {}
    for k, v in outputs.items():
        if v.ndim == 0:
            out[k] = float(v.detach().cpu())
        else:
            out[k] = float(v.detach().mean().cpu())

    return out


def running_average(
    agg: dict[str, float], counts: dict[str, int], step: LossDict
) -> None:
    """In-place Welford-style running sum for means over a batch loop."""
    for k, v in step.items():
        val = float(v.detach().mean().cpu()) if v.ndim > 0 else float(v.detach().cpu())
        agg[k] = agg.get(k, 0.0) + val
        counts[k] = counts.get(k, 0) + 1


def finalize_averages(
    agg: dict[str, float], counts: dict[str, int]
) -> dict[str, float]:
    """Convert running sums to means."""
    return {k: agg[k] / counts[k] for k in agg}

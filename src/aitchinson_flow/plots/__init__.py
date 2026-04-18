"""Stage-aware diagnostic plotting for the Bayesian Auditor pipeline."""

from aitchinson_flow.plots.plots import (
    StagePlotData,
    plot_histogram,
    plot_latent_density,
    plot_loss_curves,
    plot_token_heatmap,
    save_stage_plots,
)

__all__ = [
    "StagePlotData",
    "plot_histogram",
    "plot_latent_density",
    "plot_loss_curves",
    "plot_token_heatmap",
    "save_stage_plots",
]

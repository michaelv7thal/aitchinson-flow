"""Stage-aware diagnostic plotting for the Bayesian Auditor pipeline."""

from aitchinson_flow.plots.plots import (
    StagePlotData,
    plot_benchmark_table,
    plot_histogram,
    plot_latent_density,
    plot_loss_curves,
    plot_token_heatmap,
    save_benchmark_plots,
    save_stage_plots,
    stage_plot_data_from_scores,
)

__all__ = [
    "StagePlotData",
    "plot_benchmark_table",
    "plot_histogram",
    "plot_latent_density",
    "plot_loss_curves",
    "plot_token_heatmap",
    "save_benchmark_plots",
    "save_stage_plots",
    "stage_plot_data_from_scores",
]

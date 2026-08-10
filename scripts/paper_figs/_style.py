"""Shared style for the capstone-paper figures.

Palette: dataviz reference categorical palette, CVD-validated on a white
surface (adjacent pairs for the 7-detector set; all-pairs for the
schematic set; ordinal checks for the blue budget ramp).
Typography: cmr10 + mathtext-cm to match the paper's Computer Modern.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PAPER_FIGDIR = Path("/home/michael/projects/capstone-paper/figures")
PREVIEW_DIR = Path(
    "/tmp/claude-1000/-home-michael-projects-capstone-paper/"
    "30598ee3-292b-4d58-ad61-94cfc3b31d4e/scratchpad/figpreview"
)

# validated categorical slots (light mode)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"

# chrome / ink
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# ordinal blue ramp: light -> dark = small -> large training budget
BLUES_ORDINAL = ["#86b6ef", "#2a78d6", "#104281"]

# semantic roles for the two schematics (all-pairs validated)
ROLE_DATA = BLUE
ROLE_FIELD = ORANGE
ROLE_SHELL = AQUA
ROLE_ATTRACTOR = VIOLET

TEXTWIDTH_IN = 5.90  # \textwidth = 15.0 cm on the paper's A4 geometry

# fixed detector order + colors, shared by both OOD figures
DETECTORS = [
    # (paper label, subdir/file, key suffix, color)
    ("NLL", "nll/denoiser_nll_sweep.json", "nll", BLUE),
    ("BLR", "blr/bayes_linear_sweep.json", "energy", ORANGE),
    (r"$\mathrm{BLR}_{\mathrm{all}}$", "blr_adv/bayes_linear_adv_sweep.json", "energy", AQUA),
    (r"$\mathrm{BLR}_{\mathrm{fi}}$", "blr_fi/bayes_linear_fi_sweep.json", "energy", YELLOW),
    ("BGMM", "bgmm/bgmm_perpos_sweep.json", "gmm", MAGENTA),
    ("GPT-2 SE", "gpt2_se/gpt2_spilled_energy_sweep.json", "se", GREEN),
    ("GPT-2 NLL", "gpt2_nll/gpt2_nll_sweep.json", "se", VIOLET),
]


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "axes.formatter.use_mathtext": True,
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.7,
            "xtick.color": BASELINE,
            "ytick.color": BASELINE,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "lines.solid_joinstyle": "round",
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "legend.columnspacing": 1.0,
            "legend.handletextpad": 0.5,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(fig, name: str):
    """Write the vector PDF into the paper and a PNG preview for QA."""
    PAPER_FIGDIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    pdf = PAPER_FIGDIR / f"{name}.pdf"
    png = PREVIEW_DIR / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    print(f"wrote {pdf} and preview {png}")

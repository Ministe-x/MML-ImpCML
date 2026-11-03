"""Figure style: figures are created at their printed size (3.25 in per column, 6.75 in full width)
with 10 pt text, the size of the main text, and only Type-1 fonts (LaTeX rendering) when available.
They use constrained layout and are saved without cropping, so they are included in the paper unscaled."""
import io
import shutil

import matplotlib.pyplot as plt

from .data import FIGURES

COLUMN, FULL = 3.25, 6.75
COLORS = {20: "#0072B2", 30: "#E69F00", 40: "#009E73", "adapt": "#CC79A7", "gray": "#555555"}


def use_paper_style():
    base = {"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10, "legend.fontsize": 10,
            "xtick.labelsize": 10, "ytick.labelsize": 10, "lines.linewidth": 1.5, "lines.markersize": 4,
            "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
            "figure.constrained_layout.use": True, "figure.constrained_layout.h_pad": 0.02,
            "figure.constrained_layout.w_pad": 0.02, "savefig.bbox": None, "pdf.fonttype": 42}
    plt.rcParams.update(base)
    if shutil.which("latex"):
        plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                             "text.latex.preamble": r"\usepackage{mathptmx}\usepackage{amsmath}"})
        try:                                   # LaTeX needs the cm-super fonts for Matplotlib
            fig = plt.figure(figsize=(1, 1)); fig.text(0.5, 0.5, r"$n=40$"); fig.savefig(io.BytesIO(), format="pdf")
            plt.close(fig)
            return "usetex"
        except Exception:
            plt.close("all")
            plt.rcParams.update({"text.usetex": False})
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"]})
    return "matplotlib"


def pct(s):
    """Escape % for LaTeX-rendered labels."""
    return s.replace("%", r"\%") if plt.rcParams["text.usetex"] else s


def save(fig, name):
    """Save at the exact figure size (no bbox cropping), so that the PDF is included at 100% scale."""
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=200)

"""Helpers shared by the marimo notebooks in this folder.

The marimo notebooks re-implement notebooks/01-08 on top of the `krypto` package.  They never overwrite
the outputs of the Jupyter notebooks: everything they write (JSON results, runtime log, figures and
hidden-set predictions) goes to results_marimo/.  Expensive experiments run only when their button is
pressed; otherwise their results are read from results_marimo/ if present, else from the committed results/.
"""
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ORIGINAL = ROOT / "results"          # written by the Jupyter notebooks
OUT = ROOT / "results_marimo"        # written by the marimo notebooks
FIGURES = OUT / "figures"
HIDDEN = OUT / "hiddenlabels"
LOG = OUT / "runtime_log.json"


def save_json(obj, name):
    """Write `obj` to results_marimo/<name> and return it unchanged."""
    OUT.mkdir(exist_ok=True)
    with open(OUT / name, "w") as f:
        json.dump(obj, f, indent=1)
    return obj


def load_cached(name):
    """(data, folder) from results_marimo/ if present, else from results/; (None, None) if neither exists."""
    for folder in (OUT, ORIGINAL):
        if (folder / name).exists():
            with open(folder / name) as f:
                return json.load(f), folder.name
    return None, None


def load_json(name):
    return load_cached(name)[0]


def cached_or_run(button, name, compute):
    """Run `compute()` and save its JSON-shaped result when `button` was pressed, else load the cached result.

    Returns (data, provenance) where provenance says where the numbers come from.
    """
    if button.value:
        return save_json(compute(), name), "recomputed in this session"
    data, folder = load_cached(name)
    return data, (f"cached in {folder}/{name}" if data is not None else None)


@contextmanager
def track(name):
    """Log the wall-clock time of an experiment to results_marimo/runtime_log.json."""
    start = time.time()
    yield
    entry = {"name": name, "seconds": round(time.time() - start, 2)}
    OUT.mkdir(exist_ok=True)
    log = json.load(open(LOG)) if LOG.exists() else []
    log = [e for e in log if e["name"] != name] + [entry]
    json.dump(log, open(LOG, "w"), indent=1)
    print(f"[track] {name}: {entry['seconds']:.1f} s")


def runtime_log():
    """Runtime entries of all experiments: the Jupyter log, overridden by any marimo re-runs of the same name."""
    merged = {}
    for path in (ORIGINAL / "runtime_log.json", LOG):
        if path.exists():
            merged.update({e["name"]: e for e in json.load(open(path))})
    return list(merged.values())


def save_figure(fig, name):
    """Save a figure as PDF (for the report) and PNG (for preview) in results_marimo/figures/."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=200)
    return fig


def pm(values, scale=100, digits=1):
    """'mean ± 95% CI' over repetitions, as a string."""
    from krypto.stats import mean_ci
    m, h = mean_ci(values)
    return f"{scale * m:.{digits}f} ± {scale * h:.{digits}f}"

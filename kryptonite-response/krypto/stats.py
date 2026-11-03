"""Summary statistics over repeated splits/seeds."""
import json

import numpy as np
from scipy import stats

from .data import RESULTS


def mean_ci(values, level=0.95):
    """Mean and half-width of the Student-t confidence interval over independent repetitions."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float(v.mean()), 0.0
    half = stats.t.ppf(0.5 + level / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(half)


def binomial_ci(acc, n, level=0.95):
    """Normal-approximation CI half-width for an accuracy measured on n test points."""
    return float(stats.norm.ppf(0.5 + level / 2) * np.sqrt(acc * (1 - acc) / n))


def fmt(values, scale=100, digits=1):
    m, h = mean_ci(values)
    return f"{scale * m:.{digits}f} ± {scale * h:.{digits}f}"


def save_json(obj, name):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / name, "w") as f:
        json.dump(obj, f, indent=1)


def load_json(name):
    path = RESULTS / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

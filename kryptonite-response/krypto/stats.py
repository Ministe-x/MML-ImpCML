"""Summaries over the repeated splits, and reading/writing the JSON results.

Every number in the paper is a mean over five splits with a 95% Student-t interval.  With only five
repetitions the t quantile (2.78) is noticeably larger than the normal one (1.96), so the intervals are
honest about how few repetitions we have.
"""
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
    """Half-width of the normal-approximation interval for an accuracy measured on n test points.

    This is the +-3.5 points quoted in the paper for 800 test points at 50%, against which the mock
    paper's "worse than random" 46-50% should be read.
    """
    return float(stats.norm.ppf(0.5 + level / 2) * np.sqrt(acc * (1 - acc) / n))


def fmt(values, scale=100, digits=1):
    """Format repetitions as 'mean ± half-width' in percent, for printing."""
    m, h = mean_ci(values)
    return f"{scale * m:.{digits}f} ± {scale * h:.{digits}f}"


def save_json(obj, name):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / name, "w") as f:
        json.dump(obj, f, indent=1)


def load_json(name):
    """Load results/<name>, or return None if the notebook that writes it has not been run yet."""
    path = RESULTS / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

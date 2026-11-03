"""Data loading and the fixed train/validation/test splits used by every experiment.

All experiments on Kryptonite-n use the SAME five random 60/20/20 splits (seeds 0-4), so
that every model is compared on identical held-out test points (paired comparisons).
The test split of a seed is never used for fitting, early stopping or model selection.
"""
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Datasets"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SEEDS = (0, 1, 2, 3, 4)
NS = (20, 30, 40)
TARGET = 0.90


def load(name):
    """Load a labelled dataset, e.g. load("kryptonite-20") or load("kryptonite-40-adapt")."""
    return np.load(DATA / f"{name}-X.npy"), np.load(DATA / f"{name}-y.npy")


def load_hidden(name):
    """Load an unlabelled hidden set, e.g. load_hidden("kryptonite-40-adapt")."""
    return np.load(DATA / f"hidden-{name}-X.npy")


def split(X, y, seed, val=0.2, test=0.2):
    """Stratified train/validation/test split (60/20/20 by default)."""
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=val + test, random_state=seed, stratify=y)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=test / (val + test),
                                              random_state=seed, stratify=y_tmp)
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def holdout(X, y, seed, frac=0.2):
    """Stratified two-way split (fit / hold-out), used when training final models on all labelled data."""
    X_fit, X_hold, y_fit, y_hold = train_test_split(X, y, test_size=frac, random_state=seed, stratify=y)
    return X_fit, y_fit, X_hold, y_hold


def mock_split(X, y):
    """The exact split of the mock paper's notebook (60/20/20, random_state=42, not stratified)."""
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def bits(X):
    """Round features to bits b = sign(x) in {-1, +1}.  Every |x| > 0.5, so this is lossless for the label."""
    return np.where(X < 0, -1.0, 1.0)


def codes(X, S):
    """Integer code of the bit pattern of X restricted to the coordinates S (bit i set <=> x_{S[i]} < 0)."""
    S = list(S)
    return ((np.asarray(X)[:, S] < 0).astype(np.int64) << np.arange(len(S))).sum(1)

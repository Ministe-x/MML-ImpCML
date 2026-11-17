"""Loading the datasets and the train/validation/test splits shared by all experiments.

We fix five random 60/20/20 splits (seeds 0-4) once and reuse them for every model on Kryptonite-n.
Because all models are then scored on exactly the same test points, a difference between two models
reflects the models rather than a lucky split, and the comparisons in the paper are paired.  The test
part of a split is only ever used for the final score: fitting, early stopping and every choice of
hyper-parameter happen on the training and validation parts.
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
TARGET = 0.90                       # the accuracy the mock paper asks for


def load(name):
    """Load a labelled dataset, e.g. load("kryptonite-20") or load("kryptonite-40-adapt")."""
    return np.load(DATA / f"{name}-X.npy"), np.load(DATA / f"{name}-y.npy")


def load_hidden(name):
    """Load the unlabelled hidden set we submit predictions for, e.g. load_hidden("kryptonite-40-adapt")."""
    return np.load(DATA / f"hidden-{name}-X.npy")


def split(X, y, seed, val=0.2, test=0.2):
    """Stratified train/validation/test split, 60/20/20 by default (Adapt uses val=0.16, i.e. 64/16/20)."""
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=val + test, random_state=seed, stratify=y)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=test / (val + test),
                                              random_state=seed, stratify=y_tmp)
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def holdout(X, y, seed, frac=0.2):
    """Two-way split for the final models: most data to fit, a held-out part for model selection."""
    X_fit, X_hold, y_fit, y_hold = train_test_split(X, y, test_size=frac, random_state=seed, stratify=y)
    return X_fit, y_fit, X_hold, y_hold


def mock_split(X, y):
    """The split used in the mock paper's notebook (60/20/20, random_state=42, not stratified).

    We only use it to reproduce the mock paper's numbers exactly; our own experiments use `split`.
    """
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def bits(X):
    """Round each feature to its bit b = sign(x) in {-1, +1}.

    Every feature is a bit plus small Gaussian noise and no |x| is below 0.49 in any dataset, so the
    rounding never flips a bit: nothing the label depends on is lost.
    """
    return np.where(X < 0, -1.0, 1.0)


def codes(X, S):
    """Encode the bits of X on the coordinates S as one integer per row (bit i is set iff x_{S[i]} < 0).

    This lets us index lookup tables and truth tables with 2^|S| entries directly.
    """
    S = list(S)
    return ((np.asarray(X)[:, S] < 0).astype(np.int64) << np.arange(len(S))).sum(1)

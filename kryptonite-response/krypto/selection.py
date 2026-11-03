"""Greedy forward feature selection and the histogram (lookup-table) classifier on selected bits.

The histogram classifier estimates P(y=1 | b_S) separately for each of the 2^|S| bit patterns of the
selected coordinates S (Laplace smoothing).  It is the plug-in Bayes classifier for a label that
depends only on b_S, and is equivalent to logistic regression on ALL multilinear monomials of b_S,
i.e. a degree-|S| polynomial in the selected bits.
"""
import numpy as np

from .data import codes


class LookupTable:
    def __init__(self, S, alpha=0.5):
        self.S, self.alpha = list(S), alpha

    def fit(self, X, y):
        c = codes(X, self.S)
        cnt = np.bincount(c, minlength=1 << len(self.S))
        pos = np.bincount(c, weights=y, minlength=1 << len(self.S))
        self.p_ = (pos + self.alpha) / (cnt + 2 * self.alpha)
        return self

    def predict_proba(self, X):
        return self.p_[codes(X, self.S)]

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    def score(self, X, y):
        return float((self.predict(X) == y).mean())


def _val_nll(X_tr, y_tr, X_va, y_va, S):
    q = np.clip(LookupTable(S).fit(X_tr, y_tr).predict_proba(X_va), 1e-6, 1 - 1e-6)
    return float(-np.mean(y_va * np.log(q) + (1 - y_va) * np.log(1 - q)))


def greedy_forward_selection(X_tr, y_tr, X_va, y_va, max_k=16):
    """Select coordinates by greedily minimising the validation log-loss of a lookup table.

    No single bit is correlated with the label, so the search starts from the best PAIR, then adds one
    coordinate at a time and stops as soon as the validation log-loss no longer decreases.
    Only training and validation data are used.  Returns (S, history of validation log-losses).
    """
    n = X_tr.shape[1]
    history = [_val_nll(X_tr, y_tr, X_va, y_va, [])]
    pairs = [(_val_nll(X_tr, y_tr, X_va, y_va, [i, j]), i, j) for i in range(n) for j in range(i + 1, n)]
    nll, i, j = min(pairs)
    S = [i, j]
    history.append(nll)
    while len(S) < max_k:
        nll, l = min((_val_nll(X_tr, y_tr, X_va, y_va, S + [l]), l) for l in range(n) if l not in S)
        if nll >= history[-1]:
            break
        S.append(l)
        history.append(nll)
    return S, history

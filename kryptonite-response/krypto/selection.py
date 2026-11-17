"""Greedy forward feature selection, scored by a lookup-table classifier on the selected bits.

The lookup table estimates P(y=1 | b_S) separately for each of the 2^|S| bit patterns of the selected
coordinates S.  We use it as the scoring model because it makes no assumption about the shape of the
function on S: it is the plug-in Bayes classifier for any label that depends only on b_S, and it can
represent every multilinear polynomial in those bits.  So if adding a coordinate lowers its validation
loss, that coordinate carries information, whatever the function looks like.
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
        # add-alpha smoothing keeps empty or tiny cells near 1/2 instead of at 0 or 1, which would make
        # the log-loss blow up on validation patterns that are rare in training
        self.p_ = (pos + self.alpha) / (cnt + 2 * self.alpha)
        return self

    def predict_proba(self, X):
        return self.p_[codes(X, self.S)]

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    def score(self, X, y):
        return float((self.predict(X) == y).mean())


def _val_nll(X_tr, y_tr, X_va, y_va, S):
    """Validation log-loss of a lookup table on S fitted to the training part.

    We use the log-loss rather than accuracy because it changes smoothly as a coordinate is added and so
    also rewards coordinates that make the predictions more confident without changing many of them.
    """
    q = np.clip(LookupTable(S).fit(X_tr, y_tr).predict_proba(X_va), 1e-6, 1 - 1e-6)
    return float(-np.mean(y_va * np.log(q) + (1 - y_va) * np.log(1 - q)))


def greedy_forward_selection(X_tr, y_tr, X_va, y_va, max_k=16):
    """Select coordinates by greedily lowering the validation log-loss of a lookup table.

    We start from the best PAIR rather than the best single coordinate: in Kryptonite no single bit is
    correlated with the label (notebook 01), so a one-at-a-time search would have nothing to go on in the
    first step.  From the pair we add one coordinate at a time and stop as soon as the validation log-loss
    stops decreasing, which is also when the table starts to have too many cells for the data.
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

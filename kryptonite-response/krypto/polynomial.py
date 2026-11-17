"""Polynomial logistic regression: the mock paper's models and our version on selected coordinates.

The functions in the first section (`poly_logreg`, `agreements`, `polyfeatures_kernel_table`,
`kernel_logreg`) are taken unchanged from the mock paper's notebook (MockPaperExperiments.ipynb).  Our
reproduction therefore runs their exact model, and any difference in our results comes from the inputs
we give it, not from a reimplementation.

Our own model rests on one observation.  On +-1 bits b_i^2 = 1, so every monomial produced by
PolynomialFeatures(d) collapses to a parity chi_T(b) = prod_{i in T} b_i, and the mock paper's
normalised polynomial kernel can be written as
    k(b, b') = sum_{|T| <= d} c_|T| chi_T(b) chi_T(b') / Z,   c_s = C(k + floor((d-s)/2), k).
Kernel logistic regression in the dual is then exactly ordinary (primal) logistic regression on the
explicit features sqrt(c_|T| / Z) chi_T(b).  On all n bits this feature space is far too large to write
down, but on k = 10 selected bits it has at most 2^10 = 1024 features, so we can fit the very same model
directly and quickly.
"""
import itertools
import warnings
from math import comb

import numpy as np
import scipy.linalg as sla
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

from .data import bits

warnings.filterwarnings("ignore", message=".*encountered in matmul")


# ----------------------------------------------------------------------------------------------
# Mock paper models (unchanged)
# ----------------------------------------------------------------------------------------------
def poly_logreg(X_train, y_train, X_test, y_test, degree):
    """Logistic regression on explicit polynomial features, as in the mock paper's Figure 2 (data scaling)."""
    poly = PolynomialFeatures(degree)
    logreg = LogisticRegression(max_iter=100, solver="sag", C=0.85)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # sag hits max_iter=100, exactly as in the mock paper
        logreg.fit(poly.fit_transform(X_train), y_train)
    return float(logreg.score(poly.transform(X_test), y_test))


def agreements(X1, X2, n):
    """Number of agreeing bits between every pair of rows.  Note that the mock paper's kernel model
    itself rounds the inputs to bits with np.sign, so working on bits is not a choice we added."""
    B1, B2 = np.sign(X1).astype(np.float32), np.sign(X2).astype(np.float32)
    return np.rint((n + B1 @ B2.T) / 2).astype(np.uint8)


def polyfeatures_kernel_table(n, degree):
    """phi(b).phi(b') for PolynomialFeatures(degree) on +-1 bits as a function of the number m of agreeing
    bits, normalised so that k(b, b) = 1.  On the hypercube the kernel depends on b, b' only through m."""
    table = np.zeros(n + 1)
    for s in range(degree + 1):
        weight = comb(n + (degree - s) // 2, n)
        e_s = [sum(comb(m, a) * comb(n - m, s - a) * (-1) ** (s - a) for a in range(s + 1)) for m in range(n + 1)]
        table += weight * np.array(e_s, dtype=float)
    return table / table[n]


def kernel_logreg(K, y, C=0.85, iters=30, tol=1e-6):
    """Minimise C * sum(log-loss) + ||w||^2 / 2 in the dual by Newton's method (Rasmussen & Williams, Alg. 3.1).

    Each step factorises an N x N matrix, which is why one fit at n = 40 needs ~4 GB and minutes of CPU.
    """
    t = y.astype(np.float64); sign = 2 * t - 1
    a, f = np.zeros(len(t), np.float32), np.zeros(len(t))
    obj = C * len(t) * np.log(2)
    for _ in range(iters):
        pi = 1 / (1 + np.exp(-f))
        W = C * pi * (1 - pi); sW = np.sqrt(W).astype(np.float32)
        B = K * sW[:, None]; B *= sW[None, :]; B[np.diag_indices_from(B)] += 1
        L = sla.cholesky(B, lower=True, overwrite_a=True, check_finite=False)
        del B
        b = (W * f + C * (t - pi)).astype(np.float32)
        v = sla.solve_triangular(L, sW * (K @ b), lower=True, check_finite=False)
        a_new = b - sW * sla.solve_triangular(L, v, lower=True, trans="T", check_finite=False)
        del L
        f_new = (K @ a_new).astype(np.float64)
        new = C * np.logaddexp(0, -sign * f_new).sum() + 0.5 * float(a_new @ f_new)
        for _ in range(5):
            if new <= obj:
                break
            a_new, f_new = (a + a_new) / 2, (f + f_new) / 2
            new = C * np.logaddexp(0, -sign * f_new).sum() + 0.5 * float(a_new @ f_new)
        done = abs(obj - new) < tol * abs(new)
        a, f, obj = a_new, f_new, new
        if done:
            break
    return a


def mock_kernel_accuracies(X_train, y_train, X_test, y_test, degrees=range(1, 11), C=0.85, verbose=True):
    """The mock paper's Figure 1: normalised polynomial-kernel logistic regression on all n bits, one fit
    per degree, returning the test accuracy for each degree."""
    n = X_train.shape[1]
    M_train, M_test = agreements(X_train, X_train, n), agreements(X_test, X_train, n)
    out = {}
    for d in degrees:
        table = polyfeatures_kernel_table(n, d).astype(np.float32)
        a = kernel_logreg(table[M_train], y_train, C=C)
        out[d] = float(((table[M_test] @ a > 0).astype(int) == y_test).mean())
        if verbose:
            print(f"n={n} degree={d} features={comb(n + d, d):,} accuracy={out[d]:.3f}", flush=True)
    return out


# ----------------------------------------------------------------------------------------------
# Our model: the same kernel model, restricted to the selected coordinates and fitted in the primal
# ----------------------------------------------------------------------------------------------
def parity_features(X, S, degree):
    """Explicit feature map of the normalised PolynomialFeatures(degree) kernel on the bits b_S.

    One column per parity chi_T with |T| <= degree, scaled by sqrt(c_|T| / Z).  The scaling matters: it is
    what makes our fit identical to the mock paper's, and it gives high-degree parities small features,
    which is why the mock paper's C = 0.85 shrinks them so strongly.
    """
    B, k = bits(X)[:, list(S)], len(S)
    weights = {s: comb(k + (degree - s) // 2, k) for s in range(degree + 1)}
    Z = sum(weights[s] * comb(k, s) for s in range(min(degree, k) + 1))
    cols = [np.ones(len(B))]
    scales = [weights[0]]
    for s in range(1, min(degree, k) + 1):
        for T in itertools.combinations(range(k), s):
            cols.append(B[:, T].prod(1))
            scales.append(weights[s])
    return np.stack(cols, 1) * np.sqrt(np.array(scales) / Z)


class SparsePolyLogReg:
    """Degree-d polynomial logistic regression with the mock paper's kernel on the selected bits S.

    C defaults to the mock paper's 0.85; in our experiments it is chosen on the validation split.
    """

    def __init__(self, S, degree, C=0.85):
        self.S, self.degree, self.C = list(S), degree, C

    def fit(self, X, y):
        self.lr_ = LogisticRegression(C=self.C, fit_intercept=False, max_iter=5000, tol=1e-8)
        self.lr_.fit(parity_features(X, self.S, self.degree), y)
        return self

    def decision_function(self, X):
        return self.lr_.decision_function(parity_features(X, self.S, self.degree))

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)

    def score(self, X, y):
        return float((self.predict(X) == y).mean())

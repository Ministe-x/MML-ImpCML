"""Fourier (Walsh-Hadamard) analysis of Boolean functions on the hypercube {-1, +1}^k.

Every function g: {-1,+1}^k -> R has the unique expansion  g(b) = sum_S ghat(S) chi_S(b),
chi_S(b) = prod_{i in S} b_i, with ghat(S) = E_b[g(b) chi_S(b)] under the uniform distribution.
For a +-1 valued label s = 2y - 1 we estimate shat(S) = mean_i s_i chi_S(b_i) from data.
"""
import numpy as np

from .data import codes


def fwht(a):
    """Unnormalised fast Walsh-Hadamard transform of a vector of length 2^k."""
    a = np.array(a, dtype=float)
    h, n = 1, a.shape[0]
    while h < n:
        a = a.reshape(-1, 2, h)
        x, y = a[:, 0, :].copy(), a[:, 1, :].copy()
        a[:, 0, :], a[:, 1, :] = x + y, x - y
        a = a.reshape(n)
        h *= 2
    return a


def popcounts(k):
    return np.array([bin(S).count("1") for S in range(1 << k)])


def empirical_spectrum(X, y, S):
    """Empirical Fourier coefficients of s = 2y-1 over all 2^|S| subsets of the coordinates S.

    Entry T (a bitmask over positions in S) estimates E[s * prod_{i in T} b_{S[i]}].
    """
    c = codes(X, S)
    g = np.zeros(1 << len(S))
    np.add.at(g, c, (2 * np.asarray(y) - 1) / len(y))
    return fwht(g)


def truth_table(X, y, S):
    """Majority label (+-1) of every bit pattern of the coordinates S (ties -> +1)."""
    c = codes(X, S)
    m = np.bincount(c, weights=2 * np.asarray(y) - 1.0, minlength=1 << len(S))
    return np.where(m >= 0, 1.0, -1.0)


def spectrum_of_table(tt):
    """Exact Fourier coefficients of a +-1 truth table of length 2^k."""
    return fwht(tt) / len(tt)


def degree_weights(F):
    """Fourier weight W^{=d} = sum_{|S|=d} fhat(S)^2 for d = 0..k."""
    k = int(np.log2(len(F)))
    pc = popcounts(k)
    return np.array([(F[pc == d] ** 2).sum() for d in range(k + 1)])


def influences(F):
    """Influence of each variable, Inf_i = sum_{S containing i} fhat(S)^2."""
    k = int(np.log2(len(F)))
    return np.array([(F[[(T >> i) & 1 == 1 for T in range(1 << k)]] ** 2).sum() for i in range(k)])


def truncation_accuracy(tt, d, noise=0.0):
    """Accuracy of sign(f^{<=d}), the sign of the L2-best degree-d approximation of the +-1 table tt.

    Under the uniform distribution on the k bits and symmetric label noise with flip rate `noise`.
    """
    k = int(np.log2(len(tt)))
    F = spectrum_of_table(tt) * (popcounts(k) <= d)
    p = fwht(F)                       # evaluates sum_S F(S) chi_S(b) at every pattern b
    acc = np.mean(np.where(p > 1e-12, 1.0, np.where(p < -1e-12, -1.0, 0.0)) == tt)
    acc += 0.5 * np.mean(np.abs(p) <= 1e-12)          # ties count as coin flips
    return acc * (1 - noise) + (1 - acc) * noise


def pairwise_coefficients(B, s):
    """Matrix of all degree-2 coefficients E[s b_i b_j] (diagonal set to 0)."""
    C = (B * s[:, None]).T @ B / len(s)
    np.fill_diagonal(C, 0.0)
    return C

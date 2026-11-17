"""Fourier (Walsh-Hadamard) analysis of Boolean functions on {-1, +1}^k.

This is the main tool we use to describe the label function.  Any g: {-1,+1}^k -> R can be written
uniquely as a multilinear polynomial,
    g(b) = sum_S ghat(S) chi_S(b),   chi_S(b) = prod_{i in S} b_i,   ghat(S) = E_b[g(b) chi_S(b)],
so the coefficients tell us which groups of bits the function depends on and at which degree.  For
data with labels s = 2y - 1 we estimate shat(S) by the sample mean of s * chi_S(b).

Subsets S of k coordinates are stored as k-bit masks, so a spectrum is simply a vector of length 2^k.
"""
import numpy as np

from .data import codes


def fwht(a):
    """Unnormalised fast Walsh-Hadamard transform of a vector of length 2^k, in O(k 2^k) operations.

    The transform is its own inverse up to a factor 2^k, so the same routine maps a truth table to its
    Fourier coefficients and the coefficients back to function values.
    """
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
    """Size |S| of every subset S of k coordinates, i.e. the Fourier degree of each entry of a spectrum."""
    return np.array([bin(S).count("1") for S in range(1 << k)])


def empirical_spectrum(X, y, S):
    """Estimated Fourier coefficients of the label s = 2y-1 on all 2^|S| subsets of the coordinates S.

    Entry T (a bitmask over positions in S) estimates E[s * prod_{i in T} b_{S[i]}].  Summing the labels per
    bit pattern first and transforming once is much faster than averaging every parity separately.
    """
    c = codes(X, S)
    g = np.zeros(1 << len(S))
    np.add.at(g, c, (2 * np.asarray(y) - 1) / len(y))
    return fwht(g)


def truth_table(X, y, S):
    """Majority label (+-1) of each bit pattern on the coordinates S, our estimate of the noise-free function.

    On Kryptonite-n there are about 9-25 examples per pattern on 10 bits and only about 4% label noise, so
    the majority vote is almost always right (Adapt has only ~3 per pattern and is noisier).  Ties are
    broken towards +1.
    """
    c = codes(X, S)
    m = np.bincount(c, weights=2 * np.asarray(y) - 1.0, minlength=1 << len(S))
    return np.where(m >= 0, 1.0, -1.0)


def spectrum_of_table(tt):
    """Exact Fourier coefficients of a +-1 truth table of length 2^k."""
    return fwht(tt) / len(tt)


def degree_weights(F):
    """Fourier weight per degree, W^{=d} = sum_{|S|=d} fhat(S)^2 for d = 0..k.

    For a +-1 function the weights sum to 1, so W^{=d} is the share of the function that lives at degree d.
    """
    k = int(np.log2(len(F)))
    pc = popcounts(k)
    return np.array([(F[pc == d] ** 2).sum() for d in range(k + 1)])


def influences(F):
    """Influence of each variable, Inf_i = sum_{S containing i} fhat(S)^2.

    For a +-1 function this is the probability that flipping b_i flips the output.
    """
    k = int(np.log2(len(F)))
    return np.array([(F[[(T >> i) & 1 == 1 for T in range(1 << k)]] ** 2).sum() for i in range(k)])


def truncation_accuracy(tt, d, noise=0.0):
    """Accuracy of sign(f^{<=d}), the sign of the best (L2) degree-d approximation of the +-1 table tt.

    Inputs are uniform on the k bits and labels are flipped independently with probability `noise`.
    """
    k = int(np.log2(len(tt)))
    F = spectrum_of_table(tt) * (popcounts(k) <= d)
    p = fwht(F)                       # evaluates sum_S F(S) chi_S(b) at every pattern b
    acc = np.mean(np.where(p > 1e-12, 1.0, np.where(p < -1e-12, -1.0, 0.0)) == tt)
    acc += 0.5 * np.mean(np.abs(p) <= 1e-12)          # where the approximation is exactly 0 we guess, right half the time
    # a correct prediction scores only when the label is not flipped, a wrong one only when it is
    return acc * (1 - noise) + (1 - acc) * noise


def pairwise_coefficients(B, s):
    """All degree-2 coefficients E[s b_i b_j] as a matrix (diagonal set to 0)."""
    C = (B * s[:, None]).T @ B / len(s)
    np.fill_diagonal(C, 0.0)
    return C

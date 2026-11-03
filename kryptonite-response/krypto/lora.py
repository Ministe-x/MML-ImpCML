"""Low-rank adaptation (LoRA, Hu et al. 2022) of KryptoNet-40, implemented from scratch.

For a frozen layer h = W x + b with W in R^{d_out x d_in}, LoRA trains only
    A in R^{r x d_in}  (random init, Kaiming-uniform as in Hu et al.)
    B in R^{d_out x r} (zero init, so the adapted network equals the pretrained one at step 0)
and computes  h = W x + b + (alpha / r) * B (A x).   The update Delta W = (alpha/r) B A has rank <= r.
"""
import copy
import math
import sys

import torch
import torch.nn as nn

from .data import ROOT

sys.path.insert(0, str(ROOT))
from kryptonet import load_kryptonet  # noqa: E402  (provided with the coursework)

KRYPTONET_PATH = ROOT / "kryptonet40.pt"
LINEAR_LAYERS = (0, 2, 4, 6, 8)        # indices of the nn.Linear modules inside KryptoNet-40


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float = None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                      # W and b stay frozen
        self.r = r
        self.scale = (alpha if alpha is not None else r) / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):                                # x: (N, d_in), rows are examples
        return self.base(x) + self.scale * ((x @ self.A.T) @ self.B.T)   # B(Ax): never forms B A

    def delta_weight(self):
        return self.scale * (self.B @ self.A)

    def merged(self):
        """Return an ordinary nn.Linear with W' = W + (alpha/r) B A (zero inference overhead)."""
        lin = copy.deepcopy(self.base)
        with torch.no_grad():
            lin.weight += self.delta_weight()
        return lin


def load_base():
    return load_kryptonet(str(KRYPTONET_PATH))


def frozen(model):
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def add_lora(model, r, layers=(0,), alpha=None):
    """Copy of `model` with every parameter frozen and LoRA adapters on the given Linear layers."""
    m = frozen(model)
    for i in layers:
        m[i] = LoRALinear(m[i], r, alpha)
    return m


def linear_probe(model):
    m = frozen(model)
    m[-1] = nn.Linear(m[-1].in_features, 1)
    return m


def mlp_probe(model, width=512):
    m = frozen(model)
    m[-1] = nn.Sequential(nn.Linear(m[-1].in_features, width), nn.ReLU(), nn.Linear(width, 1))
    return m


def full_finetune(model):
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad_(True)
    return m


def swap_oracle(model, i=0, j=19):
    """Analytic rank-1 edit of the first layer, W1' = W1 + (w_j - w_i)(e_i - e_j)^T (swaps columns i, j)."""
    m = frozen(model)
    with torch.no_grad():
        W = m[0].weight
        u = (W[:, j] - W[:, i]).clone()
        v = torch.zeros(W.shape[1]); v[i], v[j] = 1.0, -1.0
        W += torch.outer(u, v)
    return m


# ------------------------------------------------------------------------------------------------
# Operation and parameter counts (product of an n x m and an m x k matrix counted as n m k)
# ------------------------------------------------------------------------------------------------
def ops_full(d_in, d_out, N, input_grad=True):
    """One training step (forward + backward) of a fully fine-tuned layer."""
    fwd = d_out * d_in * N                                  # W X
    bwd = d_out * N * d_in                                  # dW = G X^T
    if input_grad:
        bwd += d_in * d_out * N                             # dX = W^T G
    return {"forward": fwd, "backward": bwd, "total": fwd + bwd, "params": d_out * d_in + d_out}


def ops_lora_BAx(d_in, d_out, N, r, input_grad=True):
    """LoRA evaluated right-to-left, B(AX): never forms the d_out x d_in product B A."""
    fwd = d_out * d_in * N + r * d_in * N + d_out * r * N   # W X, U = A X, B U
    bwd = d_out * N * r + r * d_out * N + r * N * d_in      # dB = G U^T, dU = B^T G, dA = dU X^T
    if input_grad:
        bwd += d_in * d_out * N + d_in * r * N              # dX = W^T G + A^T dU
    return {"forward": fwd, "backward": bwd, "total": fwd + bwd, "params": r * (d_in + d_out)}


def ops_lora_BA_x(d_in, d_out, N, r, input_grad=True):
    """LoRA evaluated left-to-right, (BA)X with the merged weight W + BA."""
    fwd = d_out * r * d_in + d_out * d_in * N               # M = B A, (W + M) X
    bwd = d_out * N * d_in + d_out * d_in * r + r * d_out * d_in   # dM = G X^T, dB = dM A^T, dA = B^T dM
    if input_grad:
        bwd += d_in * d_out * N                             # dX = (W + M)^T G
    return {"forward": fwd, "backward": bwd, "total": fwd + bwd, "params": r * (d_in + d_out)}

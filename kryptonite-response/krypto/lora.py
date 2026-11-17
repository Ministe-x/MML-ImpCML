"""Low-rank adaptation (LoRA, Hu et al. 2022) of KryptoNet-40, written from scratch rather than with peft.

The idea is to keep a pretrained layer h = W x + b frozen and learn only a low-rank correction,
    h = W x + b + (alpha / r) * B (A x),   A in R^{r x d_in},  B in R^{d_out x r},
so the effective weight change Delta W = (alpha/r) B A has rank at most r and costs r (d_in + d_out)
parameters instead of d_in d_out.  As in the original paper, A starts random (Kaiming-uniform) and B
starts at zero, so the adapted network is exactly the pretrained one before the first update.
For Kryptonite-40-Adapt this is a good fit: the task differs from pretraining by moving one input
coordinate, and that change of the first layer has rank one (Proposition 2 in the paper).
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
LINEAR_LAYERS = (0, 2, 4, 6, 8)        # positions of the nn.Linear modules inside KryptoNet-40 (ReLUs in between)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float = None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                      # W and b stay frozen
        self.r = r
        self.scale = (alpha if alpha is not None else r) / r     # alpha defaults to r, i.e. no extra scaling
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):                                # x: (N, d_in), one example per row
        # multiply right to left, B(Ax): the d_out x d_in matrix B A is never formed, which is what
        # makes a LoRA step cheaper than full fine-tuning (see the operation counts below)
        return self.base(x) + self.scale * ((x @ self.A.T) @ self.B.T)

    def delta_weight(self):
        return self.scale * (self.B @ self.A)

    def merged(self):
        """A plain nn.Linear with W' = W + (alpha/r) B A, so the adapted layer costs nothing extra at inference."""
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


# Each adaptation method below returns a fresh copy of KryptoNet-40 in which only the parameters the
# method may change require gradients; krypto.mlp.train then trains exactly those.
def add_lora(model, r, layers=(0,), alpha=None):
    """Frozen copy of `model` with rank-r LoRA adapters on the Linear layers at the given positions."""
    m = frozen(model)
    for i in layers:
        m[i] = LoRALinear(m[i], r, alpha)
    return m


def linear_probe(model):
    """The mock paper's linear probe: frozen network, new trainable output layer."""
    m = frozen(model)
    m[-1] = nn.Linear(m[-1].in_features, 1)
    return m


def mlp_probe(model, width=512):
    """The mock paper's MLP probe: frozen network, a small trainable MLP instead of the output layer."""
    m = frozen(model)
    m[-1] = nn.Sequential(nn.Linear(m[-1].in_features, width), nn.ReLU(), nn.Linear(width, 1))
    return m


def full_finetune(model):
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad_(True)
    return m


def swap_oracle(model, i=0, j=19):
    """The rank-1 edit that LoRA should find, computed by hand rather than learned.

    W1' = W1 + (w_j - w_i)(e_i - e_j)^T exchanges columns i and j of the first layer, so the network reads
    coordinate j where it used to read coordinate i.  It uses knowledge from notebook 01 (which coordinate
    moved), so we treat it as a reference point, not as an adaptation method.
    """
    m = frozen(model)
    with torch.no_grad():
        W = m[0].weight
        u = (W[:, j] - W[:, i]).clone()
        v = torch.zeros(W.shape[1]); v[i], v[j] = 1.0, -1.0
        W += torch.outer(u, v)
    return m


# ------------------------------------------------------------------------------------------------
# Operation counts of one training step of a single layer, using the coursework's convention that the
# product of an n x m and an m x k matrix costs n m k operations.  X is d_in x N (one example per
# column) and G is the d_out x N gradient arriving from the layer above.  Notebook 06 checks every
# formula against PyTorch's FlopCounterMode.
# ------------------------------------------------------------------------------------------------
def ops_full(d_in, d_out, N, input_grad=True):
    """Full fine-tuning of one layer.  input_grad=False for the first layer, whose input needs no gradient."""
    fwd = d_out * d_in * N                                  # W X
    bwd = d_out * N * d_in                                  # dW = G X^T
    if input_grad:
        bwd += d_in * d_out * N                             # dX = W^T G
    return {"forward": fwd, "backward": bwd, "total": fwd + bwd, "params": d_out * d_in + d_out}


def ops_lora_BAx(d_in, d_out, N, r, input_grad=True):
    """LoRA evaluated right to left, B(AX): every extra product has r as one of its dimensions."""
    fwd = d_out * d_in * N + r * d_in * N + d_out * r * N   # W X, U = A X, B U
    bwd = d_out * N * r + r * d_out * N + r * N * d_in      # dB = G U^T, dU = B^T G, dA = dU X^T
    if input_grad:
        bwd += d_in * d_out * N + d_in * r * N              # dX = W^T G + A^T dU
    return {"forward": fwd, "backward": bwd, "total": fwd + bwd, "params": r * (d_in + d_out)}


def ops_lora_BA_x(d_in, d_out, N, r, input_grad=True):
    """LoRA evaluated left to right, (BA)X: forming B A and its gradient costs as much as full fine-tuning."""
    fwd = d_out * r * d_in + d_out * d_in * N               # M = B A, (W + M) X
    bwd = d_out * N * d_in + d_out * d_in * r + r * d_out * d_in   # dM = G X^T, dB = dM A^T, dA = B^T dM
    if input_grad:
        bwd += d_in * d_out * N                             # dX = (W + M)^T G
    return {"forward": fwd, "backward": bwd, "total": fwd + bwd, "params": r * (d_in + d_out)}

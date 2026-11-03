"""Multilayer perceptrons and a single training loop shared by every neural model in the paper.

Training: AdamW on the binary cross-entropy (logistic) loss, mini-batches, and early stopping on
VALIDATION accuracy (patience epochs without improvement), restoring the best weights.  The
test split is never seen during training.
"""
import copy
import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mlp(n_inputs, width=512, depth=3, dropout=0.0):
    """ReLU MLP with `depth` hidden layers of size `width` and one output logit."""
    layers, d = [], n_inputs
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = width
    return nn.Sequential(*layers, nn.Linear(d, 1))


def n_trainable(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def to_tensor(*arrays):
    return [torch.as_tensor(np.ascontiguousarray(a)) for a in arrays]


@torch.no_grad()
def logits(net, X, batch=8192):
    net.eval()
    X = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32)
    return torch.cat([net(X[i:i + batch]).squeeze(1) for i in range(0, len(X), batch)]).numpy()


def accuracy(net, X, y):
    return float(((logits(net, X) > 0).astype(int) == np.asarray(y)).mean())


def train(net, X_tr, y_tr, X_va, y_va, lr=1e-3, weight_decay=0.0, batch_size=128,
          max_epochs=300, patience=30, seed=0):
    """Train the parameters of `net` that require grad; early stopping on validation accuracy.

    Returns (net with best weights restored, dict with best validation accuracy and epochs run).
    """
    set_seed(seed)
    X_tr, y_tr = to_tensor(X_tr.astype(np.float32), y_tr.astype(np.float32))
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    best, best_state, since, epoch = -1.0, None, 0, 0
    for epoch in range(1, max_epochs + 1):
        net.train()
        perm = torch.randperm(len(y_tr))
        for i in range(0, len(y_tr), batch_size):
            idx = perm[i:i + batch_size]
            loss = nn.functional.binary_cross_entropy_with_logits(net(X_tr[idx]).squeeze(1), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        val = accuracy(net, X_va, y_va)
        if val > best:
            best, since, best_state = val, 0, copy.deepcopy(net.state_dict())
        else:
            since += 1
            if since >= patience:
                break
    net.load_state_dict(best_state)
    return net, {"val_acc": best, "epochs": epoch}

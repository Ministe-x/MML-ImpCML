import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import sys
    import time
    from pathlib import Path

    import marimo as mo

    sys.path.insert(0, str(mo.notebook_dir() or Path(__file__).resolve().parent))
    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from torch.utils.flop_counter import FlopCounterMode

    from kmo import save_json, track
    from krypto.lora import LoRALinear, ops_full, ops_lora_BAx, ops_lora_BA_x, load_base, add_lora, full_finetune
    from krypto.plotting import use_paper_style, COLUMN, FULL, COLORS

    PLOT_BACKEND = use_paper_style()
    torch.manual_seed(0)
    LAYERS = {"hidden layer 512×512 (input gradient needed)": (512, 512, True),
              "first layer 512×40 (no input gradient)": (40, 512, False)}

    class LoRAMerged(nn.Module):
        """LoRA evaluated left to right: forms M = B A (d_out x d_in) and then (W + M) X."""

        def __init__(self, base: nn.Linear, r: int):
            super().__init__()
            self.base = base
            for p in base.parameters():
                p.requires_grad_(False)
            self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
            self.B = nn.Parameter(torch.randn(base.out_features, r) * 0.01)

        def forward(self, x):
            return nn.functional.linear(x, self.base.weight + self.B @ self.A, self.base.bias)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 06 · Operation count of one LoRA training step (Table 2)

    How much does LoRA actually save per training step?  The answer depends on the order of evaluation, so we first
    write out the matrix products of one step and then check every count against PyTorch's `FlopCounterMode`.  For a
    layer $W\in\mathbb R^{d_\text{out}\times d_\text{in}}$ and a batch $X\in\mathbb R^{d_\text{in}\times N}$ with upstream
    gradient $G=\partial L/\partial H\in\mathbb R^{d_\text{out}\times N}$, counting an $n\times m$ by $m\times k$ product
    as $nmk$ (elementwise operations not counted):

    | method | forward | backward |
    |---|---|---|
    | full fine-tuning | $WX$: $Nd_\text{in}d_\text{out}$ | $\nabla_W=GX^\top$: $Nd_\text{in}d_\text{out}$; $W^\top G$: $Nd_\text{in}d_\text{out}$ |
    | LoRA, $B(AX)$ | $WX$, $U=AX$: $rNd_\text{in}$, $BU$: $rNd_\text{out}$ | $\nabla_B=GU^\top$, $\nabla_U=B^\top G$: $rNd_\text{out}$ each; $\nabla_A=\nabla_UX^\top$: $rNd_\text{in}$; $W^\top G+A^\top\nabla_U$ |
    | LoRA, $(BA)X$ | $M=BA$: $rd_\text{in}d_\text{out}$; $(W+M)X$ | $\nabla_M=GX^\top$: $Nd_\text{in}d_\text{out}$; $\nabla_B,\nabla_A$: $rd_\text{in}d_\text{out}$ each; $(W+M)^\top G$ |

    With the input gradient, **LoRA as $B(AX)$ costs $2Nd_\text{in}d_\text{out}+3rN(d_\text{in}+d_\text{out})$ against
    $3Nd_\text{in}d_\text{out}$ for full fine-tuning**: it skips the weight gradient of $W$.  Evaluated as $(BA)X$ it costs
    $3Nd_\text{in}d_\text{out}+3rd_\text{in}d_\text{out}$, *more* than full fine-tuning, because forming $BA$ and its
    gradient brings back products of full size.  Trainable parameters: $r(d_\text{in}+d_\text{out})$ against
    $d_\text{out}d_\text{in}+d_\text{out}$.  `FlopCounterMode` reports $2nmk$ per product (a multiply and an add per
    term), so we divide its counts by 2.
    """)
    return


@app.function
def count(module, d_in, N, input_grad):
    """Measured (forward, backward) matrix-product operations of one step, in the coursework's nmk convention."""
    x = torch.randn(N, d_in, requires_grad=input_grad)
    with FlopCounterMode(display=False) as fwd:
        out = module(x)
    with FlopCounterMode(display=False) as bwd:
        out.sum().backward()
    return fwd.get_total_flops() // 2, bwd.get_total_flops() // 2


@app.function
def check(d_in, d_out, N, r, input_grad):
    """Measure the three ways of training one layer and assert equality with the analytic formulas."""
    rows = []
    for name, module, formula in (
            ("full fine-tuning", nn.Linear(d_in, d_out), ops_full(d_in, d_out, N, input_grad)),
            ("LoRA, B(AX)", LoRALinear(nn.Linear(d_in, d_out), r), ops_lora_BAx(d_in, d_out, N, r, input_grad)),
            ("LoRA, (BA)X", LoRAMerged(nn.Linear(d_in, d_out), r), ops_lora_BA_x(d_in, d_out, N, r, input_grad))):
        f, b = count(module, d_in, N, input_grad)
        assert (f, b) == (formula["forward"], formula["backward"]), (name, f, b, formula)
        rows.append((name, f, b, f + b, formula["params"]))
    return rows


@app.cell(hide_code=True)
def _():
    layer = mo.ui.dropdown(options=LAYERS, value=list(LAYERS)[0], label="layer")
    batch = mo.ui.dropdown(options={str(v): v for v in (16, 64, 256, 1024)}, value="64", label="batch N")
    rank = mo.ui.slider(steps=[1, 2, 4, 8, 16, 32, 64, 85, 128, 256], value=4, label="rank r", show_value=True)
    mo.vstack([mo.md("## Explore: one layer, measured against the formulas"),
               mo.hstack([layer, batch, rank], justify="start")])
    return batch, layer, rank


@app.cell(hide_code=True)
def _(batch, layer, rank):
    _d_in, _d_out, _grad = layer.value
    _rows = check(_d_in, _d_out, batch.value, rank.value, _grad)
    _full = _rows[0][3]
    _rs = np.unique(np.r_[np.geomspace(1, 512, 60).round().astype(int), rank.value])
    _ratio = {name: [fn(_d_in, _d_out, batch.value, r, _grad)["total"] / _full for r in _rs]
              for name, fn in (("LoRA, B(AX)", ops_lora_BAx), ("LoRA, (BA)X", ops_lora_BA_x))}
    _fig, _ax = plt.subplots(figsize=(COLUMN, 2.2))
    _ax.axhline(1, color="k", ls="--", lw=1, label="full fine-tuning")
    _ax.plot(_rs, _ratio["LoRA, B(AX)"], color=COLORS[40], label="LoRA, $B(AX)$")
    _ax.plot(_rs, _ratio["LoRA, (BA)X"], color=COLORS["adapt"], label="LoRA, $(BA)X$")
    _ax.axvline(rank.value, color=COLORS["gray"], ls=":", lw=1)
    _ax.set_xscale("log"); _ax.set_xlabel("rank $r$"); _ax.set_ylabel("operations / full")
    _ax.legend(loc="upper left", fontsize=8)
    _even = _d_in * _d_out / (3 * (_d_in + _d_out)) if _grad else None
    mo.vstack([
        mo.ui.table([{"method": n, "forward": f"{f:,}", "backward": f"{b:,}", "total": f"{t:,}",
                      "× full": f"{t / _full:.3f}", "trainable params": f"{p:,}"} for n, f, b, t, p in _rows],
                    selection=None, pagination=False),
        mo.md("✔ Measured counts equal the formulas for all three methods."
              + (f"  For this layer $B(AX)$ is cheaper than full fine-tuning exactly when "
                 f"$r < d_\\text{{in}}d_\\text{{out}}/(3(d_\\text{{in}}+d_\\text{{out}})) = {_even:.1f}$." if _even else
                 "  Without an input gradient the frozen first layer needs neither $\\nabla_W$ nor $W^\\top G$, "
                 "so LoRA saves almost half of the step.")),
        _fig,
    ])
    return


@app.function
def compute_grid():
    """The configurations of Table 2, each checked against the formulas."""
    table = {}
    with track("06_flop_counts"):
        for d_in, d_out, input_grad, name in ((512, 512, True, "hidden layer 512x512"),
                                              (40, 512, False, "first layer 512x40")):
            for N, r in ((64, 1), (64, 4), (64, 64), (1024, 4)):
                table[f"{name}|N={N}|r={r}"] = [list(row) for row in check(d_in, d_out, N, r, input_grad)]
    return table


@app.cell
def _():
    flops = save_json(compute_grid(), "lora_flops.json")
    return (flops,)


@app.cell(hide_code=True)
def _(flops):
    _rows = []
    for _key, _methods in flops.items():
        _layer, _N, _r = _key.split("|")
        _full = _methods[0][3]
        for _name, _f, _b, _t, _p in _methods:
            _rows.append({"layer": _layer, "batch": _N, "rank": _r, "method": _name, "forward": f"{_f:,}",
                          "backward": f"{_b:,}", "× full": f"{_t / _full:.3f}", "trainable": f"{_p:,}"})
    mo.vstack([mo.md("## All configurations of Table 2 (every count asserted equal to its formula)"),
               mo.ui.table(_rows, selection=None, page_size=12)])
    return


@app.function
def network_cost():
    """Operations, trainable parameters, Adam state and step time of one training step of KryptoNet-40 (N = 64)."""
    base = load_base()
    configs = {"full fine-tuning": full_finetune(base), "LoRA W1, r=1": add_lora(base, 1, (0,)),
               "LoRA W1-W4, r=4": add_lora(base, 4, (0, 2, 4, 6))}
    x, yb = torch.randn(64, 40), torch.randint(0, 2, (64,)).float()
    rows = {}
    for name, net in configs.items():
        with FlopCounterMode(display=False) as fwd:
            out = net(x).squeeze(1)
        loss = nn.functional.binary_cross_entropy_with_logits(out, yb)
        with FlopCounterMode(display=False) as bwd:
            loss.backward()
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-3)
        times = []
        for _ in range(300):
            t0 = time.perf_counter()
            opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(net(x).squeeze(1), yb).backward()
            opt.step()
            times.append(time.perf_counter() - t0)
        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        rows[name] = (fwd.get_total_flops() // 2, bwd.get_total_flops() // 2, trainable, float(np.median(times[50:])))
    return rows


@app.cell
def _():
    with mo.status.spinner("Timing 300 training steps of three KryptoNet-40 variants..."):
        net_cost = save_json(network_cost(), "lora_network_cost.json")
    return (net_cost,)


@app.cell(hide_code=True)
def _(net_cost):
    _full = sum(net_cost["full fine-tuning"][:2])
    mo.vstack([
        mo.md("""
        ## Whole-network cost of one training step of KryptoNet-40 (batch N = 64)
        For LoRA on $W_1$ only, the backward pass must still propagate input gradients through the four frozen
        layers above it, so the saving is about a third — LoRA's large savings are in trainable parameters,
        optimiser state (two Adam moments per trainable parameter) and per-task storage.
        """),
        mo.ui.table([{"method": n, "forward": f"{f:,}", "backward": f"{b:,}", "total": f"{f + b:,}",
                      "× full": f"{(f + b) / _full:.3f}", "trainable": f"{p:,}", "Adam state": f"{2 * p:,}",
                      "median step (ms)": f"{1e3 * t:.2f}"} for n, (f, b, p, t) in net_cost.items()],
                    selection=None, pagination=False),
    ])
    return


if __name__ == "__main__":
    app.run()

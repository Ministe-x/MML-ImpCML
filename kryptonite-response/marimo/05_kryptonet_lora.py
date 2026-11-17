import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from pathlib import Path

    import marimo as mo

    sys.path.insert(0, str(mo.notebook_dir() or Path(__file__).resolve().parent))
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression

    from kmo import cached_or_run, load_cached, save_json, save_figure, track, pm
    from kryptonet import kryptonet
    from krypto.data import load, split
    from krypto.lora import load_base, add_lora, linear_probe, mlp_probe, full_finetune, swap_oracle
    from krypto.mlp import train, accuracy, n_trainable, set_seed
    from krypto.stats import mean_ci
    from krypto.plotting import use_paper_style, FULL, COLORS, pct

    PLOT_BACKEND = use_paper_style()
    SEEDS = (0, 1, 2, 3, 4)
    LRS = (3e-4, 1e-3, 3e-3)
    SWAP = "Rank-1 swap (analytic, no training)"
    TRAINED = ["Linear probe", "MLP probe", "Scratch (same architecture)", "Full fine-tuning",
               "LoRA, W1, r=1", "LoRA, W1, r=2", "LoRA, W1, r=4", "LoRA, W1, r=8",
               "LoRA, W1-W4, r=1", "LoRA, W1-W4, r=4", "LoRA, W1-W4, r=8"]
    METHODS = ["Zero-shot", *TRAINED, SWAP]
    CURVE_METHODS = ["LoRA, W1, r=1", "LoRA, W1-W4, r=4", "Full fine-tuning", "Scratch (same architecture)"]
    SIZES = (100, 200, 400, 800, 1600, 2560)
    PRE_COORDS = [0, 3, 6, 8, 10, 13, 17, 26, 37, 38]      # the 10 coordinates KryptoNet-40 reads (notebook 01)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 05 · Adapting KryptoNet-40 with LoRA (Figure 4, Table 3)

    The mock paper reports that KryptoNet-40, a network pretrained on a related task, does worse than a coin flip on
    Kryptonite-40-Adapt, even with linear and MLP probes on its representation.  Notebook 01 found the reason: Adapt is
    the pretraining task with input coordinate 0 replaced by coordinate 19.  This gives two predictions (**H4**).  Every
    method that keeps the representation frozen must fail, because the network never learned to read $b_{19}$; and a
    rank-1 change of the first layer is enough to redirect it from coordinate 0 to 19, so LoRA with $r=1$ should adapt
    the network from a few thousand examples.

    **LoRA** (`krypto/lora.py`, written from scratch) keeps $W,\mathbf b$ frozen and learns a low-rank correction,
    $\mathbf h = W\mathbf x+\mathbf b+\tfrac{\alpha}{r}B(A\mathbf x)$, with $A\in\mathbb R^{r\times d_\text{in}}$
    Kaiming-initialised and $B\in\mathbb R^{d_\text{out}\times r}$ initialised at zero, so training starts exactly at the
    pretrained network; $\alpha=r$, and the forward pass never forms $BA$.

    We use 5 stratified 64/16/20 splits of the 4,000 labelled examples (800 test points, as in the mock paper).  All
    trainable methods share one recipe: AdamW, batch 64, early stopping on validation accuracy (patience 30, at most 300
    epochs) and a learning rate chosen on validation from $\{3\cdot10^{-4}, 10^{-3}, 3\cdot10^{-3}\}$.  Besides LoRA we
    run the mock paper's methods (zero-shot, linear and MLP probe) and two controls: training the same architecture from
    scratch, which shows whether pretraining helps at all, and full fine-tuning, the obvious alternative to LoRA.
    """)
    return


@app.cell
def _():
    base = load_base()
    X, y = load("kryptonite-40-adapt")
    oracle = swap_oracle(base, 0, 19)
    mo.md(f"KryptoNet-40: **{sum(p.numel() for p in base.parameters()):,}** parameters; "
          f"Kryptonite-40-Adapt: **{len(y):,}** labelled examples.")
    return X, base, oracle, y


@app.function
def make_method(name, base):
    """A fresh copy of KryptoNet-40 (or the same architecture from scratch) with the method's trainable parameters."""
    if name.startswith("LoRA"):
        layers = (0,) if "W1," in name else (0, 2, 4, 6)
        return add_lora(base, int(name.split("r=")[1]), layers)
    return {"Linear probe": lambda: linear_probe(base), "MLP probe": lambda: mlp_probe(base),
            "Scratch (same architecture)": lambda: kryptonet(), "Full fine-tuning": lambda: full_finetune(base)}[name]()


@app.function
def compute_mechanism(base, oracle, X, y):
    """Why zero-shot and probes fail, and why rank 1 is enough."""
    X_sw = X.copy()
    X_sw[:, 0] = X[:, 19]
    dW = oracle[0].weight - base[0].weight
    rng = np.random.default_rng(0)
    B = rng.choice([-1.0, 1.0], size=(20_000, 40))
    Xs = (B + 0.1 * rng.standard_normal(B.shape)).astype(np.float32)
    with torch.no_grad():
        H = base[:-1](torch.from_numpy(Xs)).numpy()                # frozen representation h(4)
    col_norm = base[0].weight.norm(dim=0).detach().numpy()
    out = {"zero_shot_all": accuracy(base, X, y), "zero_shot_swapped_all": accuracy(base, X_sw, y),
           "oracle_all": accuracy(oracle, X, y), "oracle_rank": int(torch.linalg.matrix_rank(dW).item()),
           "probe_bit": {}, "col_norm": {},
           "median_ignored_col_norm": float(np.median(np.delete(col_norm, PRE_COORDS)))}
    for i in (0, 10, 19):
        probe = LogisticRegression(max_iter=3000).fit(H[:15_000], B[:15_000, i] > 0)
        out["probe_bit"][str(i)] = float(probe.score(H[15_000:], B[15_000:, i] > 0))
        out["col_norm"][str(i)] = float(col_norm[i])
    return out


@app.cell
def _(X, base, oracle, y):
    with mo.status.spinner("Mechanism: zero-shot, swap and bit probes (a few seconds)..."):
        mechanism = save_json(compute_mechanism(base, oracle, X, y), "kryptonet_mechanism.json")
    return (mechanism,)


@app.cell(hide_code=True)
def _(mechanism):
    _m = mechanism
    mo.md(rf"""
    ## 1. Mechanism: why zero-shot and probes fail, and why rank 1 is enough

    | check | result |
    |---|---|
    | zero-shot accuracy on all 4,000 Adapt examples | {_m['zero_shot_all']:.4f} |
    | zero-shot after moving coordinate 19 into slot 0 | **{_m['zero_shot_swapped_all']:.4f}** |
    | analytic edit $W_1' = W_1 + (\mathbf w_{{19}}-\mathbf w_0)(\mathbf e_0-\mathbf e_{{19}})^\top$ | rank {_m.get('oracle_rank', 1)}, accuracy **{_m['oracle_all']:.4f}** |

    What does the frozen representation $\mathbf h^{{(4)}}$ (input of the last layer) know about each input bit?

    | bit | linear probe $\mathbf h^{{(4)}}\to b_i$ | first-layer column norm $\lVert\mathbf w_i\rVert$ |
    |---|---|---|
    | $b_0$ | {_m['probe_bit']['0']:.3f} | {_m['col_norm']['0']:.2f} |
    | $b_{{10}}$ | {_m['probe_bit']['10']:.3f} | {_m['col_norm']['10']:.2f} |
    | $b_{{19}}$ | **{_m['probe_bit']['19']:.3f}** | **{_m['col_norm']['19']:.2f}** |

    The median column norm over the 30 coordinates KryptoNet-40 ignores is
    {_m.get('median_ignored_col_norm', float('nan')):.2f}: the network ignores coordinate 19.  Since the Adapt label
    is $s=b_{{19}}\,z$ with $z$ independent of $b_{{19}}$, **every** classifier of a representation that does not
    depend on $b_{{19}}$ has accuracy exactly ½ (Proposition 2b) — zero-shot, linear probe and MLP probe alike.
    """)
    return


@app.function
def compute_methods(base, oracle, X, y):
    """All adaptation methods on the 5 splits, learning rate chosen on validation data."""
    res, lrs, params = {m: [] for m in METHODS}, {m: [] for m in METHODS}, {"Zero-shot": 0, SWAP: 0}
    for seed in SEEDS:
        X_tr, y_tr, X_va, y_va, X_te, y_te = split(X, y, seed, val=0.16, test=0.2)
        res["Zero-shot"].append(accuracy(base, X_te, y_te))
        res[SWAP].append(accuracy(oracle, X_te, y_te))
        for name in TRAINED:
            with track(f"05_{name}_s{seed}"):
                runs = []
                for lr in LRS:
                    set_seed(seed)
                    net, info = train(make_method(name, base), X_tr, y_tr, X_va, y_va, lr=lr, batch_size=64,
                                      seed=seed)
                    runs.append((info["val_acc"], lr, net))
                _, lr, net = max(runs, key=lambda t: t[0])
            params[name] = n_trainable(net)
            res[name].append(accuracy(net, X_te, y_te))
            lrs[name].append(lr)
        print(f"seed {seed}: " + ", ".join(f"{m}: {v[-1]:.3f}" for m, v in res.items()), flush=True)
    return {"accuracy": res, "lr": lrs, "params": params}


@app.function
def compute_sample_efficiency(base, X, y):
    """Accuracy against the number of adaptation examples (nested subsets, learning rate re-chosen per size)."""
    curve = {m: {str(s): [] for s in SIZES} for m in CURVE_METHODS}
    with track("05_sample_efficiency"):
        for seed in SEEDS:
            X_tr, y_tr, X_va, y_va, X_te, y_te = split(X, y, seed, val=0.16, test=0.2)
            order = np.random.default_rng(seed).permutation(len(y_tr))
            for m in CURVE_METHODS:
                for s in SIZES:
                    runs = []
                    for lr in LRS:
                        set_seed(seed)
                        net, info = train(make_method(m, base), X_tr[order[:s]], y_tr[order[:s]], X_va, y_va,
                                          lr=lr, batch_size=64, seed=seed)
                        runs.append((info["val_acc"], net))
                    curve[m][str(s)].append(accuracy(max(runs, key=lambda t: t[0])[1], X_te, y_te))
            print(f"seed {seed}: " + "; ".join(f"{m}: " + " ".join(f"{curve[m][str(s)][-1]:.2f}" for s in SIZES)
                                             for m in CURVE_METHODS), flush=True)
    return curve


@app.cell(hide_code=True)
def _():
    run_methods = mo.ui.run_button(label="Recompute 13 methods × 5 splits (~25 min)")
    run_curve = mo.ui.run_button(label="Recompute sample efficiency (~28 min)")
    mo.vstack([
        mo.md("""
        ## Experiments
        By default the results are read from `results_marimo/` (if re-run here) or from the committed `results/`.
        """),
        mo.hstack([run_methods, run_curve], justify="start"),
    ])
    return run_curve, run_methods


@app.cell
def _(X, base, oracle, run_methods, y):
    with mo.status.spinner("13 adaptation methods x 5 splits x 3 learning rates..."):
        methods, methods_source = cached_or_run(run_methods, "kryptonet_methods.json",
                                                lambda: compute_methods(base, oracle, X, y))
    mo.stop(methods is None, mo.callout("No cached adaptation results: press the button.", kind="warn"))
    return methods, methods_source


@app.cell
def _(X, base, run_curve, y):
    with mo.status.spinner("Sample efficiency: 4 methods x 6 sizes x 5 splits x 3 learning rates..."):
        curve, curve_source = cached_or_run(run_curve, "kryptonet_sample_efficiency.json",
                                            lambda: compute_sample_efficiency(base, X, y))
    mo.stop(curve is None, mo.callout("No cached sample-efficiency results: press the button.", kind="warn"))
    return curve, curve_source


@app.cell(hide_code=True)
def _(methods, methods_source):
    _acc, _lr, _params = methods["accuracy"], methods["lr"], methods["params"]
    mo.vstack([
        mo.md(f"""
        ## 2. Table 3: adaptation methods on 5 splits (800 test points each)
        Results {methods_source}.  ✔ marks the 90% target.
        """),
        mo.ui.table([{"method": m, "trainable params": f"{_params[m]:,}",
                      "test accuracy (%)": pm(_acc[m]) + ("  ✔" if np.mean(_acc[m]) >= 0.9 else ""),
                      "chosen lr": str(_lr.get(m, []))} for m in METHODS], selection=None, pagination=False),
        mo.md(r"""
        Frozen-representation methods sit at chance (the mock paper's "worse than random" 46–50% lies inside the
        ±3.5-point binomial interval around 50%).  Training the same architecture from scratch also stays at chance,
        so pretraining matters.  Full fine-tuning of all 809,473 parameters works but is noisier; LoRA of rank 1 on
        $W_1$ alone (552 parameters) reaches the Bayes level and matches the analytic rank-1 swap.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(curve, curve_source):
    mo.vstack([
        mo.md(f"""
        ## 3. Sample efficiency
        Nested random subsets of each split's training part; validation and test parts fixed; learning rate
        re-chosen on validation for every size.  Results {curve_source} (test accuracy %, mean ± 95% CI).
        """),
        mo.ui.table([{"adaptation examples": s, **{m: pm(curve[m][str(s)]) for m in CURVE_METHODS}} for s in SIZES],
                    selection=None, pagination=False),
    ])
    return


@app.function
def figure4(methods, curve):
    """Figure 4: methods at a glance (left) and accuracy against the number of adaptation examples (right)."""
    show = ["Zero-shot", "Linear probe", "MLP probe", "Scratch (same architecture)", "Full fine-tuning",
            "LoRA, W1-W4, r=4", "LoRA, W1, r=1"]
    label = {"Zero-shot": "zero-shot", "Linear probe": "linear probe", "MLP probe": "MLP probe",
             "Scratch (same architecture)": "from scratch", "Full fine-tuning": "full fine-tuning",
             "LoRA, W1-W4, r=4": r"LoRA $W_{1..4}$, $r{=}4$", "LoRA, W1, r=1": r"LoRA $W_1$, $r{=}1$"}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(FULL, 2.15), gridspec_kw={"width_ratios": [1.15, 1]})
    st = np.array([mean_ci(methods["accuracy"][m]) for m in show]) * 100
    ax.barh(range(len(show)), st[:, 0], xerr=st[:, 1], color=["#bbbbbb"] * 5 + [COLORS["adapt"], COLORS[40]],
            height=0.65, capsize=2)
    for i in range(len(show)):
        ax.text(st[i, 0] + st[i, 1] + 1, i, f"{st[i, 0]:.1f}", va="center")
    ax.axvline(90, color="k", ls="--", lw=1)
    ax.set_yticks(range(len(show))); ax.set_yticklabels([label[m] for m in show]); ax.invert_yaxis()
    ax.set_xlim(40, 106); ax.set_xlabel(pct("Test accuracy (%)")); ax.set_xticks([40, 50, 60, 70, 80, 90, 100])
    for m, col, mk in (("LoRA, W1, r=1", COLORS[40], "o"), ("LoRA, W1-W4, r=4", COLORS["adapt"], "D"),
                       ("Full fine-tuning", "#777777", "s"), ("Scratch (same architecture)", "#bbbbbb", "^")):
        c = np.array([mean_ci(curve[m][str(s)]) for s in SIZES]) * 100
        bx.errorbar(SIZES, c[:, 0], yerr=c[:, 1], color=col, marker=mk, ms=3.5, capsize=2, label=label[m])
    bx.axhline(90, color="k", ls="--", lw=1)
    bx.set_xscale("log"); bx.set_xticks(SIZES); bx.set_xticklabels([str(s) for s in SIZES]); bx.minorticks_off()
    bx.set_xlabel("Adaptation examples"); bx.set_ylabel(pct("Test accuracy (%)")); bx.set_ylim(45, 100)
    fig.legend(*bx.get_legend_handles_labels(), ncol=4, loc="outside upper center", handlelength=1.8,
               columnspacing=1.2)
    return save_figure(fig, "fig4_kryptonet")


@app.cell(hide_code=True)
def _(curve, methods):
    mo.vstack([
        mo.md("## Figure 4"),
        figure4(methods, curve),
        mo.md(r"""
        **Reading Figure 4.** Left: the frozen-representation methods and training from scratch stay at chance, as
        predicted.  Full fine-tuning of all 809,473 parameters reaches about 90% with a wide interval, while LoRA on the
        first layer with rank 1 (552 trainable parameters) reaches the Bayes level of about 96%, matching the analytic
        rank-1 edit.  Right: rank-1 LoRA is already near 95% with 800 examples, full fine-tuning needs all 2,560 to reach
        about 90%, and training from scratch never leaves chance in this range: **H4 is supported**.
        """),
    ])
    return


@app.function
def lora_lab(base, oracle, X, y, r, layers, lr, seed):
    """Train one LoRA model on one split and describe its first-layer update."""
    X_tr, y_tr, X_va, y_va, X_te, y_te = split(X, y, seed, val=0.16, test=0.2)
    set_seed(seed)
    net, info = train(add_lora(base, r, layers), X_tr, y_tr, X_va, y_va, lr=lr, batch_size=64, seed=seed)
    dW = net[0].delta_weight().detach()
    dW_swap = (oracle[0].weight - base[0].weight).detach()
    return {"test_acc": accuracy(net, X_te, y_te), "val_acc": info["val_acc"], "epochs": info["epochs"],
            "params": n_trainable(net), "col_norms": dW.norm(dim=0).numpy().tolist(),
            "A": net[0].A.detach().numpy().tolist(),
            "cosine": torch.nn.functional.cosine_similarity(dW.flatten(), dW_swap.flatten(), dim=0).item()}


@app.cell(hide_code=True)
def _():
    lab_r = mo.ui.dropdown(options={str(r): r for r in (1, 2, 4, 8, 16)}, value="1", label="rank r")
    lab_layers = mo.ui.dropdown(options={"W1 only": (0,), "W1–W4": (0, 2, 4, 6)}, value="W1 only", label="layers")
    lab_lr = mo.ui.dropdown(options={str(v): v for v in LRS}, value="0.003", label="learning rate")
    lab_seed = mo.ui.dropdown(options={str(s): s for s in SEEDS}, value="0", label="split seed")
    lab_run = mo.ui.run_button(label="Train (~10–40 s)")
    mo.vstack([
        mo.md(r"""
        ## 4. LoRA lab: what does LoRA learn?
        Train one LoRA model and look at its first-layer update $\Delta W_1=\tfrac{\alpha}{r}BA$: which input
        coordinates does it act on, and how close is it to the analytic swap of coordinates 0 and 19?  Before you
        train, the cached result of the default setting ($r=1$ on $W_1$, lr $3\cdot10^{-3}$, split 0) is shown.
        """),
        mo.hstack([lab_r, lab_layers, lab_lr, lab_seed, lab_run], justify="start"),
    ])
    return lab_layers, lab_lr, lab_r, lab_run, lab_seed


@app.cell
def _(X, base, lab_layers, lab_lr, lab_r, lab_run, lab_seed, oracle, y):
    if lab_run.value:
        with mo.status.spinner(f"Training LoRA r={lab_r.value} on {lab_layers.selected_key}..."):
            _lab = lora_lab(base, oracle, X, y, lab_r.value, lab_layers.value, lab_lr.value, lab_seed.value)
        if lab_r.value == 1 and lab_layers.value == (0,):
            save_json({"A": np.ravel(_lab["A"]).tolist(), "cosine": _lab["cosine"]}, "kryptonet_lora_inspection.json")
        lab_title = (f"r = {lab_r.value} on {lab_layers.selected_key}, lr {lab_lr.value}, split {lab_seed.value}: "
                     f"test accuracy **{100 * _lab['test_acc']:.2f}%** (validation {100 * _lab['val_acc']:.2f}%, "
                     f"{_lab['epochs']} epochs), **{_lab['params']:,}** trainable parameters, "
                     f"cosine(ΔW₁, analytic ΔW₁) = **{_lab['cosine']:.3f}**")
        lab_profile, lab_ylabel = np.array(_lab["col_norms"]), r"column norm of $\Delta W_1$"
    else:
        _cached, _where = load_cached("kryptonet_lora_inspection.json")
        mo.stop(_cached is None, mo.md("Press **Train** to run the lab."))
        lab_title = (f"Cached ({_where}/kryptonet_lora_inspection.json): r = 1 on W1, lr 0.003, split 0; "
                     f"cosine(ΔW₁, analytic ΔW₁) = **{_cached['cosine']:.3f}**")
        lab_profile, lab_ylabel = np.abs(np.ravel(_cached["A"])), r"$|A_{1j}|$ (proportional to column norm)"
    return lab_profile, lab_title, lab_ylabel


@app.cell(hide_code=True)
def _(lab_profile, lab_title, lab_ylabel):
    _fig, _ax = plt.subplots(figsize=(FULL, 1.9))
    _colors = [COLORS[40] if j in (0, 19) else ("#9ecae1" if j in PRE_COORDS else "#cccccc") for j in range(40)]
    _ax.bar(range(40), lab_profile, color=_colors, width=0.75)
    _ax.set_xlabel("Input coordinate $j$"); _ax.set_ylabel(lab_ylabel); _ax.set_xticks([0, 10, 19, 30, 39])
    _top = np.argsort(-lab_profile)[:4]
    mo.vstack([
        mo.md(lab_title),
        _fig,
        mo.md(f"Largest entries at coordinates `{_top.tolist()}` (dark: 0 and 19; light blue: other coordinates "
              f"KryptoNet-40 reads).  A rank-1 update concentrated on coordinates 0 and 19 is exactly "
              f"\"read coordinate 19 where the network used to read coordinate 0\"."),
    ])
    return


if __name__ == "__main__":
    app.run()

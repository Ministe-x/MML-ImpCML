import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from pathlib import Path

    import marimo as mo

    sys.path.insert(0, str(mo.notebook_dir() or Path(__file__).resolve().parent))
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

    from kmo import cached_or_run, load_cached, save_json, save_figure, track, pm
    from krypto.data import load, split, NS, SEEDS, TARGET
    from krypto.polynomial import poly_logreg, SparsePolyLogReg
    from krypto.selection import greedy_forward_selection
    from krypto.mlp import mlp, train, accuracy, set_seed
    from krypto.stats import mean_ci
    from krypto.plotting import use_paper_style, FULL, COLORS, pct

    PLOT_BACKEND = use_paper_style()
    CURVE_MODELS = ("MLP", "Sparse poly", "Quadratic LR")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 04 · Does more data help? (Figure 3)

    The mock paper's second claim is that quadrupling the training data barely helps, so the difficulty must be
    intrinsic.  But it only scaled the quadratic model, whose accuracy is capped at 56.5% even with infinite data
    (notebook 02): its flat curve shows approximation error, not that data is useless.  Our hypothesis (**H3**) is that
    more data does help every model whose hypothesis class contains a good classifier.

    We train on nested random subsets of each training split, from 250 points up to the full split (7,200 / 12,000 /
    19,200 points for $n$ = 20 / 30 / 40), and keep the validation and test sets fixed, so that only the amount of
    training data changes; nesting the subsets makes the curves smoother than independent draws.  We compare the mock
    paper's quadratic model with our MLP (configuration from notebook 03) and our sparse polynomial model.
    """)
    return


@app.function
def sizes_for(n_train):
    return [m for m in (250, 500, 1000, 2000, 4000, 8000, 16000) if m < n_train] + [n_train]


@app.function
def compute_curves(mlp_config):
    """Test accuracy of the three models on nested subsets of each split's training part."""
    curves = {}
    for n in NS:
        X, y = load(f"kryptonite-{n}")
        c = {"sizes": None, **{m: {} for m in CURVE_MODELS}}
        with track(f"04_learning_curves_n{n}"):
            for seed in SEEDS:
                X_tr, y_tr, X_va, y_va, X_te, y_te = split(X, y, seed)
                order = np.random.default_rng(seed).permutation(len(y_tr))        # nested subsets
                c["sizes"] = sizes_for(len(y_tr))
                for m in c["sizes"]:
                    Xm, ym = X_tr[order[:m]], y_tr[order[:m]]
                    set_seed(seed)
                    net, _ = train(mlp(n, 512, **mlp_config[str(n)]), Xm, ym, X_va, y_va, seed=seed)
                    c["MLP"].setdefault(str(m), []).append(accuracy(net, X_te, y_te))
                    S, _ = greedy_forward_selection(Xm, ym, X_va, y_va)
                    fits = [SparsePolyLogReg(S, 10, C).fit(Xm, ym) for C in (0.85, 10, 100, 1000, 10000)]
                    c["Sparse poly"].setdefault(str(m), []).append(max(fits, key=lambda f: f.score(X_va, y_va))
                                                                   .score(X_te, y_te))
                    c["Quadratic LR"].setdefault(str(m), []).append(poly_logreg(Xm, ym, X_te, y_te, 2))
                    print(f"n={n} seed={seed} N={m:6d}: " + "  ".join(f"{k} {c[k][str(m)][-1]:.3f}"
                                                                     for k in CURVE_MODELS), flush=True)
        curves[str(n)] = c
    return curves


@app.cell(hide_code=True)
def _():
    run_curves = mo.ui.run_button(label="Recompute learning curves (~1.4 h)")
    mo.vstack([
        mo.md("""
        ## Experiment
        Uses the MLP configuration chosen in notebook 03 (`mlp_config.json`) and the quadratic model's
        infinite-data limit from notebook 02 (`poly_limit.json`).  Results are read from the cache unless you
        press the button.
        """),
        run_curves,
    ])
    return (run_curves,)


@app.cell
def _(run_curves):
    mlp_config, _ = load_cached("mlp_config.json")
    mo.stop(mlp_config is None, mo.callout("Run notebook 03 first (no mlp_config.json).", kind="warn"))
    with mo.status.spinner("Learning curves: 3 models x 5 seeds x up to 8 sizes x 3 datasets..."):
        curves, curves_source = cached_or_run(run_curves, "learning_curves.json", lambda: compute_curves(mlp_config))
    mo.stop(curves is None, mo.callout("No cached learning curves: press the button.", kind="warn"))
    _lim, _ = load_cached("poly_limit.json")
    QUAD_LIMIT = _lim["limit"]["2"] if _lim else 0.5 * (1 + 9 / 64) * (1 - 2 * 0.039) + 0.039
    return QUAD_LIMIT, curves, curves_source, mlp_config


@app.function
def n_at_target(sizes, means, target=TARGET):
    """Smallest training-set size reaching `target` on the mean curve (log-linear interpolation)."""
    for (m0, a0), (m1, a1) in zip(zip(sizes, means), zip(sizes[1:], means[1:])):
        if a0 < target <= a1:
            return float(np.exp(np.log(m0) + (target - a0) / (a1 - a0) * (np.log(m1) - np.log(m0))))
    return None if means[-1] < target else float(sizes[0])


@app.cell
def _(curves):
    thresholds = {}
    for _n in NS:
        for _model in CURVE_MODELS:
            _sizes = curves[str(_n)]["sizes"]
            thresholds[f"{_n}|{_model}"] = n_at_target(_sizes, [np.mean(curves[str(_n)][_model][str(m)])
                                                                for m in _sizes])
    save_json(thresholds, "learning_curve_thresholds.json")
    return (thresholds,)


@app.cell(hide_code=True)
def _(QUAD_LIMIT, curves, curves_source, mlp_config, thresholds):
    _rows = []
    for _model in CURVE_MODELS:
        _row = {"model": _model}
        for _n in NS:
            _full = curves[str(_n)]["sizes"][-1]
            _t = thresholds[f"{_n}|{_model}"]
            _row[f"n={_n}: full data (%)"] = pm(curves[str(_n)][_model][str(_full)])
            _row[f"n={_n}: N for 90%"] = "never" if _t is None else f"{_t:,.0f}"
        _rows.append(_row)
    mo.vstack([
        mo.md(f"""
        ## Smallest training-set size reaching the 90% target
        Mean over 5 seeds, log-linear interpolation between measured sizes.  Learning curves {curves_source};
        MLP configuration `{mlp_config}`; quadratic infinite-data limit {100 * QUAD_LIMIT:.2f}%.
        """),
        mo.ui.table(_rows, selection=None, pagination=False),
    ])
    return


@app.function
def figure3(curves, quad_limit):
    """Figure 3: test accuracy against the number of training points, one panel per n."""
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.2), sharey=True)
    styles = {"MLP": ("-", "o", None), "Sparse poly": ("--", "D", None), "Quadratic LR": (":", "s", COLORS["gray"])}
    labels = {"MLP": "MLP (ours)", "Sparse poly": "sparse poly LR (ours)", "Quadratic LR": "quadratic LR (mock)"}
    for ax, n in zip(axes, NS):
        c = curves[str(n)]
        sizes = c["sizes"]
        ax.axvspan(sizes[-1] / 4, sizes[-1], color="#eeeeee", lw=0, label="mock paper's range")
        ax.axhline(100 * TARGET, color="#bbbbbb", ls="--", lw=1)
        ax.axhline(100 * quad_limit, color=COLORS["gray"], lw=0.8, ls=(0, (1, 3)))
        for model, (ls, mk, col) in styles.items():
            st = np.array([mean_ci(c[model][str(m)]) for m in sizes]) * 100
            ax.errorbar(sizes, st[:, 0], yerr=st[:, 1], ls=ls, marker=mk, ms=3.2, capsize=2,
                        color=col or COLORS[n], label=labels[model])
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([250, 1000, 4000, 16000]))
        ax.xaxis.set_major_formatter(FixedFormatter(["250", "1k", "4k", "16k"]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Training points $N$"); ax.set_ylim(45, 100)
        ax.text(0.03, 0.97, f"$n={n}$", transform=ax.transAxes, va="top")
    axes[0].set_ylabel(pct("Test accuracy (%)"))
    axes[0].text(260, 100 * quad_limit + 1.2, "quadratic limit", color=COLORS["gray"])
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, ncol=4, loc="outside upper center", handlelength=1.8, columnspacing=1.0)
    return save_figure(fig, "fig3_learning_curves")


@app.cell(hide_code=True)
def _(QUAD_LIMIT, curves):
    mo.vstack([
        mo.md("## Figure 3"),
        figure3(curves, QUAD_LIMIT),
        mo.md(r"""
        **Reading Figure 3.** The mock paper only looked inside the grey band (25–100% of the training split) and
        only at the quadratic model, whose curve can never exceed its infinite-data limit (dotted, 56.5%).  Models
        that can represent the label function improve sharply with data: the sparse polynomial model crosses 90% at
        about 2,900 points for every $n$, and the MLP shows the plateau-then-jump typical of SGD on sparse parities,
        with the jump *inside* the range the mock paper examined.  The data it needs grows with the number of
        distractor coordinates, far more slowly than the $\binom{n+6}{6}$ features of a degree-6 kernel:
        **H3 is supported**.
        """),
    ])
    return


@app.cell
def _():
    n_view = mo.ui.dropdown(options={f"n = {n}": n for n in NS}, value="n = 40", label="Dataset")
    return (n_view,)


@app.cell(hide_code=True)
def _(curves, n_view):
    _c = curves[str(n_view.value)]
    mo.vstack([
        mo.md("### Every point of the curves (mean ± 95% CI over 5 seeds, %)"),
        n_view,
        mo.ui.table([{"N": m, **{k: pm(_c[k][str(m)]) for k in CURVE_MODELS}} for m in _c["sizes"]],
                    selection=None, pagination=False),
    ])
    return


if __name__ == "__main__":
    app.run()

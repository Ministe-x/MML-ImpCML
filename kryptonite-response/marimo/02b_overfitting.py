import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import sys
    import warnings
    from pathlib import Path

    import marimo as mo

    sys.path.insert(0, str(mo.notebook_dir() or Path(__file__).resolve().parent))
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    from kmo import cached_or_run, load_json, save_figure, track
    from krypto.data import load, split, mock_split, NS, SEEDS
    from krypto.polynomial import parity_features, agreements, polyfeatures_kernel_table, kernel_logreg
    from krypto.stats import mean_ci
    from krypto.plotting import use_paper_style, FULL, COLORS, pct

    PLOT_BACKEND = use_paper_style()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)    # the largest C values converge slowly
    DEGREES = (6, 10)
    C_VALUES = [0.1, 0.85, 3, 10, 30, 100, 300, 1000, 3000, 10000, 1e5, 1e6]
    C_GRID = [0.85, 10, 100, 1000, 10000]          # the candidates of notebook 02
    PARTS = ("train", "val", "test")
    FIT_DEGREES = list(range(1, 11))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 02b · Overfitting or underfitting? (Figure 5)

    Notebook 02 shows that the test accuracy of the polynomial models depends strongly on the inputs and on $C$.  Test
    accuracy alone, however, cannot tell us *how* a model fails: a model that memorises its training data and a model
    that is too constrained to fit it can end up with the same test score.  Comparing training with test accuracy
    separates the two.  The Bayes level $1-\eta\approx96.1\%$ is a convenient yardstick: no classifier can be right on
    more than about 96% of fresh labels, so a model that fits more of its *training* labels than that is fitting label
    noise.

    We ask two questions.  First, does choosing $C$ on validation data, as notebook 02 does, make our sparse model
    overfit?  Second, are the mock paper's low accuracies a case of overfitting or of underfitting?  Both parts reuse the
    selected coordinates and chosen values of $C$ of notebook 02 (`poly_sparse.json`).
    """)
    return


@app.cell
def _():
    sparse = load_json("poly_sparse.json")
    lim = load_json("poly_limit.json")
    mo.stop(sparse is None or lim is None,
            mo.callout("Run notebook 02 first: this notebook needs poly_sparse.json and poly_limit.json.", kind="warn"))
    eta = lim["eta"]
    return eta, lim, sparse


@app.function
def compute_sweep(sparse):
    """Part 1: train/validation/test accuracy of the sparse model on the selected bits for a grid of C."""
    acc = {str(n): {str(d): {p: [] for p in PARTS} for d in DEGREES} for n in NS}
    cells = {}
    for n in NS:
        X, y = load(f"kryptonite-{n}")
        with track(f"02_overfit_n{n}"):
            for seed in SEEDS:
                X_tr, y_tr, X_va, y_va, X_te, y_te = split(X, y, seed)
                S = sparse[str(n)]["S"][seed]
                if seed == 0:
                    count = np.bincount((X_tr[:, S] < 0).astype(int) @ (1 << np.arange(len(S))),
                                        minlength=1 << len(S))
                    cells[str(n)] = {"train": len(y_tr), "empty": int((count == 0).sum()),
                                     "median": float(np.median(count))}
                for d in DEGREES:
                    F = [parity_features(Z, S, d) for Z in (X_tr, X_va, X_te)]
                    lr = LogisticRegression(fit_intercept=False, max_iter=5000, tol=1e-8, warm_start=True)
                    rows = []
                    for C in C_VALUES:                       # increasing C, warm-started from the previous fit
                        lr.set_params(C=C).fit(F[0], y_tr)
                        rows.append([lr.score(f, t) for f, t in zip(F, (y_tr, y_va, y_te))])
                    for p, column in zip(PARTS, np.array(rows).T):
                        acc[str(n)][str(d)][p].append(column.tolist())
    return {"C": C_VALUES, "accuracy": acc, "cells_seed0": cells}


@app.function
def compute_fit(sparse):
    """Part 2: train and test accuracy against degree for the three models of Figure 2."""
    fit = {}
    for n in NS:
        X, y = load(f"kryptonite-{n}")
        fit[str(n)] = {"mock": {}, "fixed": {}, "tuned": {}}
        with track(f"02_fit_mock_n{n}"):
            X_tr, y_tr, _, _, X_te, y_te = mock_split(X, y)
            M_tr, M_te = agreements(X_tr, X_tr, n), agreements(X_te, X_tr, n)
            for d in FIT_DEGREES:
                table = polyfeatures_kernel_table(n, d).astype(np.float32)
                K = table[M_tr]                                  # training kernel matrix, N x N
                a = kernel_logreg(K, y_tr, C=0.85)
                fit[str(n)]["mock"][str(d)] = {"train": float(((K @ a > 0) == y_tr).mean()),
                                               "test": float(((table[M_te] @ a > 0) == y_te).mean())}
                del K                                            # free the ~1.5 GB matrix before the next degree
        with track(f"02_fit_sparse_n{n}"):
            for key in ("fixed", "tuned"):
                fit[str(n)][key] = {str(d): {"train": [], "test": []} for d in FIT_DEGREES}
            for seed in SEEDS:
                X_tr, y_tr, _, _, X_te, y_te = split(X, y, seed)
                S = sparse[str(n)]["S"][seed]
                for d in FIT_DEGREES:
                    F_tr, F_te = parity_features(X_tr, S, d), parity_features(X_te, S, d)
                    for key, C in (("fixed", 0.85), ("tuned", sparse[str(n)]["C"][str(d)][seed])):
                        lr = LogisticRegression(C=C, fit_intercept=False, max_iter=5000, tol=1e-8).fit(F_tr, y_tr)
                        fit[str(n)][key][str(d)]["train"].append(lr.score(F_tr, y_tr))
                        fit[str(n)][key][str(d)]["test"].append(lr.score(F_te, y_te))
    return fit


@app.cell(hide_code=True)
def _():
    run_sweep = mo.ui.run_button(label="Recompute part 1 (~4 min)")
    run_fit = mo.ui.run_button(label="Recompute part 2 (~6 min, ~4 GB RAM)")
    mo.vstack([
        mo.md("""
        ## Experiments
        By default the results are read from `results_marimo/` (if you re-ran them here) or from the committed
        `results/`; press a button to recompute them from scratch.
        """),
        mo.hstack([run_sweep, run_fit], justify="start"),
    ])
    return run_fit, run_sweep


@app.cell
def _(run_sweep, sparse):
    with mo.status.spinner("Part 1: sweeping C on the selected bits..."):
        sweep, sweep_source = cached_or_run(run_sweep, "poly_overfit.json", lambda: compute_sweep(sparse))
    mo.stop(sweep is None, mo.callout("No cached result for part 1: press its button.", kind="warn"))
    return sweep, sweep_source


@app.cell
def _(run_fit, sparse):
    with mo.status.spinner("Part 2: train and test accuracy against degree..."):
        fit, fit_source = cached_or_run(run_fit, "poly_fit_diagnosis.json", lambda: compute_fit(sparse))
    mo.stop(fit is None, mo.callout("No cached result for part 2: press its button.", kind="warn"))
    return fit, fit_source


@app.function
def sweep_rows(sweep, eta):
    """One row per n and degree: train-test gaps and the highest training accuracy over the C grid."""
    rows = []
    hi = [i for i, C in enumerate(sweep["C"]) if C >= 1000]
    for n in NS:
        c = sweep["cells_seed0"][str(n)]
        for d in DEGREES:
            a = {p: np.array(sweep["accuracy"][str(n)][str(d)][p]) for p in PARTS}     # seeds x C
            gap = (a["train"] - a["test"]).mean(0)
            rows.append({"n": n, "d": d, "training points": c["train"], "median per cell": int(c["median"]),
                         "gap at C = 0.85 (pts)": f"{100 * gap[sweep['C'].index(0.85)]:+.1f}",
                         "largest gap, C ≥ 1000 (pts)": f"{100 * gap[hi].max():+.1f}",
                         "highest train accuracy (%)": f"{100 * a['train'].max():.1f}",
                         "Bayes level (%)": f"{100 * (1 - eta):.1f}"})
    return rows


@app.function
def figure_sweep(sweep, eta):
    """Supplementary figure: train, validation and test accuracy against C."""
    Cs = np.array(sweep["C"])
    fig, axes = plt.subplots(len(DEGREES), len(NS), figsize=(FULL, 3.6), sharex=True, sharey=True)
    styles = {"train": dict(color="k", ls="-", label="train"),
              "val": dict(color=COLORS["gray"], ls="--", label="validation"),
              "test": dict(ls="-", marker="o", ms=3, label="test")}
    for row, d in enumerate(DEGREES):
        for col, n in enumerate(NS):
            ax = axes[row, col]
            a = {p: np.array(sweep["accuracy"][str(n)][str(d)][p]) for p in PARTS}
            ax.axhline(100 * (1 - eta), color="#bbbbbb", ls=":", lw=1.2, label=r"Bayes $1-\eta$")
            ax.axvspan(min(C_GRID), max(C_GRID), color="#f2f2f2", zorder=0, label="grid of nb 02")
            for p in PARTS:
                st = np.array([mean_ci(a[p][:, i]) for i in range(len(Cs))]) * 100
                kw = dict(styles[p], color=styles[p].get("color", COLORS[n]))
                ax.plot(Cs, st[:, 0], **kw)
                ax.fill_between(Cs, st[:, 0] - st[:, 1], st[:, 0] + st[:, 1], color=kw["color"], alpha=0.2, lw=0)
            best = [C_GRID[int(np.argmax([v[sweep["C"].index(c)] for c in C_GRID]))] for v in a["val"]]
            ax.axvline(np.median(best), color=COLORS[n], ls="-.", lw=1, label=r"median chosen $C$")
            ax.set_xscale("log")
            ax.set_ylim(70, 100)
            ax.text(0.97, 0.05, f"$n={n}$, $d={d}$", transform=ax.transAxes, ha="right", va="bottom")
            if row == len(DEGREES) - 1:
                ax.set_xlabel("Inverse regularisation $C$")
            if col == 0:
                ax.set_ylabel(pct("Accuracy (%)"))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="outside upper center", handlelength=2.0, columnspacing=1.2)
    return save_figure(fig, "figS_overfit")


@app.cell(hide_code=True)
def _(eta, sweep, sweep_source):
    mo.vstack([
        mo.md(f"""
        ## 1. Does choosing $C$ on validation data overfit?
        For each $n$, split and degree $d\\in\\{{6,10\\}}$ we refit the sparse model on the 10 selected bits for a grid
        of $C$ from below the mock paper's 0.85 to an essentially unregularised $10^6$.  If tuning $C$ led to
        overfitting, training accuracy would climb above the Bayes level at large $C$ while test accuracy falls.
        Results {sweep_source}.
        """),
        mo.ui.table(sweep_rows(sweep, eta), selection=None, pagination=False),
        figure_sweep(sweep, eta),
        mo.md(r"""
        **Reading the figure.** At the mock paper's $C=0.85$ all three accuracies lie well below the Bayes level: the
        model is held back by its regulariser.  As $C$ grows, training and test accuracy rise together and level off at
        about 96%.  Training accuracy never climbs much above the Bayes level, and the gap to test accuracy stays at
        about one point even without regularisation.  Choosing $C$ on validation data therefore does not overfit here,
        as expected from 1,024 features estimated from 7,200–19,200 training points (a median of 7–19 per cell).
        """),
    ])
    return


@app.function
def fit_rows(fit, n):
    """Train and test accuracy (%) of the three models for one n, one row per degree."""
    rows = []
    for d in FIT_DEGREES:
        row = {"degree": d}
        for key, name in (("mock", "all bits"), ("fixed", "10 bits, C = 0.85"), ("tuned", "10 bits, tuned C")):
            r = fit[str(n)][key][str(d)]
            row[f"{name}: train"] = f"{100 * np.mean(r['train']):.1f}"
            row[f"{name}: test"] = f"{100 * np.mean(r['test']):.1f}"
        rows.append(row)
    return rows


@app.function
def figure5(fit, limit):
    """Figure 5: train (open markers) and test (filled markers) accuracy against degree; shading = the gap."""
    fig, axes = plt.subplots(1, len(NS), figsize=(FULL, 2.3), sharey=True)
    d = np.array(FIT_DEGREES)
    for ax, n in zip(axes, NS):
        ax.plot(d, [100 * limit[str(k)] for k in d], color="k", ls=":", lw=1.2, label=r"$N\to\infty$ limit")
        for key, color, ls, mk, lab in (("mock", COLORS["gray"], "-", "o", r"all $n$ bits (mock), $C{=}0.85$"),
                                        ("fixed", COLORS[n], "--", "s", r"10 selected bits, $C{=}0.85$"),
                                        ("tuned", COLORS[n], "-", "D", r"10 selected bits, tuned $C$")):
            r = fit[str(n)][key]
            tr = np.array([100 * np.mean(r[str(k)]["train"]) for k in d])
            te = np.array([100 * np.mean(r[str(k)]["test"]) for k in d])
            ax.fill_between(d, te, tr, color=color, alpha=0.12, lw=0)
            ax.plot(d, tr, color=color, ls=ls, lw=1, marker=mk, ms=3.5, mfc="white", alpha=0.9)
            ax.plot(d, te, color=color, ls=ls, marker=mk, ms=3.5, label=lab)
        ax.set_xlabel("Polynomial degree $d$")
        ax.set_xticks([1, 2, 4, 6, 8, 10]); ax.set_ylim(48, 101)
        ax.text(0.03, 0.97, f"$n={n}$", transform=ax.transAxes, va="top")
    axes[0].set_ylabel(pct("Accuracy (%)"))
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [Line2D([], [], color="k", marker="o", ms=3.5, mfc="white", lw=1, alpha=0.9),   # legend keys for
                Line2D([], [], color="k", marker="o", ms=3.5)]                                 # train vs test
    labels += ["train (open markers)", "test (filled markers)"]
    fig.legend(handles, labels, ncol=3, loc="outside upper center", handlelength=2.0, columnspacing=1.2)
    return save_figure(fig, "fig5_fit")


@app.cell
def _():
    n_view = mo.ui.dropdown(options={f"n = {n}": n for n in NS}, value="n = 40", label="Dataset")
    return (n_view,)


@app.cell(hide_code=True)
def _(fit, fit_source, lim, n_view):
    mo.vstack([
        mo.md(f"""
        ## 2. Is the mock paper's failure overfitting or underfitting?
        Training and test accuracy against degree for the three models of Figure 2: the mock paper's kernel model on
        all $n$ bits (its own split, $C=0.85$), and the same model on the 10 selected bits with $C=0.85$ or with the
        $C$ chosen in notebook 02 (5 splits).  Notebook 02 did not record the mock model's training accuracy, so it is
        refitted here with the mock paper's code.  Results {fit_source}.
        """),
        n_view,
        mo.ui.table(fit_rows(fit, n_view.value), selection=None, pagination=False),
        mo.md("## Figure 5"),
        figure5(fit, lim["limit"]),
        mo.md(r"""
        **Reading Figure 5.** The three models fail, or succeed, in three different ways.  On all $n$ bits the mock
        paper's model *overfits*: from degree 3 on, its training accuracy climbs steeply, to 98.8–100% at degree 10,
        above the Bayes level, so it has memorised label noise; its test accuracy meanwhile stays below 59%.  With
        $\binom{n+d}{d}$ features it can fit almost any labelling of the training points, and $C=0.85$ does not
        prevent that.  On the 10 selected bits the same $C$ has the opposite effect: from degree 4 on, training and test
        accuracy both stay below the infinite-data limit, so the model *underfits*, because the regulariser shrinks the
        high-degree parities too strongly.  Once $C$ is chosen on validation data, training and test accuracy agree to
        within half a point and reach the limit from degree 6.  A single fixed $C$ cannot suit both feature spaces: the
        mock paper's low numbers reflect the dimension of its inputs and an untuned regulariser, not a limit of what
        polynomials can represent.
        """),
    ])
    return


if __name__ == "__main__":
    app.run()

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
    import matplotlib.pyplot as plt
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    from kmo import cached_or_run, track, pm
    from krypto.data import load, split, NS, SEEDS
    from krypto.polynomial import poly_logreg, mock_kernel_accuracies, SparsePolyLogReg
    from krypto.selection import greedy_forward_selection
    from krypto.mlp import mlp, train, accuracy, set_seed
    from krypto.stats import mean_ci
    from krypto.plotting import use_paper_style, FULL, COLORS, pct

    PLOT_BACKEND = use_paper_style()
    GRID = [{"depth": d, "dropout": p} for d in (2, 3) for p in (0.0, 0.2)]
    MODELS = ["Logistic regression (linear)", "Quadratic LR (mock paper)", "Poly-kernel LR, d=10, all bits (mock)",
              "k-nearest neighbours (k tuned)", "RBF-kernel SVM", "Random forest (300 trees)",
              "Gradient-boosted trees", "MLP, all n inputs (ours)", "Sparse polynomial LR (ours)"]
    REPRESENTATION = ["fixed"] * 5 + ["greedy splits"] * 2 + ["learned", "selected"]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 03 · Which learners solve Kryptonite-$n$? (Table 1)

    Notebooks 01 and 02 tell us what a successful learner has to do: find ten relevant bits among $n$ and model
    degree-6 interactions between them.  Here we ask which standard learners manage this (**H2**).  We expect learners
    that *learn their features* (an MLP trained by gradient descent, or our explicit selection followed by a polynomial)
    to reach the 90% target.  Learners whose representation is fixed in advance should stay near chance: kernel methods
    and $k$-NN measure similarity over all $n$ coordinates, so the $n-10$ irrelevant ones dominate; trees grow by
    greedy splits on single coordinates, and no single coordinate is informative here.

    Every model sees the same 5 stratified 60/20/20 splits, so the comparison is paired.  Hyper-parameters are chosen
    on the validation split only, each test split is used once per model, and we report the mean test accuracy with a
    95% Student-$t$ interval over the 5 splits.  This notebook also chooses the MLP configuration used by notebooks 04
    and 07.
    """)
    return


@app.function
def compute_mlp_grid():
    """Choose the MLP configuration once, on the validation split of seed 0, and reuse it everywhere else.

    Width 512, AdamW with lr 1e-3, batch 128, early stopping (patience 30, at most 300 epochs); the grid is
    depth {2, 3} x dropout {0, 0.2}.  Returns {n: best configuration}."""
    config = {}
    for n in NS:
        X, y = load(f"kryptonite-{n}")
        X_tr, y_tr, X_va, y_va, _, _ = split(X, y, 0)
        scores = []
        with track(f"03_mlp_grid_n{n}"):
            for cfg in GRID:
                set_seed(0)
                _, info = train(mlp(n, 512, **cfg), X_tr, y_tr, X_va, y_va, seed=0)
                scores.append((info["val_acc"], cfg))
                print(f"n={n} {cfg}: val {info['val_acc']:.4f} ({info['epochs']} epochs)", flush=True)
        config[str(n)] = max(scores, key=lambda t: t[0])[1]
    return config


@app.function
def compute_models(mlp_config):
    """Nine learners on the five shared splits; returns test accuracies and wall-clock seconds per fit."""
    acc = {str(n): {m: [] for m in MODELS} for n in NS}
    sec = {str(n): {m: [] for m in MODELS} for n in NS}

    def knn(X_tr, y_tr, X_va, y_va, X_te, y_te):
        k = max([1, 5, 15, 51, 151], key=lambda k: KNeighborsClassifier(k).fit(X_tr, y_tr).score(X_va, y_va))
        return KNeighborsClassifier(k).fit(X_tr, y_tr).score(X_te, y_te)

    def mlp_fit(n, seed, X_tr, y_tr, X_va, y_va, X_te, y_te):
        set_seed(seed)
        net, _ = train(mlp(n, 512, **mlp_config[str(n)]), X_tr, y_tr, X_va, y_va, seed=seed)
        return accuracy(net, X_te, y_te)

    def sparse_poly(X_tr, y_tr, X_va, y_va, X_te, y_te):
        S, _ = greedy_forward_selection(X_tr, y_tr, X_va, y_va)
        fits = [SparsePolyLogReg(S, 10, C).fit(X_tr, y_tr) for C in (0.85, 10, 100, 1000, 10000)]
        return max(fits, key=lambda m: m.score(X_va, y_va)).score(X_te, y_te)

    for n in NS:
        X, y = load(f"kryptonite-{n}")
        for seed in SEEDS:
            D = split(X, y, seed)
            X_tr, y_tr, X_va, y_va, X_te, y_te = D
            fitters = {
                MODELS[0]: lambda: LogisticRegression(C=0.85, max_iter=2000).fit(X_tr, y_tr).score(X_te, y_te),
                MODELS[1]: lambda: poly_logreg(X_tr, y_tr, X_te, y_te, 2),
                MODELS[2]: lambda: mock_kernel_accuracies(X_tr, y_tr, X_te, y_te, [10], verbose=False)[10],
                MODELS[3]: lambda: knn(*D),
                MODELS[4]: lambda: SVC(C=10, gamma="scale").fit(X_tr, y_tr).score(X_te, y_te),
                MODELS[5]: lambda: RandomForestClassifier(300, n_jobs=2, random_state=seed).fit(X_tr, y_tr)
                                                                                        .score(X_te, y_te),
                MODELS[6]: lambda: HistGradientBoostingClassifier(max_iter=500, random_state=seed).fit(X_tr, y_tr)
                                                                                                  .score(X_te, y_te),
                MODELS[7]: lambda: mlp_fit(n, seed, *D),
                MODELS[8]: lambda: sparse_poly(*D),
            }
            with track(f"03_models_n{n}_s{seed}"):
                for m, fit in fitters.items():
                    t0 = time.time()
                    acc[str(n)][m].append(float(fit()))
                    sec[str(n)][m].append(time.time() - t0)
            print(f"n={n} seed={seed}: " + ", ".join(f"{m.split(' (')[0]} {v[-1]:.3f}"
                                                     for m, v in acc[str(n)].items()), flush=True)
    return {"accuracy": acc, "seconds": sec}


@app.cell(hide_code=True)
def _():
    run_grid = mo.ui.run_button(label="Recompute MLP grid (~13 min)")
    run_models = mo.ui.run_button(label="Recompute 9 models × 5 splits (~1.3 h)")
    mo.vstack([
        mo.md("""
        ## Experiments
        By default the results are read from `results_marimo/` (if re-run here) or from the committed `results/`.
        The model comparison retrains the seed-0 MLP instead of reusing the grid-search network (same seed, same
        configuration), so the MLP grid and the comparison can be re-run independently.
        """),
        mo.hstack([run_grid, run_models], justify="start"),
    ])
    return run_grid, run_models


@app.cell
def _(run_grid):
    with mo.status.spinner("MLP grid search on seed 0..."):
        mlp_config, grid_source = cached_or_run(run_grid, "mlp_config.json", compute_mlp_grid)
    mo.stop(mlp_config is None, mo.callout("No cached MLP configuration: press its button.", kind="warn"))
    return grid_source, mlp_config


@app.cell
def _(mlp_config, run_models):
    with mo.status.spinner("Nine learners on five splits for n = 20, 30, 40..."):
        comparison, comparison_source = cached_or_run(run_models, "model_comparison.json",
                                                      lambda: compute_models(mlp_config))
    mo.stop(comparison is None, mo.callout("No cached model comparison: press its button.", kind="warn"))
    return comparison, comparison_source


@app.cell(hide_code=True)
def _(comparison, comparison_source, grid_source, mlp_config):
    _rows = []
    for _m, _rep in zip(MODELS, REPRESENTATION):
        _row = {"model": _m, "representation": _rep}
        for _n in NS:
            _a = comparison["accuracy"][str(_n)][_m]
            _row[f"n={_n} (%)"] = pm(_a) + ("  ✔" if np.mean(_a) >= 0.9 else "")
        _s = comparison["seconds"]["40"][_m]
        _row["time n=40 (s)"] = f"{np.mean(_s[1:] if _m.startswith('MLP') and len(_s) > 1 else _s):.1f}"
        _rows.append(_row)
    _latex = "\n".join(
        f"{_m} & " + " & ".join(
            (lambda mu, h: (rf"\textbf{{{100 * mu:.1f} $\pm$ {100 * h:.1f}}}" if mu >= 0.9
                            else rf"{100 * mu:.1f} $\pm$ {100 * h:.1f}"))(*mean_ci(comparison["accuracy"][str(_n)][_m]))
            for _n in NS) + r" \\" for _m in MODELS)
    mo.vstack([
        mo.md(f"""
        ## Table 1
        MLP configuration ({grid_source}): `{mlp_config}`.  Accuracies ({comparison_source}): mean ± 95% CI over
        5 splits; ✔ marks the 90% target.  Time: seconds per fit at $n=40$ including hyper-parameter search.
        """),
        mo.ui.table(_rows, selection=None, pagination=False),
        mo.accordion({"LaTeX rows": mo.md(f"```latex\n{_latex}\n```")}),
    ])
    return


@app.cell
def _():
    n_view = mo.ui.dropdown(options={f"n = {n}": n for n in NS}, value="n = 40", label="Dataset")
    return (n_view,)


@app.cell(hide_code=True)
def _(comparison, n_view):
    _acc = comparison["accuracy"][str(n_view.value)]
    _fig, _ax = plt.subplots(figsize=(FULL, 2.6))
    for _i, _m in enumerate(MODELS):
        _v = 100 * np.array(_acc[_m])
        _mu, _h = mean_ci(_v)
        _ax.errorbar(_mu, _i, xerr=_h, fmt="D", color=COLORS[n_view.value] if _i >= 7 else COLORS["gray"],
                     capsize=2, ms=4)
        _ax.plot(_v, np.full(len(_v), _i) + 0.18, "o", ms=2.5, color="#999999")
    _ax.axvline(90, color="k", ls="--", lw=1)
    _ax.set_yticks(range(len(MODELS))); _ax.set_yticklabels([m.split(" (")[0] for m in MODELS]); _ax.invert_yaxis()
    _ax.set_xlabel(pct("Test accuracy (%)")); _ax.set_xlim(45, 100)
    mo.vstack([mo.md("### Per-split accuracies (grey dots) and mean ± 95% CI"), n_view, _fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Reading Table 1.** All learners with a fixed representation score between 50% and 59%: linear logistic
    regression is at chance because $W^{=1}[f]=0$; the quadratic and degree-10 polynomial models, $k$-NN and the
    RBF-kernel SVM sit near the degree-2 ceiling because the $n-10$ distractor bits dominate their features or
    distances; the tree ensembles get no marginal signal for their greedy splits.  The two learners that adapt
    their representation — the MLP on all raw inputs and sparse polynomial logistic regression — clear the target
    for every $n$: **H2 is supported**.
    """)
    return


if __name__ == "__main__":
    app.run()

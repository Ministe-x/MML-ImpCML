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
    from sklearn.linear_model import LogisticRegression

    from kmo import cached_or_run, save_json, save_figure, track
    from krypto.data import load, split, mock_split, NS, SEEDS, TARGET
    from krypto.polynomial import mock_kernel_accuracies, parity_features
    from krypto.selection import greedy_forward_selection
    from krypto.fourier import truth_table
    from krypto.stats import mean_ci
    from krypto.plotting import use_paper_style, FULL, COLORS, pct

    PLOT_BACKEND = use_paper_style()
    DEGREES = list(range(1, 11))
    C_GRID = [0.85, 10, 100, 1000, 10000]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 02 · Polynomial models: approximation is not the problem (Figure 2)

    The mock paper fits polynomial logistic regression up to degree 10, stays below 59%, and reads this as a failure of
    universal approximation.  Notebook 01 suggests a different explanation.  The label is a degree-6 polynomial
    threshold function of only 10 bits, so polynomials of degree $\ge 6$ can represent it; but on all $n$ inputs a
    degree-$d$ model has $\binom{n+d}{d}$ features and nowhere near enough data to estimate them.  We test this
    explanation (**H1**) by changing one thing at a time and otherwise using the mock paper's model exactly (normalised
    `PolynomialFeatures` kernel, logistic loss, $C=0.85$):

    | step | input | model / regulariser | split | what it tells us |
    |---|---|---|---|---|
    | 1. reproduction | all $n$ bits | mock paper's code, $C=0.85$ | mock paper's split | we get its numbers back |
    | 2. fewer inputs | **the 10 selected bits** | same model, $C=0.85$ | 5 stratified 60/20/20 splits | is the problem estimation? |
    | 3. regularisation | 10 selected bits | $C\in\{0.85,10,10^2,10^3,10^4\}$ chosen on validation | 5 splits | was $C=0.85$ holding it back? |
    | 4. theory | population | infinite data, (almost) no regularisation | — | the best each degree could do |
    """)
    return


@app.function
def compute_mock():
    """Step 1: the mock paper's normalised polynomial-kernel model on all n bits, degrees 1-10, its own split."""
    out = {}
    for n in NS:
        X, y = load(f"kryptonite-{n}")
        X_tr, y_tr, _, _, X_te, y_te = mock_split(X, y)
        with track(f"02_mock_kernel_n{n}"):
            acc = mock_kernel_accuracies(X_tr, y_tr, X_te, y_te, DEGREES)
        out[str(n)] = {str(d): a for d, a in acc.items()}
    return out


@app.function
def compute_sparse():
    """Steps 2-3: the same model on the coordinates selected from training/validation data, 5 splits."""
    out = {}
    for n in NS:
        X, y = load(f"kryptonite-{n}")
        r = {"S": [], "fixed": {str(d): [] for d in DEGREES}, "tuned": {str(d): [] for d in DEGREES},
             "C": {str(d): [] for d in DEGREES}}
        with track(f"02_sparse_poly_n{n}"):
            for seed in SEEDS:
                X_tr, y_tr, X_va, y_va, X_te, y_te = split(X, y, seed)
                S, _ = greedy_forward_selection(X_tr, y_tr, X_va, y_va)
                r["S"].append(sorted(int(v) for v in S))
                for d in DEGREES:
                    F_tr, F_va, F_te = (parity_features(Z, S, d) for Z in (X_tr, X_va, X_te))
                    fits = {C: LogisticRegression(C=C, fit_intercept=False, max_iter=5000, tol=1e-8).fit(F_tr, y_tr)
                            for C in C_GRID}
                    best_C = max(C_GRID, key=lambda C: fits[C].score(F_va, y_va))
                    r["fixed"][str(d)].append(fits[0.85].score(F_te, y_te))
                    r["tuned"][str(d)].append(fits[best_C].score(F_te, y_te))
                    r["C"][str(d)].append(best_C)
                print(f"n={n} seed={seed}: S={sorted(S)}", flush=True)
        out[str(n)] = r
    return out


@app.cell(hide_code=True)
def _():
    run_mock = mo.ui.run_button(label="Recompute step 1 (~11 min, ~4 GB RAM)")
    run_sparse = mo.ui.run_button(label="Recompute steps 2–3 (~28 min)")
    mo.vstack([
        mo.md("""
        ## Experiments
        Steps 1–3 are expensive.  By default their results are read from `results_marimo/` (if you re-ran them here)
        or from the committed `results/`; press a button to recompute them from scratch.
        """),
        mo.hstack([run_mock, run_sparse], justify="start"),
    ])
    return run_mock, run_sparse


@app.cell
def _(run_mock):
    with mo.status.spinner("Step 1: mock paper's kernel model, degrees 1-10, n = 20, 30, 40..."):
        mock, mock_source = cached_or_run(run_mock, "poly_mock_reproduction.json", compute_mock)
    mo.stop(mock is None, mo.callout("No cached result for step 1: press its button.", kind="warn"))
    return mock, mock_source


@app.cell
def _(run_sparse):
    with mo.status.spinner("Steps 2-3: selected bits, 5 splits, 10 degrees, 5 values of C..."):
        sparse, sparse_source = cached_or_run(run_sparse, "poly_sparse.json", compute_sparse)
    mo.stop(sparse is None, mo.callout("No cached result for steps 2-3: press their button.", kind="warn"))
    return sparse, sparse_source


@app.function
def compute_limit(S, n_ref=40):
    """Step 4: infinite-data limit of degree-d logistic regression.

    The irrelevant bits do not change which class is more likely, so we model the population as the uniform
    distribution over the 2^10 patterns of the relevant bits, labels flipped at rate eta: every pattern appears
    once with its majority label (weight 1 - eta) and once with the flipped label (weight eta).
    """
    X, y = load(f"kryptonite-{n_ref}")
    X_tr, y_tr, X_va, y_va, _, _ = split(X, y, 0)
    X_tv, y_tv = np.r_[X_tr, X_va], np.r_[y_tr, y_va]
    tt = truth_table(X_tv, y_tv, S)
    eta = float(np.mean((2 * y_tv - 1) != tt[(X_tv[:, S] < 0).astype(int) @ (1 << np.arange(10))]))
    patterns = np.where((np.arange(1024)[:, None] >> np.arange(10)) & 1, -1.0, 1.0)
    X_pop = np.r_[patterns, patterns]
    y_pop = np.r_[(tt > 0).astype(int), (tt < 0).astype(int)]
    w_pop = np.r_[np.full(1024, 1 - eta), np.full(1024, eta)]
    limit, clean = {}, {}
    for d in DEGREES:
        F = parity_features(X_pop, range(10), d)
        lr = LogisticRegression(C=1e6, fit_intercept=False, max_iter=20000, tol=1e-10).fit(F, y_pop,
                                                                                         sample_weight=w_pop)
        clean[str(d)] = float(((lr.decision_function(F[:1024]) > 0) == (tt > 0)).mean())
        limit[str(d)] = clean[str(d)] * (1 - 2 * eta) + eta
    return {"eta": eta, "limit": limit, "noise_free_agreement": clean}


@app.cell
def _(sparse):
    with mo.status.spinner("Step 4: population fits for degrees 1-10..."):
        limit = save_json(compute_limit(sparse["40"]["S"][0]), "poly_limit.json")
    return (limit,)


@app.cell(hide_code=True)
def _(limit):
    _eta = limit["eta"]
    mo.md(rf"""
    ## Infinite-data limit (step 4)
    A finite-sample result alone cannot tell us whether a degree is too low or the data too few; the infinite-data limit
    separates the two.  With estimated label-noise rate $\eta = {_eta:.4f}$, the population is the uniform distribution
    over the $2^{{10}}$ patterns of the relevant bits, each with its majority label kept with probability $1-\eta$.  For
    $d=2$ there is a closed form, because the only useful term of degree $\le 2$ is $c\,b_pb_q$ on the parity pair:
    $\tfrac12(1+\tfrac{{9}}{{64}})(1-2\eta)+\eta = {0.5 * (1 + 9 / 64) * (1 - 2 * _eta) + _eta:.4f}$ (Proposition 1).

    | degree $d$ | {" | ".join(str(d) for d in DEGREES)} |
    |---|{"---|" * len(DEGREES)}
    | population accuracy (%) | {" | ".join(f"{100 * limit['limit'][str(d)]:.1f}" for d in DEGREES)} |
    | noise-free agreement (%) | {" | ".join(f"{100 * limit['noise_free_agreement'][str(d)]:.1f}" for d in DEGREES)} |

    Degree 6 already reaches {100 * limit['limit']['6']:.1f}%, and from degree 7 the limit equals the Bayes accuracy
    $1-\eta$: polynomials of degree $\ge 6$ **can** represent the label function.
    """)
    return


@app.cell
def _():
    n_view = mo.ui.dropdown(options={f"n = {n}": n for n in NS}, value="n = 40", label="Dataset")
    return (n_view,)


@app.cell(hide_code=True)
def _(mock, mock_source, n_view, sparse, sparse_source):
    _n = str(n_view.value)
    _rows = []
    for _d in DEGREES:
        _f, _hf = mean_ci(sparse[_n]["fixed"][str(_d)])
        _t, _ht = mean_ci(sparse[_n]["tuned"][str(_d)])
        _rows.append({"degree": _d, "all bits, mock split (%)": f"{100 * mock[_n][str(_d)]:.1f}",
                      "10 bits, C = 0.85 (%)": f"{100 * _f:.1f} ± {100 * _hf:.1f}",
                      "10 bits, tuned C (%)": f"{100 * _t:.1f} ± {100 * _ht:.1f}",
                      "C chosen (5 splits)": str(sparse[_n]["C"][str(_d)])})
    mo.vstack([
        mo.md(f"## Results by degree\nStep 1 {mock_source}; steps 2–3 {sparse_source}."),
        n_view,
        mo.ui.table(_rows, selection=None, pagination=False),
        mo.md(f"Selected coordinates on the 5 splits: `{sparse[_n]['S']}`"),
    ])
    return


@app.function
def figure2(mock, sparse, limit):
    """Figure 2: accuracy against degree for the four settings, one panel per n."""
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.2), sharey=True)
    for ax, n in zip(axes, NS):
        k = str(n)
        ax.axhline(100 * TARGET, color="#bbbbbb", ls="--", lw=1)
        ax.plot(DEGREES, [100 * limit["limit"][str(d)] for d in DEGREES], color="k", ls=":", lw=1.2,
                label=r"$N\to\infty$ limit")
        ax.plot(DEGREES, [100 * mock[k][str(d)] for d in DEGREES], "o-", color=COLORS["gray"], mfc="white",
                label=r"all $n$ bits (mock)")
        for key, ls, mk, lab in (("fixed", "--", "s", r"10 selected bits, $C{=}0.85$"),
                                 ("tuned", "-", "D", r"10 selected bits, tuned $C$")):
            st = np.array([mean_ci(sparse[k][key][str(d)]) for d in DEGREES]) * 100
            ax.errorbar(DEGREES, st[:, 0], yerr=st[:, 1], color=COLORS[n], ls=ls, marker=mk, ms=3.5, capsize=2,
                        label=lab)
        ax.set_xlabel("Polynomial degree $d$")
        ax.set_xticks([1, 2, 4, 6, 8, 10]); ax.set_ylim(48, 100)
        ax.text(0.03, 0.97, f"$n={n}$", transform=ax.transAxes, va="top")
    axes[0].set_ylabel(pct("Test accuracy (%)"))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="outside upper center", handlelength=2.0, columnspacing=1.5)
    return save_figure(fig, "fig2_polynomial")


@app.cell(hide_code=True)
def _(limit, mock, sparse):
    mo.vstack([
        mo.md("## Figure 2"),
        figure2(mock, sparse, limit),
        mo.md(r"""
        **Reading Figure 2.** On all $n$ bits (grey) the mock paper's model stays near 55% at every degree, as the mock
        paper reports.  Restricting the *same* model to the 10 selected coordinates lifts it to 83–90% at degree 10, even
        with the mock paper's strong regulariser, and once $C$ is chosen on validation data it reaches the Bayes level
        (~96%) from degree 6 on, exactly the polynomial-threshold degree found in notebook 01.  The dotted infinite-data
        limit says the same from the theoretical side.  The mock paper's models therefore failed for lack of data relative
        to their $\binom{n+d}{d}$-dimensional feature space, not for lack of expressive power: **H1 is supported**.
        Whether this failure is overfitting or underfitting is examined in notebook 02b.
        """),
    ])
    return


if __name__ == "__main__":
    app.run()

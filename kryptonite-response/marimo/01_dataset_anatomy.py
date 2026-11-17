import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import itertools
    import sys
    from pathlib import Path

    import marimo as mo

    sys.path.insert(0, str(mo.notebook_dir() or Path(__file__).resolve().parent))
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from scipy.optimize import linprog

    from kmo import save_json, save_figure, track
    from krypto.data import load, split, bits
    from krypto.fourier import truth_table, spectrum_of_table, degree_weights, influences
    from krypto.selection import greedy_forward_selection, LookupTable
    from krypto.lora import load_base
    from krypto.mlp import logits
    from krypto.plotting import use_paper_style, FULL, COLORS

    PLOT_BACKEND = use_paper_style()
    NAMES = {20: "kryptonite-20", 30: "kryptonite-30", 40: "kryptonite-40", "adapt": "kryptonite-40-adapt"}
    KEYS = ["pre", 20, 30, 40, "adapt"]
    LABELS = {20: "Kryptonite-20", 30: "Kryptonite-30", 40: "Kryptonite-40", "pre": "KryptoNet-40 (Pre)",
              "adapt": "Kryptonite-40-Adapt"}
    CODE_BITS = (np.arange(1024)[:, None] >> np.arange(10)) & 1     # bit i of every 10-bit pattern code


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 01 · Anatomy of the Kryptonite datasets (Figure 1)

    The mock paper treats Kryptonite-$n$ as a black box and concludes from its failed models that the task is
    intrinsically hard.  Before comparing learners ourselves we want to know what the labels actually depend on: if
    there is a simple structure, it should explain which models fail and why, and it tells us what a successful model
    has to be able to represent.

    We go through five questions in order: what the inputs look like, which coordinates matter, what the label function
    on those coordinates is, whether the five tasks share it, and how it is built.  Everything is computed from the
    training and validation parts of split seed 0; the test parts are never used here, so the exploration cannot leak
    into later test scores.  The whole notebook runs in well under a minute.
    """)
    return


@app.function
def input_summary():
    """Input distribution and single-bit correlations of every labelled dataset."""
    rows = []
    for name in NAMES.values():
        X, y = load(name)
        s = 2 * y - 1
        rows.append({"dataset": name, "N": len(y), "n": X.shape[1],
                     "min |x|": round(float(np.abs(X).min()), 3),
                     "noise sd": round(float((np.abs(X) - 1).std()), 3),
                     "P(y=1)": round(float(y.mean()), 3),
                     "max_i |corr(b_i, y)|": round(float(np.abs(bits(X).T @ s / len(s)).max()), 3),
                     "1/sqrt(N)": round(float(1 / np.sqrt(len(y))), 3)})
    return rows


@app.cell(hide_code=True)
def _():
    mo.vstack([
        mo.md(r"""
        ## 1. The inputs are noisy bits
        Each feature is $x_i = b_i + \varepsilon_i$ with $b_i\in\{-1,+1\}$ and $\varepsilon_i\sim\mathcal N(0, 0.1^2)$, and
        no feature in any dataset comes closer to zero than 0.49.  Taking `sign(x)` therefore recovers every bit exactly,
        and from here on we can treat the data as Boolean.

        The table also tests the most natural first idea, that some single bit predicts the label.  None does: the largest
        correlation over the $n$ bits stays below $3/\sqrt N$, roughly what the largest of $n$ pure-noise correlations would
        be.  The label must depend on several bits jointly, which already rules out linear models.
        """),
        mo.ui.table(input_summary(), selection=None),
    ])
    return


@app.function
def select_coordinates():
    """Greedy forward selection on every task; Pre is the function computed by KryptoNet-40 itself."""
    data, sel, hist = {}, {}, {}
    with track("01_selection"):
        for key, name in NAMES.items():
            X, y = load(name)
            X_tr, y_tr, X_va, y_va, _, _ = split(X, y, seed=0, val=0.16 if key == "adapt" else 0.2)
            S, h = greedy_forward_selection(X_tr, y_tr, X_va, y_va)
            data[key], sel[key], hist[key] = (np.r_[X_tr, X_va], np.r_[y_tr, y_va]), sorted(S), h
        rng = np.random.default_rng(0)
        B = rng.choice([-1.0, 1.0], size=(200_000, 40))
        X_pre = (B + 0.1 * rng.standard_normal(B.shape)).astype(np.float32)
        y_pre = (logits(load_base(), X_pre) > 0).astype(int)
        S, h = greedy_forward_selection(X_pre[:150_000], y_pre[:150_000], X_pre[150_000:], y_pre[150_000:])
        data["pre"], sel["pre"], hist["pre"] = (X_pre, y_pre), sorted(S), h
    return data, sel, hist


@app.cell
def _():
    with mo.status.spinner("Greedy forward selection on five tasks (about 10 s)..."):
        data, sel, hist = select_coordinates()
    return data, hist, sel


@app.cell(hide_code=True)
def _(hist, sel):
    mo.vstack([
        mo.md(r"""
        ## 2. Which coordinates does the label depend on?
        Since no single bit is informative, we cannot rank coordinates one at a time.  Instead we use greedy forward
        selection (`krypto/selection.py`): start from the best *pair*, then keep adding the coordinate that lowers the
        validation log-loss of a lookup table on the selected bits, and stop when the loss no longer falls.  The lookup
        table assumes nothing about the form of the function, so the procedure only asks whether a coordinate carries
        information.  KryptoNet-40's pretraining data is not released, so for its task we label 200,000 synthetic inputs
        with the network itself and run the same selection.
        """),
        mo.ui.table([{"task": LABELS[k], "k": len(sel[k]), "selected coordinates": str(sel[k]),
                      "val log-loss": f"{hist[k][0]:.3f} → {hist[k][-1]:.3f}"} for k in KEYS], selection=None),
    ])
    return


@app.function
def analyse(data, sel):
    """Majority truth table, exact Fourier spectrum, degree weights, influences and label noise of each task."""
    tables, spectra, rows = {}, {}, []
    for key in KEYS:
        (X, y), S = data[key], sel[key]
        tt = truth_table(X, y, S)
        tables[key], spectra[key] = tt, spectrum_of_table(tt)
        W = degree_weights(spectra[key])
        noise = 1 - LookupTable(S).fit(X, y).score(X, y)
        rows.append({"task": LABELS[key], **{f"W={d}": round(float(W[d]), 3) for d in range(11)},
                     "influences": str(sorted(np.round(influences(spectra[key]), 3).tolist())),
                     "label noise": round(float(noise), 3)})
    return tables, spectra, rows


@app.cell
def _(data, sel):
    tables, spectra, spectrum_rows = analyse(data, sel)
    return spectra, spectrum_rows, tables


@app.cell(hide_code=True)
def _(spectra, spectrum_rows):
    mo.vstack([
        mo.md(r"""
        ## 3. The label function on the selected bits
        With 10 relevant bits there are only $2^{10}=1024$ input patterns, so we can estimate the function directly:
        `truth_table` takes the majority label of each pattern.  From it we compute the weight per degree
        $W^{=d}=\sum_{|S|=d}\hat f(S)^2$ and the influence $\mathrm{Inf}_i=\sum_{S\ni i}\hat f(S)^2$ of each variable; the
        share of labels that disagree with their pattern's majority estimates the label-noise rate.

        Two things matter for the rest of the project.  For Kryptonite-$n$ there is no weight at degrees 0–1 and only 2%
        at degree 2, which is why linear and quadratic models cannot work.  And the influences take only three values,
        which suggests the function is assembled from a few simple pieces.
        """),
        mo.ui.table(spectrum_rows, selection=None),
        mo.md(f"Distinct |Fourier coefficient| values × 64 (Kryptonite-40): "
              f"`{np.unique(np.round(np.abs(spectra[40]) * 64, 3)).tolist()}` — a hand-built function."),
    ])
    return


@app.cell
def _():
    task = mo.ui.dropdown(options={LABELS[k]: k for k in KEYS}, value=LABELS[40], label="Inspect task")
    return (task,)


@app.cell(hide_code=True)
def _(sel, spectra, task):
    _F = spectra[task.value]
    _fig, (_ax, _bx) = plt.subplots(1, 2, figsize=(FULL, 2.0))
    _ax.bar(range(11), degree_weights(_F), color=COLORS[40], width=0.7)
    _ax.set_xlabel("Fourier degree $d$"); _ax.set_ylabel("Weight $W^{=d}$"); _ax.set_xticks(range(0, 11, 2))
    _inf = influences(_F)
    _bx.bar([str(c) for c in sel[task.value]], _inf, color=COLORS["adapt"], width=0.7)
    _bx.set_xlabel("Selected coordinate"); _bx.set_ylabel("Influence"); _bx.set_ylim(0, 1.05)
    mo.vstack([task, _fig])
    return


@app.function
def influence_classes(F):
    """Variables ranked by influence: the top 2 (parity pair), the next 4 and the last 4 (gadget bits)."""
    order = list(np.argsort(-influences(F), kind="stable"))
    return [sorted(order[:2]), sorted(order[2:6]), sorted(order[6:])]


@app.function
def best_permutation(ref_tt, tt):
    """Best agreement between two truth tables over the 2!4!4! = 1152 class-respecting variable permutations."""
    c_ref, c = influence_classes(spectrum_of_table(ref_tt)), influence_classes(spectrum_of_table(tt))
    best = (0.0, None)
    for p0 in itertools.permutations(c[0]):
        for p1 in itertools.permutations(c[1]):
            for p2 in itertools.permutations(c[2]):
                perm = np.zeros(10, int)
                for cls_ref, cls in zip(c_ref, (p0, p1, p2)):
                    perm[list(cls_ref)] = cls
                agree = (ref_tt == tt[(CODE_BITS << perm).sum(1)]).mean()
                if agree > best[0]:
                    best = (agree, perm)
    return best


@app.function
def match_tasks(tables, sel):
    """Map every task onto KryptoNet-40's function; returns {task: (agreement, {pre coordinate: coordinate})}."""
    out = {}
    for key in [20, 30, 40, "adapt"]:
        agree, perm = best_permutation(tables["pre"], tables[key])
        out[key] = (float(agree), {sel["pre"][v]: sel[key][perm[v]] for v in range(10)})
    return out


@app.function
def swap_check(data):
    """KryptoNet-40 zero-shot on Adapt (train+val), before and after copying coordinate 19 into slot 0."""
    X, y = data["adapt"]
    X_sw = X.copy()
    X_sw[:, 0] = X[:, 19]
    net = load_base()
    return float(((logits(net, X) > 0) == y).mean()), float(((logits(net, X_sw) > 0) == y).mean())


@app.cell
def _(data, sel, tables):
    matches = match_tasks(tables, sel)
    zero_shot, zero_shot_swapped = swap_check(data)
    return matches, zero_shot, zero_shot_swapped


@app.cell(hide_code=True)
def _(matches, zero_shot, zero_shot_swapped):
    mo.vstack([
        mo.md(r"""
        ## 4. One Boolean function behind all five tasks
        All five tasks show the same influence profile: two variables of influence 1, four of 5/8 and four of 3/8.  If they
        share one function up to a relabelling of coordinates, the relabelling must map each influence class to itself,
        which leaves only $2!\,4!\,4! = 1152$ permutations.  We try all of them and keep the best match with
        KryptoNet-40's function.  The match is exact for Kryptonite-30 and -40 and nearly so for Kryptonite-20, whose truth
        table rests on fewer examples per pattern; Adapt, with only about three examples per pattern, agrees less, and its
        mapping shows what changed.
        """),
        mo.ui.table([{"task": LABELS[k], "agreement with KryptoNet-40": f"{a:.4f}",
                      "pre-coordinate → coordinate": str(m)} for k, (a, m) in matches.items()], selection=None),
        mo.callout(mo.md(
            f"**Kryptonite-40-Adapt is the pretraining task with coordinate 0 replaced by 19.**  KryptoNet-40 zero-shot "
            f"on Adapt: **{zero_shot:.4f}**; after moving coordinate 19 into slot 0: **{zero_shot_swapped:.4f}**."),
            kind="info"),
    ])
    return


@app.function
def gadget_structure(F, S):
    """Recover the parity pair and the two gadget blocks of f = chi_P h_A h_B from its Fourier coefficients."""
    pos = {v: i for i, v in enumerate(S)}
    pair = [S[i] for i in influence_classes(F)[0]]
    others = [v for v in S if v not in pair]

    def coef(vs):
        return F[sum(1 << pos[v] for v in pair + list(vs))]

    # fhat(P+S+T) fhat(P) = fhat(P+S) fhat(P+T) for S, T in different blocks: a violation links u and v
    parent = {v: v for v in others}

    def root(v):
        while parent[v] != v:
            v = parent[v]
        return v

    for u, v in itertools.combinations(others, 2):
        if abs(coef([u, v]) * coef([]) - coef([u]) * coef([v])) > 1e-9:
            parent[root(u)] = root(v)
    blocks = [sorted(v for v in others if root(v) == r) for r in sorted({root(v) for v in others})]
    magnitudes = {str(blk): sorted(set(np.round(np.abs([coef(T) for s in range(5)
                                                         for T in itertools.combinations(blk, s)]) * 64, 3).tolist()))
                  for blk in blocks}
    return pair, blocks, magnitudes


@app.function
def ptf_degree(tt):
    """Smallest d with f = sign(p) for a degree-d polynomial p: feasibility LP  f(b) p(b) >= 1 on all 1024 patterns."""
    B10 = np.where(CODE_BITS, -1.0, 1.0)
    for d in range(1, 11):
        cols = [np.ones(1024)] + [B10[:, list(T)].prod(1) for s in range(1, d + 1)
                                  for T in itertools.combinations(range(10), s)]
        A = np.stack(cols, 1)
        res = linprog(np.zeros(A.shape[1]), A_ub=-(tt[:, None] * A), b_ub=-np.ones(1024),
                      bounds=[(None, None)] * A.shape[1], method="highs")
        if res.status == 0:
            return d
    return None


@app.cell
def _(sel, spectra, tables):
    pair, blocks, block_magnitudes = gadget_structure(spectra["pre"], sel["pre"])
    PTF_DEGREE = ptf_degree(tables["pre"])
    return PTF_DEGREE, block_magnitudes, blocks, pair


@app.cell(hide_code=True)
def _(PTF_DEGREE, block_magnitudes, blocks, pair):
    mo.md(rf"""
    ## 5. How the function is built: a 2-bit parity times two 4-bit gadgets
    The influence profile suggests $f = \chi_P\,h_A\,h_B$: the parity of the two influence-1 bits times two functions on
    disjoint blocks of four bits.  If so, $\hat f(P\cup S\cup T)\,\hat f(P) = \hat f(P\cup S)\,\hat f(P\cup T)$ whenever
    $S, T$ lie in different blocks.  We use this in reverse: two bits that violate the identity must share a block, and
    joining all such pairs recovers the blocks.

    Which polynomial degree is really needed?  A degree-$d$ model can classify perfectly only if $f=\mathrm{{sign}}(p)$
    for a polynomial $p$ of degree $d$ (a polynomial threshold function, PTF).  Whether such a $p$ exists is a linear
    feasibility problem, so a linear program for $d=1,2,\dots$ finds the smallest such degree, which notebook 02 then
    puts to the test.

    | | KryptoNet-40 (Pre) coordinates |
    |---|---|
    | parity pair $P$ | `{pair}` |
    | gadget blocks | `{blocks}` |
    | block \|coefficients\| × 64 | `{block_magnitudes}` |
    | **PTF degree** | **{PTF_DEGREE}** (not {PTF_DEGREE - 1}) |
    """)
    return


@app.function
def figure1(spectra, sel, ptf):
    """Figure 1: coordinate roles of the five tasks (left) and Fourier weight by degree (right)."""
    roles = np.full((5, 40), np.nan)           # 0 irrelevant, 1 weak gadget, 2 strong gadget, 3 parity pair
    rows = [20, 30, 40, "pre", "adapt"]
    for r, key in enumerate(rows):
        roles[r, :40 if key in ("pre", "adapt") else key] = 0
        for role, members in zip((3, 2, 1), influence_classes(spectra[key])):
            for i in members:
                roles[r, sel[key][i]] = role
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(FULL, 1.8), gridspec_kw={"width_ratios": [2.35, 1]})
    cmap = ListedColormap(["#f2f2f2", "#9ecae1", "#3182bd", "#08306b"])
    cmap.set_bad("white")
    ax.imshow(np.ma.masked_invalid(roles), cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(5)); ax.set_yticklabels([LABELS[k] for k in rows])
    ax.set_xticks([0, 10, 19, 30, 39]); ax.set_xlabel("Input coordinate $i$")
    ax.set_xticks(np.arange(-0.5, 40, 1), minor=True); ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6); ax.tick_params(which="minor", length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.annotate("", xy=(19, 4), xytext=(0, 4), arrowprops=dict(arrowstyle="->", color="#CC79A7", lw=1.2))
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in (3, 2, 1, 0)]
    fig.legend(handles, ["parity pair", "gadget bit (influence 5/8)", "gadget bit (influence 3/8)", "irrelevant"],
               ncol=4, loc="outside upper center", handlelength=1.0, columnspacing=1.2, handletextpad=0.4)
    bx.bar(range(11), degree_weights(spectra[40]), color="#3182bd", width=0.7)
    for key, mk in [(20, "o"), (30, "s"), ("pre", "^"), ("adapt", "x")]:
        bx.plot(range(11), degree_weights(spectra[key]), mk, ms=3.5, mfc="none", color="k", lw=0)
    bx.axvline(ptf - 0.5, color=COLORS["gray"], ls=":", lw=1)
    bx.text(ptf - 0.8, 0.245, "PTF degree", va="top", ha="right")
    bx.set_xlabel("Fourier degree $d$"); bx.set_ylabel(r"Weight $W^{=d}[f]$")
    bx.set_xticks([0, 2, 4, 6, 8, 10]); bx.set_ylim(0, 0.25)
    return save_figure(fig, "fig1_anatomy")


@app.cell(hide_code=True)
def _(PTF_DEGREE, sel, spectra):
    mo.vstack([
        mo.md("## Figure 1"),
        figure1(spectra, sel, PTF_DEGREE),
        mo.md(r"""
        **Reading Figure 1.** On the left, every task reads exactly ten coordinates with the same pattern of roles (two
        parity bits and two blocks of four gadget bits); only the positions differ.  Adapt differs from KryptoNet-40's
        pretraining task in a single place, a parity bit moved from coordinate 0 to 19 (arrow).  On the right, the Fourier
        weight lies almost entirely at degree 4 and above, and the bars (Kryptonite-40) and markers (the other four tasks)
        coincide.  A successful model therefore has to find ten specific bits among $n$ and model degree-6 interactions
        between them; this combination, rather than the function itself, is what makes Kryptonite hard.
        """),
    ])
    return


@app.cell
def _(PTF_DEGREE, blocks, matches, pair, sel, spectra, tables):
    save_json({"selected": {str(k): v for k, v in sel.items()},
               "degree_weights": {str(k): degree_weights(v).tolist() for k, v in spectra.items()},
               "mapping_from_pre": {str(k): {str(a): int(b) for a, b in m.items()} for k, (_, m) in matches.items()},
               "pair": [int(v) for v in pair], "blocks": [[int(v) for v in b] for b in blocks],
               "ptf_degree": PTF_DEGREE, "truth_table_40": tables[40].tolist(), "selected_40": sel[40]},
              "anatomy.json")
    mo.md("Saved `results_marimo/anatomy.json` and `results_marimo/figures/fig1_anatomy.{pdf,png}`.")
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from collections import defaultdict
    from pathlib import Path

    import marimo as mo

    sys.path.insert(0, str(mo.notebook_dir() or Path(__file__).resolve().parent))
    import numpy as np
    import matplotlib.pyplot as plt

    from kmo import load_cached, runtime_log, save_json
    from krypto.energy import estimate
    from krypto.plotting import use_paper_style, COLUMN, COLORS

    PLOT_BACKEND = use_paper_style()
    GROUPS = {"01": "Dataset anatomy (nb 01)", "02": "Polynomial models (nb 02)", "03": "Model comparison (nb 03)",
              "04": "Learning curves (nb 04)", "05": "KryptoNet-40 and LoRA (nb 05)", "06": "Operation counts (nb 06)",
              "07": "Final models (nb 07)"}
    GB_GRID, GB_GRID_HI = 128.8, 175.9


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 08 · Environmental impact of our experiments (Table 4)

    Every experiment logged its wall-clock time (`runtime_log.json`; entries re-run in marimo override the Jupyter
    ones).  We have no access to power measurements on the virtual machine we used, so we convert time to energy with
    the Green Algorithms model (Lannelongue et al., 2021),
    $E = t\,(n_\text{cores}P_\text{core} + m\,P_\text{mem})\,\text{PUE}$, and energy to emissions with the carbon
    intensity of the electricity grid.

    The model needs assumptions, which we state rather than hide.  The defaults are those of the report: 2 cores at
    $P_\text{core}=12$ W, $m=4$ GB of memory at 0.3725 W/GB, PUE = 1 (we ignore data-centre overhead, which typically
    adds 10–60%), and 128.8 gCO$_2$/kWh (Great Britain average 2025), with 175.9 gCO$_2$e/kWh as an upper value.  The
    sliders show how much the totals depend on these choices; the comparison between models further down is the more
    informative part.
    """)
    return


@app.cell(hide_code=True)
def _():
    cores = mo.ui.slider(1, 16, value=2, label="cores", show_value=True)
    p_core = mo.ui.slider(2, 40, step=1, value=12, label="W per core", show_value=True)
    mem_gb = mo.ui.slider(1, 32, value=4, label="memory (GB)", show_value=True)
    p_mem = mo.ui.slider(0.1, 1.0, step=0.0025, value=0.3725, label="W per GB", show_value=True)
    pue = mo.ui.slider(1.0, 2.0, step=0.05, value=1.0, label="PUE", show_value=True)
    grid = mo.ui.slider(0, 800, step=0.1, value=GB_GRID, label="gCO2/kWh", show_value=True)
    mo.vstack([mo.hstack([cores, p_core, mem_gb], justify="start"),
               mo.hstack([p_mem, pue, grid], justify="start")])
    return cores, grid, mem_gb, p_core, p_mem, pue


@app.cell
def _(cores, grid, mem_gb, p_core, p_mem, pue):
    def footprint(seconds, intensity=None):
        """(kWh, gCO2e) of a computation under the current assumptions."""
        return estimate(seconds, cores=cores.value, p_core=p_core.value, mem_gb=mem_gb.value, p_mem=p_mem.value,
                        pue=pue.value, grid_g_per_kwh=grid.value if intensity is None else intensity)
    return (footprint,)


@app.cell
def _(footprint):
    log = runtime_log()
    _secs = defaultdict(float)
    for _e in log:
        _secs[_e["name"][:2]] += _e["seconds"]
    group_rows = []
    for _k, _name in GROUPS.items():
        _kwh, _g = footprint(_secs.get(_k, 0.0))
        _, _g_hi = footprint(_secs.get(_k, 0.0), GB_GRID_HI)
        group_rows.append((_name, _secs.get(_k, 0.0) / 3600, 1000 * _kwh, _g, _g_hi))
    total_seconds = sum(_secs.get(_k, 0.0) for _k in GROUPS)
    total_kwh, total_g = footprint(total_seconds)
    _, total_g_hi = footprint(total_seconds, GB_GRID_HI)
    return group_rows, log, total_g, total_g_hi, total_kwh, total_seconds


@app.cell(hide_code=True)
def _(grid, group_rows, total_g, total_g_hi, total_kwh, total_seconds):
    _fig, _ax = plt.subplots(figsize=(COLUMN, 2.0))
    _ax.barh([r[0].split(" (")[0] for r in group_rows], [r[2] for r in group_rows], color=COLORS[40])
    _ax.invert_yaxis(); _ax.set_xlabel("Energy (Wh)")
    mo.vstack([
        mo.md(f"## Table 4: time, energy and emissions by notebook  (grid: {grid.value:.1f} gCO₂/kWh)"),
        mo.ui.table([{"experiment": n, "time (h)": f"{t:.2f}", "energy (Wh)": f"{wh:.1f}", "gCO2e": f"{g:.1f}",
                      f"gCO2e at {GB_GRID_HI}": f"{gh:.1f}"} for n, t, wh, g, gh in group_rows]
                    + [{"experiment": "Total", "time (h)": f"{total_seconds / 3600:.2f}",
                        "energy (Wh)": f"{1000 * total_kwh:.1f}", "gCO2e": f"{total_g:.1f}",
                        f"gCO2e at {GB_GRID_HI}": f"{total_g_hi:.1f}"}], selection=None, pagination=False),
        _fig,
        mo.md(f"All experiments: **{total_seconds / 3600:.2f} h, {1000 * total_kwh:.1f} Wh, {total_g:.1f} g CO₂e** — "
              f"about the energy of boiling a kettle once or twice."),
    ])
    return


@app.cell
def _(footprint, log):
    _mc, _ = load_cached("model_comparison.json")
    _res, _ = load_cached("kryptonet_methods.json")
    fit_rows = []
    if _mc:
        for _model, _ts in _mc["seconds"]["40"].items():
            _ts = _ts[1:] if _model.startswith("MLP") and len(_ts) > 1 else _ts   # seed-0 MLP may come from the grid
            _t = float(np.mean(_ts))
            _kwh, _g = footprint(_t)
            fit_rows.append((_model, _t, 1000 * _kwh, _g, 100 * np.mean(_mc["accuracy"]["40"][_model])))
    _ad = defaultdict(list)
    for _e in log:
        if _e["name"].startswith("05_") and "_s" in _e["name"]:
            _ad[_e["name"][3:].rsplit("_s", 1)[0]].append(_e["seconds"])
    adapt_rows = []
    if _res:
        for _m in ("LoRA, W1, r=1", "LoRA, W1-W4, r=4", "Full fine-tuning", "Scratch (same architecture)"):
            if _ad[_m]:
                _t = float(np.mean(_ad[_m]))
                _kwh, _g = footprint(_t)
                adapt_rows.append((_m, _t, 1000 * _kwh, _g, 100 * np.mean(_res["accuracy"][_m])))
    return adapt_rows, fit_rows


@app.cell(hide_code=True)
def _(adapt_rows, fit_rows):
    def _table(rows):
        return mo.ui.table([{"model": m, "time (s)": f"{t:.1f}", "energy (Wh)": f"{wh:.3f}", "gCO2e": f"{g:.3f}",
                             "accuracy (%)": f"{a:.1f}"} for m, t, wh, g, a in sorted(rows, key=lambda r: r[1])],
                           selection=None, pagination=False)

    mo.vstack([
        mo.md("""
        ## Cost of a single fit at n = 40 (notebook 03)
        Model choice matters more than the totals: the mock paper's degree-10 kernel model costs about as much as
        our sparse polynomial model per fit but is over 40 points less accurate.
        """),
        _table(fit_rows),
        mo.md("""
        ## Adaptation of KryptoNet-40 (notebook 05; 3 learning rates per fit, 2,560 training examples)
        Adaptation reuses the energy already spent on pretraining; rank-1 LoRA trains in seconds and stores 552
        numbers per task.
        """),
        _table(adapt_rows),
    ])
    return


@app.cell
def _(cores, fit_rows, grid, group_rows, mem_gb, p_core, p_mem, pue, total_g, total_g_hi, total_kwh, total_seconds):
    save_json({"assumptions": {"cores": cores.value, "p_core": p_core.value, "mem_gb": mem_gb.value,
                               "p_mem": p_mem.value, "pue": pue.value, "grid_g_per_kwh": grid.value},
               "groups": group_rows, "total_hours": total_seconds / 3600, "total_Wh": 1000 * total_kwh,
               "total_g": total_g, "total_g_hi": total_g_hi, "fits_n40": fit_rows}, "sustainability.json")
    return


if __name__ == "__main__":
    app.run()

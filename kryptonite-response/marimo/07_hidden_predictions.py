import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import subprocess
    import sys
    from pathlib import Path

    import marimo as mo

    sys.path.insert(0, str(mo.notebook_dir() or Path(__file__).resolve().parent))
    import numpy as np

    from kmo import ROOT, OUT, HIDDEN, cached_or_run, load_cached, track
    from krypto.data import load, load_hidden, holdout, NS
    from krypto.polynomial import SparsePolyLogReg
    from krypto.selection import greedy_forward_selection
    from krypto.mlp import mlp, train, logits, set_seed
    from krypto.lora import load_base, add_lora, swap_oracle
    from krypto.stats import mean_ci

    FILES = {"20": "y_predicted_20.npy", "30": "y_predicted_30.npy", "40": "y_predicted_40.npy",
             "40_adapt": "y_predicted_40_adapt.npy"}


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 07 · Final models and hidden-set predictions

    For each dataset we submit the predictions of one model, chosen on the evidence of notebooks 02–05:

    * **Kryptonite-20/30/40:** the sparse polynomial model (greedy selection, then a degree-10 polynomial on the selected
      bits with $C$ chosen on held-out data), the most accurate model in Table 1 and cheap to fit.  Selection and $C$ are
      chosen on an 80/20 split of all labelled data; once these choices are made, the final model is refitted on all
      labelled data, since more data can only help it.
    * **Kryptonite-40-Adapt:** KryptoNet-40 with rank-1 LoRA on $W_1$, the cheapest method that reached the Bayes level in
      notebook 05.  We train it on five different 80/20 splits of the 4,000 labelled examples (the 20% for early
      stopping) and submit the majority vote of the five, which evens out the split-to-split variation seen in Table 3.

    Without hidden labels we cannot score these predictions, so we run sanity checks instead: agreement with an
    independently trained MLP, agreement with the analytic rank-1 swap, the predicted class balance (which should be
    close to 50%), and `check_submission.py` for the file format.

    This notebook writes its predictions to **`results_marimo/hiddenlabels/`**; the submitted files in `hiddenlabels/`
    are never overwritten.  Copy them over yourself if you want to submit these.
    """)
    return


@app.function
def compute_predictions(mlp_config, lora_lr):
    """Fit the final models, write results_marimo/hiddenlabels/*.npy and return a summary."""
    HIDDEN.mkdir(parents=True, exist_ok=True)
    expected, _ = load_cached("model_comparison.json")
    summary = {}
    for n in NS:
        X, y = load(f"kryptonite-{n}")
        H = load_hidden(f"kryptonite-{n}")
        with track(f"07_final_n{n}"):
            X_fit, y_fit, X_b, y_b = holdout(X, y, seed=0)                # 80% fit / 20% model selection
            S, _ = greedy_forward_selection(X_fit, y_fit, X_b, y_b)
            C = max((0.85, 10, 100, 1000, 10000),
                    key=lambda C: SparsePolyLogReg(S, 10, C).fit(X_fit, y_fit).score(X_b, y_b))
            final = SparsePolyLogReg(S, 10, C).fit(X, y)                 # refit on all labelled data
            pred = final.predict(H).astype(np.int64)
            set_seed(0)
            net, _ = train(mlp(n, 512, **mlp_config[str(n)]), X_fit, y_fit, X_b, y_b, seed=0)
            agree_mlp = float(((logits(net, H) > 0).astype(int) == pred).mean())
        np.save(HIDDEN / FILES[str(n)], pred)
        mu, h = (mean_ci(expected["accuracy"][str(n)]["Sparse polynomial LR (ours)"]) if expected
                 else (float("nan"), float("nan")))
        summary[str(n)] = {"S": sorted(int(v) for v in S), "C": C, "train_acc": final.score(X, y),
                           "agreement_with_MLP": agree_mlp, "positive_rate": float(pred.mean()),
                           "expected_accuracy": [mu, h]}
        print(f"n={n}: S={sorted(S)}, C={C}, agreement with MLP {agree_mlp:.4f}", flush=True)

    X, y = load("kryptonite-40-adapt")
    H = load_hidden("kryptonite-40-adapt")
    base = load_base()
    votes, val_accs = [], []
    with track("07_final_adapt"):
        for seed in range(5):
            X_tr, y_tr, X_va, y_va = holdout(X, y, seed)                 # 80% train, 20% for early stopping
            set_seed(seed)
            net, info = train(add_lora(base, 1, (0,)), X_tr, y_tr, X_va, y_va, lr=lora_lr, batch_size=64, seed=seed)
            votes.append((logits(net, H) > 0).astype(int))
            val_accs.append(info["val_acc"])
    pred = (np.mean(votes, 0) > 0.5).astype(np.int64)
    oracle = (logits(swap_oracle(base, 0, 19), H) > 0).astype(int)
    np.save(HIDDEN / FILES["40_adapt"], pred)
    summary["40_adapt"] = {"val_accs": val_accs, "agreement_with_rank1_swap": float((pred == oracle).mean()),
                           "pairwise_agreement_of_members": float(np.mean([(a == b).mean() for i, a in enumerate(votes)
                                                                          for b in votes[i + 1:]])),
                           "positive_rate": float(pred.mean())}
    return summary


@app.cell(hide_code=True)
def _():
    run_final = mo.ui.run_button(label="Fit final models and predict (~7 min)")
    mo.vstack([mo.md("## Final models"), run_final])
    return (run_final,)


@app.cell
def _(run_final):
    mlp_config, _ = load_cached("mlp_config.json")
    _methods, _ = load_cached("kryptonet_methods.json")
    mo.stop(mlp_config is None or _methods is None,
            mo.callout("Run notebooks 03 and 05 first (mlp_config.json, kryptonet_methods.json).", kind="warn"))
    _lrs = _methods["lr"]["LoRA, W1, r=1"]
    lora_lr = max(set(_lrs), key=_lrs.count)                 # learning rate most often chosen in notebook 05
    with mo.status.spinner("Fitting final models and predicting the hidden sets..."):
        summary, summary_source = cached_or_run(run_final, "hidden_predictions_summary.json",
                                                lambda: compute_predictions(mlp_config, lora_lr))
    mo.stop(summary is None, mo.callout("No cached summary: press the button.", kind="warn"))
    return lora_lr, summary, summary_source


@app.cell(hide_code=True)
def _(lora_lr, summary, summary_source):
    _rows = []
    for _n in NS:
        _s = summary[str(_n)]
        _mu, _h = _s["expected_accuracy"]
        _rows.append({"dataset": f"Kryptonite-{_n}", "model": "sparse polynomial LR",
                      "details": f"S = {_s['S']}, C = {_s['C']}",
                      "expected accuracy (5 splits)": f"{100 * _mu:.1f} ± {100 * _h:.1f}%",
                      "cross-check": f"{100 * _s['agreement_with_MLP']:.2f}% agreement with an MLP",
                      "hidden P(y=1)": f"{_s['positive_rate']:.3f}"})
    _a = summary["40_adapt"]
    _rows.append({"dataset": "Kryptonite-40-Adapt", "model": "5 × LoRA (W1, r=1), majority vote",
                  "details": f"lr {lora_lr}; member validation accuracies {np.round(_a['val_accs'], 4).tolist()}",
                  "expected accuracy (5 splits)": "see notebook 05",
                  "cross-check": f"{100 * _a['agreement_with_rank1_swap']:.2f}% agreement with the analytic swap",
                  "hidden P(y=1)": f"{_a['positive_rate']:.3f}"})
    mo.vstack([mo.md(f"Summary {summary_source}."), mo.ui.table(_rows, selection=None, pagination=False)])
    return


@app.cell(hide_code=True)
def _(summary):
    _folder = OUT if HIDDEN.exists() else ROOT
    _check = subprocess.run([sys.executable, str(ROOT / "check_submission.py"), str(_folder)],
                            capture_output=True, text=True)
    _rows = []
    if HIDDEN.exists():
        for _key, _file in FILES.items():
            _new, _old = HIDDEN / _file, ROOT / "hiddenlabels" / _file
            if _new.exists() and _old.exists():
                _rows.append({"file": _file, "agreement with submitted hiddenlabels/":
                              f"{100 * (np.load(_new) == np.load(_old)).mean():.2f}%"})
    mo.vstack([
        mo.md(f"## Format check: `check_submission.py {_folder.relative_to(ROOT.parent)}`"),
        mo.md(f"```\n{_check.stdout}\n```"),
        mo.ui.table(_rows, selection=None) if _rows else mo.md(
            "Predictions shown above come from the cache; press the button to write `results_marimo/hiddenlabels/` "
            "and compare it with the submitted files."),
    ])
    return


if __name__ == "__main__":
    app.run()

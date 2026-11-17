# marimo versions of the notebooks

[marimo](https://marimo.io) re-implementations of `notebooks/01`–`08` (including `02b`). They reproduce the same figures, tables and JSON results, reuse the `krypto/` package, and add a few interactive controls. Each notebook is a plain Python file, so it diffs cleanly in git and can also be run as a script.

## Running

```bash
pip install "marimo>=0.25"            # in addition to ../requirements.txt
cd kryptonite-response/marimo

marimo edit 01_dataset_anatomy.py     # interactive editor
marimo run 05_kryptonet_lora.py       # read-only app (code hidden)
python 06_lora_operation_count.py     # run as a script (cached results, no UI)
```

## How they differ from the Jupyter notebooks

- **Nothing in the original outputs is overwritten.** The marimo notebooks write JSON results, the runtime log, figures and hidden-set predictions to `../results_marimo/`, never to `results/`, `figures/` or `hiddenlabels/`.
- **Expensive experiments run only on demand.** Every experiment that takes minutes or hours has a *Recompute* button. Until you press it, the notebook reads that experiment's result from `results_marimo/` (if you re-ran it here) or else from the committed `results/`. This means every notebook opens in seconds. Cheap analyses (notebook 01, the infinite-data limit in 02, the mechanism in 05, all of 06 and 08) always run live.
- **Reactive.** Changing a control re-runs only the cells that depend on it. Recomputing an upstream result (e.g. the MLP grid in 03) automatically refreshes every table and figure built from it.

| Notebook | Always live | Behind a button (≈ time on 2 CPU cores) | Interactive additions |
|---|---|---|---|
| `01_dataset_anatomy` | everything (< 1 min) | — | inspect the degree weights and influences of any task |
| `02_polynomial_models` | infinite-data limit | mock kernel model (~11 min, ~4 GB); selected bits × 5 splits (~28 min) | per-degree table for any n |
| `02b_overfitting` | tables and Figure 5 from cached results | $C$ sweep (~4 min); train/test against degree (~6 min, ~4 GB) | train/test table for any n |
| `03_model_comparison` | Table 1 | MLP grid (~13 min); 9 models × 5 splits (~1.3 h) | per-split accuracies for any n |
| `04_learning_curves` | thresholds, Figure 3 | learning curves (~1.4 h) | every point of the curves for any n |
| `05_kryptonet_lora` | mechanism (swap, probes, column norms) | 13 methods × 5 splits (~25 min); sample efficiency (~28 min) | **LoRA lab**: train LoRA with any rank / layers / learning rate / split and see which input coordinates ΔW₁ acts on |
| `06_lora_operation_count` | everything (seconds) | — | **operation-count explorer**: layer, batch size and rank sliders, measured vs formula, break-even rank |
| `07_hidden_predictions` | summary, format check | final models + predictions (~7 min) → `results_marimo/hiddenlabels/` | agreement with the submitted `hiddenlabels/` |
| `08_sustainability` | everything | — | sliders for cores, power, memory, PUE and grid carbon intensity |

Dependencies between notebooks are the same as in Jupyter: 03 → 04, 07 (MLP configuration); 02 → 02b (selected sets, chosen $C$, limit), 04 (quadratic limit); 05 → 07 (LoRA learning rate). With the committed `results/` every notebook can be opened in any order.

## Implementation notes

- `kmo.py` holds the marimo-specific helpers: output folder, cache lookup (`cached_or_run`), runtime logging (`track`) and figure saving. Everything else comes from `../krypto/`.
- Each experiment is an `@app.function`, so it can be imported and tested outside marimo, e.g. on reduced settings.
- 03 retrains the seed-0 MLP in the model comparison instead of reusing the grid-search network (same seed and configuration). This lets the grid and the comparison be re-run independently.
- 05 additionally stores the rank of the analytic edit and the median column norm of the ignored coordinates in `kryptonet_mechanism.json`.
- Verified against the committed results: notebooks 01, 02 (limit), 04 (thresholds), 06 (operation counts) and 08 reproduce them exactly. The bit-probe accuracies in 05 differ only in the 4th decimal, and step timings depend on the machine.

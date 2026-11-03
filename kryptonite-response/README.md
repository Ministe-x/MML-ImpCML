# Ten Hidden Bits and a Rank-One Fix — a response to Kryptonite-3.0

Coursework for Mathematics for Machine Learning (60037/70015), Imperial College London (ImpCML 2026).
The report is `report/main.pdf`; this repository contains the code that reproduces every figure and table in it,
the executed notebooks, and our hidden-set predictions in `hiddenlabels/`.

## Main findings

* Every Kryptonite task (n = 20, 30, 40, KryptoNet-40's pretraining task and Kryptonite-40-Adapt) is the **same
  Boolean function of only 10 of the n input bits** — a 2-bit parity XOR two 4-bit "gadgets" — observed with ≈3.9%
  label noise, so the best achievable accuracy is ≈96%. Inputs are bits plus N(0, 0.1²) noise.
* The function has no Fourier weight below degree 2 and little below degree 4, which explains why the mock paper's
  polynomial/kernel models stay near 55%. It is a degree-6 polynomial threshold function, so polynomials *can*
  represent it: the mock paper's own model on 10 coordinates selected from training data reaches the Bayes level,
  and an MLP trained from scratch on all inputs clears 90% for every n.
* Quadratic logistic regression is provably capped at 56.5% (Proposition 1), so the mock paper's "more data" experiment
  measured approximation error. Capable models improve steeply with data.
* Kryptonite-40-Adapt is KryptoNet-40's pretraining task with input coordinate 0 moved to 19. A rank-1 update of the
  first layer fixes it: LoRA with r = 1 on W1 (552 trainable parameters) reaches ≈96%, while zero-shot use and probes
  are at chance (Proposition 2).

## Layout

| Path | Contents |
|---|---|
| `krypto/` | Shared code: data splits, Fourier analysis, feature selection, polynomial models (incl. the mock paper's kernel code), MLP training, LoRA (from scratch) and operation counts, statistics, runtime logging, plot style |
| `notebooks/01_dataset_anatomy.ipynb` | Figure 1 — input distribution, feature selection, Fourier spectrum, the shared Boolean function |
| `notebooks/02_polynomial_models.ipynb` | Figure 2 — mock paper's kernel model reproduced; same model on selected bits; infinite-data limit |
| `notebooks/03_model_comparison.ipynb` | Table 1 — nine learners on five shared splits (also selects the MLP configuration) |
| `notebooks/04_learning_curves.ipynb` | Figure 3 — learning curves from 250 points to the full training split |
| `notebooks/05_kryptonet_lora.ipynb` | Figure 4 and Table 3 — KryptoNet-40: mechanism, probes, full fine-tuning, LoRA ranks, sample efficiency |
| `notebooks/06_lora_operation_count.ipynb` | Table 2 — LoRA operation counts verified with PyTorch's `FlopCounterMode` |
| `notebooks/07_hidden_predictions.ipynb` | Final models and `hiddenlabels/*.npy` (runs `check_submission.py`) |
| `notebooks/08_sustainability.ipynb` | Table 4 — wall-clock time, energy and CO2e of all experiments |
| `results/` | JSON outputs of the notebooks (every number in the report is generated from these) |
| `figures/` | Figures as PDF (used in the report) and PNG |
| `hiddenlabels/` | `y_predicted_20.npy`, `y_predicted_30.npy`, `y_predicted_40.npy`, `y_predicted_40_adapt.npy` |
| `report/` | LaTeX source (ImpCML template), `make_numbers.py` (writes `report/generated/` from `results/` and copies the figures into `report/figures/`, so the folder compiles on its own, e.g. on Overleaf), `main.pdf` |
| `Datasets/`, `kryptonet.py`, `kryptonet40.pt`, `check_submission.py` | Provided with the coursework, unchanged |

## Reproducing

```bash
pip install -r requirements.txt
cd notebooks
# order matters: 03 chooses the MLP configuration used by 04 and 07; 05 chooses the LoRA learning rate used by 07
for nb in 01_dataset_anatomy 03_model_comparison 04_learning_curves 02_polynomial_models \
          05_kryptonet_lora 06_lora_operation_count 07_hidden_predictions 08_sustainability; do
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 $nb.ipynb
done
cd .. && python check_submission.py .
python report/make_numbers.py && cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Everything runs on a CPU. On two cores the notebooks take about 5 hours in total, dominated by MLP training
(notebooks 03 and 04) and the mock paper's degree-10 kernel model at n = 40 (notebooks 02 and 03, ~4 GB of memory).
Figures are rendered with LaTeX (Type-1 fonts) when `latex` and the `cm-super` fonts are installed, and with
Matplotlib's own fonts otherwise.

## Experimental protocol (summary)

* Kryptonite-n: the same five stratified 60/20/20 train/validation/test splits (seeds 0–4) for every model.
  Feature selection, hyper-parameters, regularisation and early stopping use only training and validation data.
* Kryptonite-40-Adapt: five stratified 64/16/20 splits (800 test points, as in the mock paper).
* Results are reported as mean ± 95% Student-t confidence interval over the five splits.
* Final hidden-set models are refitted on all labelled data (selection/early stopping on an internal 20% hold-out).

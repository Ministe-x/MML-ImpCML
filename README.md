# ImpCML 2026 · A Response to Kryptonite-3.0

Group coursework for Mathematics for Machine Learning (60037/70015), Imperial College London. We read the mock paper *Kryptonite-3.0: The Final End to Machine Learning Hype?*, refute its three claims with valid experimental design, and adapt KryptoNet-40 to Kryptonite-40-Adapt with LoRA.

- Report: [`kryptonite-response/report/main.pdf`](kryptonite-response/report/main.pdf) (copy: [`Kryptonite_response_report.pdf`](Kryptonite_response_report.pdf))
- Hidden-set predictions: [`kryptonite-response/hiddenlabels/`](kryptonite-response/hiddenlabels/) (zipped: [`hiddenlabels.zip`](hiddenlabels.zip))
- Technical README of the code: [`kryptonite-response/README.md`](kryptonite-response/README.md)

**Main finding.** Every Kryptonite task is **the same Boolean function of only 10 input bits** (a 2-bit parity times two 4-bit gadgets, 3.9% label noise, Bayes accuracy ≈ 96%). The mock paper's three failures come from estimation error, approximation error and the wrong choice of adaptation method, not from a limit of machine learning. Expected accuracy on all four hidden sets is 95–96% (target: 90%).

---

## 1. Environment setup

CPU only. Running every notebook takes about 5 hours on two cores; the degree-10 kernel model at n = 40 needs about 4 GB of memory.

### 1.1 Python

Python 3.12 or 3.13 (the report's results were produced with 3.13; 3.12.9 is also tested).

```bash
# Option A: conda
conda create -n kryptonite python=3.12 -y
conda activate kryptonite

# Option B: venv
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install the dependencies
pip install -r kryptonite-response/requirements.txt
```

Dependencies in [`requirements.txt`](kryptonite-response/requirements.txt) (the comments there give the exact versions used for the report):

| Package | Minimum | Used for |
|---|---|---|
| numpy | ≥ 2.0 | arrays |
| scipy | ≥ 1.13 | linear algebra (Cholesky), linear programming (`linprog`), t-distribution |
| scikit-learn | ≥ 1.5 | logistic regression, k-NN, SVM, tree ensembles, data splits |
| torch | ≥ 2.3 | MLP, KryptoNet-40, LoRA, `FlopCounterMode` (CPU build is enough) |
| matplotlib | ≥ 3.8 | figures |
| jupyter | — | running and re-executing the notebooks |

Check the installation:

```bash
python -c "import numpy, scipy, sklearn, torch, matplotlib; print(numpy.__version__, scipy.__version__, sklearn.__version__, torch.__version__, matplotlib.__version__)"
```

### 1.2 LaTeX (optional, recommended)

- **Compiling the report** needs `pdflatex` and `bibtex` (TeX Live, MacTeX or MiKTeX). Alternatively, upload `kryptonite-response/report/` to Overleaf; it compiles on its own.
- **Figures.** If `latex` and the `cm-super` fonts are installed, Matplotlib renders text with LaTeX, which gives the **Type-1 fonts** required by the submission guidelines. Otherwise it falls back to Matplotlib's own fonts: the figures are still produced, but not with Type-1 fonts. See [`krypto/plotting.py`](kryptonite-response/krypto/plotting.py).

```bash
# Ubuntu / Debian
sudo apt install texlive-latex-extra texlive-fonts-recommended cm-super dvipng
# macOS: install MacTeX (includes cm-super)
```

---

## 2. Running the code

Run every notebook from inside `kryptonite-response/notebooks/` (they locate the `krypto/` package through `Path.cwd().parent`).

### 2.1 Full reproduction

**The order matters**: 03 chooses the MLP configuration used by 04 and 07; 02 computes the quadratic model's infinite-data limit used by 04; 05 chooses the LoRA learning rate used by 07.

```bash
cd kryptonite-response/notebooks
for nb in 01_dataset_anatomy 03_model_comparison 04_learning_curves 02_polynomial_models \
          05_kryptonet_lora 06_lora_operation_count 07_hidden_predictions 08_sustainability; do
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 $nb.ipynb
done
cd ..

python check_submission.py .                      # check the format of hiddenlabels/
python report/make_numbers.py                     # results/*.json -> report/generated/*.tex, copy figures into report/figures/
cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

### 2.2 Running only part of it

The JSON outputs of every notebook are committed in `results/`, so you can re-run only what you need:

| Goal | Run | Approx. time (2 cores) |
|---|---|---|
| Understand the data structure | 01 | < 1 min |
| Verify the LoRA operation counts | 06 | < 1 min |
| Regenerate the hidden-set predictions | 07 (uses the existing `mlp_config.json` and `kryptonet_methods.json`) | ~7 min |
| Recompile after editing the report text | `make_numbers.py` + LaTeX | seconds |
| Polynomial experiments | 02 | ~40 min |
| KryptoNet-40 / LoRA | 05 | ~50 min |
| Model comparison, learning curves | 03, 04 | ~1.2–1.4 h each |

---

## 3. Repository structure

```
kryptonite-coursework/
├── README.md                          this file
├── Kryptonite_response_report.pdf     report PDF (copy)
├── hiddenlabels.zip                   the four prediction files, zipped for submission
└── kryptonite-response/
    ├── README.md                      technical README
    ├── requirements.txt               Python dependencies
    ├── check_submission.py            [provided] checks the prediction file format
    ├── kryptonet.py                   [provided] KryptoNet-40 architecture and loader
    ├── kryptonet40.pt                 [provided] KryptoNet-40 pretrained weights
    ├── Datasets/                      [provided] labelled and hidden sets (*.npy)
    ├── krypto/                        shared code package (see §4.1)
    ├── notebooks/                     01–08, each reproduces one figure or table (see §4.2)
    ├── results/                       JSON outputs of the notebooks; every number in the report comes from here
    ├── figures/                       figures (PDF for the report, PNG for preview)
    ├── hiddenlabels/                  y_predicted_{20,30,40,40_adapt}.npy
    └── report/
        ├── main.tex                   report source (ImpCML / ICML 2021 template)
        ├── references.bib             bibliography
        ├── make_numbers.py            generates generated/ and copies figures/
        ├── generated/                 auto-generated number macros and table bodies (do not edit by hand)
        ├── figures/                   figures used by the report (copied by make_numbers.py)
        ├── icml2021.sty / .bst, ...   template files
        └── main.pdf                   compiled report
```

### Data flow

```
Datasets/*.npy ──▶ notebooks/0X_*.ipynb ──▶ results/*.json ──▶ report/make_numbers.py ──▶ report/generated/*.tex ──▶ report/main.tex
                          │                                                                        ▲
                          └──▶ figures/*.pdf ──────────── (copied to report/figures/) ─────────────┘
```

No experimental number is typed into the report by hand: every number is a `\newcommand` macro (e.g. `\SparseFortyMean`) generated from the JSON files by `make_numbers.py`. After re-running an experiment, the text, tables and figures therefore stay consistent automatically.

---

## 4. What the code does

### 4.1 The `krypto/` package

| Module | Main contents |
|---|---|
| [`data.py`](kryptonite-response/krypto/data.py) | `load` / `load_hidden` read the data; `split(X, y, seed)` is the stratified 60/20/20 split shared by all experiments (seeds 0–4); `mock_split` reproduces the mock paper's split; `holdout` is a two-way split for the final models; `bits` rounds features to ±1; `codes` encodes the bit pattern on selected coordinates as an integer |
| [`fourier.py`](kryptonite-response/krypto/fourier.py) | fast Walsh–Hadamard transform; `truth_table` (majority-vote truth table); `spectrum_of_table` (exact Fourier coefficients); `degree_weights` (weight $W^{=d}$ per degree); `influences` (influence of each variable) |
| [`selection.py`](kryptonite-response/krypto/selection.py) | `LookupTable`: estimates P(y=1) for every bit pattern (Laplace smoothing); `greedy_forward_selection`: starts from the best coordinate pair and adds coordinates greedily by validation log-loss |
| [`polynomial.py`](kryptonite-response/krypto/polynomial.py) | the mock paper's models, **copied unchanged**: `poly_logreg`, `agreements`, `polyfeatures_kernel_table`, `kernel_logreg`, `mock_kernel_accuracies`; our models: `parity_features` (explicit feature map of the normalised kernel) and `SparsePolyLogReg` (polynomial LR on selected bits) |
| [`mlp.py`](kryptonite-response/krypto/mlp.py) | `mlp(n, width, depth, dropout)`; one shared training loop `train` (AdamW, BCE loss, early stopping on validation accuracy, best weights restored); `logits` / `accuracy` |
| [`lora.py`](kryptonite-response/krypto/lora.py) | **LoRA implemented from scratch**: `LoRALinear` ($W\mathbf x+\mathbf b+\frac{\alpha}{r}B(A\mathbf x)$, Kaiming-initialised $A$, $B=0$, `merged()` folds the update into $W$); `add_lora`, `linear_probe`, `mlp_probe`, `full_finetune`, `swap_oracle` (analytic rank-1 swap); operation counts `ops_full`, `ops_lora_BAx`, `ops_lora_BA_x` |
| [`stats.py`](kryptonite-response/krypto/stats.py) | `mean_ci` (95% Student-t interval), `binomial_ci`, `save_json` / `load_json` |
| [`energy.py`](kryptonite-response/krypto/energy.py) | `track(name)` logs the wall-clock time of each experiment to `results/runtime_log.json`; `estimate` converts time to energy and CO₂ with the Green Algorithms model |
| [`plotting.py`](kryptonite-response/krypto/plotting.py) | paper figure style (figures at printed size, 10 pt text, LaTeX-rendered Type-1 fonts), `save` |

### 4.2 Notebooks

| Notebook | Question / hypothesis | Method | Output |
|---|---|---|---|
| [`01_dataset_anatomy`](kryptonite-response/notebooks/01_dataset_anatomy.ipynb) | What structure is in the data? | Inspect the input distribution → greedy feature selection (10 coordinates for every task) → Fourier analysis (degree weights, influences) → class-respecting permutation matching shows all five tasks are one function → multiplicative identity recovers the gadget blocks → linear program gives PTF degree 6 | **Figure 1**, `anatomy.json` |
| [`02_polynomial_models`](kryptonite-response/notebooks/02_polynomial_models.ipynb) | **H1**: polynomial models fail through estimation, not approximation | Reproduce the mock paper's kernel model exactly → same model on the 10 selected bits only → additionally tune C on validation data → infinite-data limit | **Figure 2**, `poly_*.json` |
| [`03_model_comparison`](kryptonite-response/notebooks/03_model_comparison.ipynb) | **H2**: only learners that learn or select their representation succeed | Choose MLP hyper-parameters on seed 0; nine learners × the same five splits, mean ± 95% CI; time every fit | **Table 1**, `model_comparison.json`, `mlp_config.json` |
| [`04_learning_curves`](kryptonite-response/notebooks/04_learning_curves.ipynb) | **H3**: more data helps capable models | Nested subsets from 250 points to the full split; quadratic LR / MLP / sparse polynomial; interpolate the data needed to reach 90% | **Figure 3**, `learning_curves*.json` |
| [`05_kryptonet_lora`](kryptonite-response/notebooks/05_kryptonet_lora.ipynb) | **H4**: LoRA adapts KryptoNet-40 | Mechanism (coordinate swap, bit probes, column norms) → 13 methods × 5 splits → sample-efficiency curves → inspect the learned $A$ | **Figure 4**, **Table 3**, `kryptonet_*.json` |
| [`06_lora_operation_count`](kryptonite-response/notebooks/06_lora_operation_count.ipynb) | Cost of one LoRA training step | Measure forward + backward with `FlopCounterMode` and `assert` equality with the analytic formulas; compare full fine-tuning, $B(AX)$ and $(BA)X$; whole-network step time | **Table 2**, `lora_flops.json`, `lora_network_cost.json` |
| [`07_hidden_predictions`](kryptonite-response/notebooks/07_hidden_predictions.ipynb) | Final submission | 20/30/40: sparse polynomial LR refitted on all labelled data; Adapt: majority vote of five rank-1 LoRA models; cross-checks with independent models; runs `check_submission.py` | `hiddenlabels/*.npy`, `hidden_predictions_summary.json` |
| [`08_sustainability`](kryptonite-response/notebooks/08_sustainability.ipynb) | Environmental impact | Convert the logged runtimes to energy and CO₂ with the Green Algorithms model (with a sensitivity value); compare cost and accuracy of single fits | **Table 4**, `sustainability.json` |

### 4.3 Experimental protocol (shared by all notebooks)

- **Splits**: Kryptonite-n uses five stratified 60/20/20 splits (seeds 0–4), shared by every model so comparisons are paired; Adapt uses five 64/16/20 splits (800 test points, as in the mock paper).
- **No leakage**: feature selection, hyper-parameters, regularisation and early stopping use only training and validation data; each test set is used once per model and split.
- **Reporting**: mean ± 95% Student-t confidence interval.
- **Controls**: first reproduce the mock paper's results with its own code and split, then change one factor at a time.

---

## 5. Paper outline

Source: [`kryptonite-response/report/main.tex`](kryptonite-response/report/main.tex). Title: *Ten Hidden Bits and a Rank-One Fix: A Response to Kryptonite-3.0*. Main text 3–6 pages (ICML two-column format); references and appendices do not count.

| Section | Content | Figures / tables / propositions |
|---|---|---|
| **Abstract** | One paragraph: the mock paper's three claims; every task is the same 10-bit function (3.9% noise, Bayes ≈ 96%); degree-6 PTF → sparse polynomial and MLP above 90%; quadratic model capped at 56.5%; Adapt is a coordinate swap → rank-1 LoRA (552 parameters) ≈ 95% | — |
| **1 Introduction** | Challenge datasets are useful only if each failure is **attributed correctly** (approximation / estimation / optimisation); each claim treated as a falsifiable hypothesis; method overview (Fourier analysis + controlled experiments); main findings; GitLab footnote | Figure 1 (anatomy) |
| **2 Methodology** | | |
| 2.1 Data Model and Fourier Analysis | Bits $\mathbf b=\mathrm{sgn}(\mathbf x)$; Fourier–Walsh expansion; degree weights, influences; a function of k coordinates is exactly a polynomial of degree ≤ k | Figure 2 (polynomials) |
| 2.2 Polynomial Logistic Regression | Loss and C; monomials on ±1 reduce to parities → PTF; the mock paper's normalised kernel ↔ primal features | **Proposition 1** (quadratic model is capped) |
| 2.3 Kernels, Trees and Neural Networks | UAT is an existence statement; dot-product kernels learn only degree ≈ $\log N/\log n$; trees need marginal signal; MLP with the AdamW update and its hyper-parameters; leap complexity | Table 1 (model comparison) |
| 2.4 Low-Rank Adaptation and Its Operation Count | **Required by the coursework**: LoRA definition (what is trained, what is frozen, initialisation of A and B, role of the rank r, α); term-by-term operation count; $B(AX)$ vs full fine-tuning vs $(BA)X$; trainable parameters; effect of evaluation order | Table 2 (operation counts), **Proposition 2** (rank 1 suffices; probes are at exactly ½), Figure 3 (learning curves) |
| **3 Experimental Design** | Hypotheses H1–H4; validity safeguards (shared repeated splits, no test leakage, reproduction and controls, sampling error); feature selection; models and hyper-parameters | — |
| **4 Experimental Results** | | |
| 4.1 Anatomy of Kryptonite-n | Noisy bits; 10 coordinates; one function for five tasks; $f=b_pb_q h_A h_B$; no low-degree weight; PTF degree 6; leap 2; Adapt's coordinate swap | Figure 4 (KryptoNet-40) |
| 4.2 Polynomial Models: Estimation, Not Approximation | Reproduction → 10 selected bits → tuned C → jump from degree 5 to 6; infinite-data limit. **H1 supported** | Figure 2 |
| 4.3 Which Learners Succeed | Fixed-representation learners at 50–59%; MLP and sparse polynomial above 90%. **H2 supported** | Table 1 |
| 4.4 More Data Helps Capable Models | Quadratic model hugs its limit; MLP plateaus then jumps, inside the range the mock paper examined. **H3 supported** | Figure 3 |
| 4.5 Adapting KryptoNet-40 | Probes at chance and why; "worse than random" lies within ±3.5 points; scratch training fails; full fine-tuning 90.2%; rank-1 LoRA 95.4%; what LoRA learned; sample efficiency; hidden-set prediction procedure. **H4 supported** | Figure 4, Table 3 |
| **5 Discussion** | Three failures, three causes; limitations and further experiments; **environmental impact assessment** (time → energy → CO₂, effect of model choice); societal perspective | Table 4 (energy) |
| **Appendix A** Proofs | Proofs of Propositions 1 and 2 | — |
| **Appendix B** Generator | Exact coordinates of the pretraining task; structure of the gadgets | — |
| **Appendix C** Additional Details | Notebook behind each figure/table; baseline settings; full table of adaptation methods; energy table | Table 3, Table 4 |

**Figures and tables**

| Item | Content | Produced by |
|---|---|---|
| Figure 1 | Coordinate-role heat map + Fourier degree weights | 01 |
| Figure 2 | Polynomial LR accuracy vs degree (all bits / 10 bits with fixed C / 10 bits with tuned C / infinite-data limit) | 02 |
| Figure 3 | Learning curves (log x-axis, mock paper's range and quadratic limit marked) | 04 |
| Figure 4 | KryptoNet-40 adaptation methods + sample efficiency | 05 |
| Table 1 | Nine learners on n = 20/30/40: accuracy and time | 03 |
| Table 2 | LoRA operation counts (measured) | 06 |
| Table 3 | All KryptoNet-40 adaptation methods | 05 |
| Table 4 | Energy and emissions | 08 |

---

## 6. Pre-submission checklist

- [ ] Author names in `main.tex` (currently `Group Member One…`) and corresponding author
- [ ] GitLab URL and commit hash in the first-page footnote (currently `GROUP` / `COMMITHASH`)
- [ ] The coursework requires `hiddenlabels/` at the **top level of the submitted repository**: either move `kryptonite-response/hiddenlabels/` to the top level, or use `kryptonite-response/` itself as the repository root
- [ ] `python check_submission.py .` reports OK for all four files
- [ ] All notebooks are committed with their outputs
- [ ] Main text ≤ 6 pages; figure text no smaller than the body font; PDF uses only Type-1 fonts (`pdffonts main.pdf`)
- [ ] Every group member submits the peer assessment form

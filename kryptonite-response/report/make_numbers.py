"""Generate report/generated/*.tex (numbers and table bodies) from results/*.json and copy figures/ into report/.

Run after the notebooks:  python report/make_numbers.py
Every number quoted in the paper comes from here, so text, tables and notebooks cannot disagree.
"""
import json
import shutil
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "report" / "generated"
OUT.mkdir(exist_ok=True)
WORDS = {20: "Twenty", 30: "Thirty", 40: "Forty"}


def load(name):
    p = RES / name
    return json.load(open(p)) if p.exists() else None


def mci(v):
    v = np.asarray(v, float)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return 100 * v.mean(), 100 * h


def pm(v, bold_above=None):
    m, h = mci(v)
    s = f"{m:.1f} $\\pm$ {h:.1f}"
    return f"\\textbf{{{s}}}" if bold_above is not None and m >= bold_above else s


macros = {}


def define(name, value):
    macros[name] = value


# ---------------------------------------------------------------- anatomy / polynomial limits
lim = load("poly_limit.json")
if lim:
    define("etaPct", f"{100 * lim['eta']:.1f}")
    define("BayesPct", f"{100 * (1 - lim['eta']):.1f}")
    define("LimitTwo", f"{100 * lim['limit']['2']:.1f}")
    define("LimitSix", f"{100 * lim['limit']['6']:.1f}")
    define("LimitSeven", f"{100 * lim['limit']['7']:.1f}")
an = load("anatomy.json")
if an:
    W = np.array(an["degree_weights"]["pre"])
    define("WeightFourEight", f"{100 * W[4:9].sum():.0f}")

mock = load("poly_mock_reproduction.json")
if mock:
    for n in (20, 30, 40):
        define(f"Mock{WORDS[n]}Ten", f"{100 * mock[str(n)]['10']:.1f}")
sp = load("poly_sparse.json")
if sp:
    for n in (20, 30, 40):
        define(f"SelFixed{WORDS[n]}", f"{mci(sp[str(n)]['fixed']['10'])[0]:.1f}")
        define(f"SelTuned{WORDS[n]}", f"{mci(sp[str(n)]['tuned']['10'])[0]:.1f}")
        define(f"SelTunedSix{WORDS[n]}", f"{mci(sp[str(n)]['tuned']['6'])[0]:.1f}")

# ---------------------------------------------------------------- Table 1: model comparison
mc = load("model_comparison.json")
if mc:
    acc, sec = mc["accuracy"], mc["seconds"]
    for n in (20, 30, 40):
        define(f"MLP{WORDS[n]}Mean", f"{mci(acc[str(n)]['MLP, all n inputs (ours)'])[0]:.1f}")
        define(f"Sparse{WORDS[n]}Mean", f"{mci(acc[str(n)]['Sparse polynomial LR (ours)'])[0]:.1f}")
    ROWS = [("Logistic regression (linear)", "Linear logistic regression", "fixed"),
            ("Quadratic LR (mock paper)", "Quadratic LR (mock paper)", "fixed"),
            ("Poly-kernel LR, d=10, all bits (mock)", "Degree-10 kernel LR (mock paper)", "fixed"),
            ("k-nearest neighbours (k tuned)", "$k$-nearest neighbours", "fixed"),
            ("RBF-kernel SVM", "RBF-kernel SVM", "fixed"),
            ("Random forest (300 trees)", "Random forest", "greedy splits"),
            ("Gradient-boosted trees", "Gradient-boosted trees", "greedy splits"),
            ("MLP, all n inputs (ours)", "MLP, all $n$ inputs (ours)", "learned"),
            ("Sparse polynomial LR (ours)", "Sparse polynomial LR (ours)", "selected")]
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Model & Representation & $n=20$ & $n=30$ & $n=40$ & Time (s) \\", r"\midrule"]
    for key, label, rep in ROWS:
        t = sec["40"][key][1:] if key.startswith("MLP") else sec["40"][key]
        cells = " & ".join(pm(acc[str(n)][key], bold_above=90) for n in (20, 30, 40))
        tm = float(np.mean(t))
        tstr = "$<$0.1" if tm < 0.1 else (f"{tm:.1f}" if tm < 10 else f"{tm:.0f}")
        lines.append(f"{label} & {rep} & {cells} & {tstr} \\\\")
        if key.startswith("Gradient"):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_models.tex").write_text("\n".join(lines) + "\n")
    define("TimeKernel", f"{np.mean(sec['40']['Poly-kernel LR, d=10, all bits (mock)']):.0f}")
    define("TimeSparse", f"{np.mean(sec['40']['Sparse polynomial LR (ours)']):.0f}")
    define("TimeMLP", f"{np.mean(sec['40']['MLP, all n inputs (ours)'][1:]):.0f}")
cfg = load("mlp_config.json")
if cfg:
    define("MLPconfigs", "; ".join(f"$n={n}$: depth {c['depth']}, dropout {c['dropout']}" for n, c in cfg.items()))

# ---------------------------------------------------------------- learning curves
th = load("learning_curve_thresholds.json")
if th:
    def rnd(v):
        return "n/a" if v is None else f"{int(round(v, -2)):,}"
    for n in (20, 30, 40):
        define(f"Nmlp{WORDS[n]}", rnd(th[f"{n}|MLP"]))
        define(f"Nsparse{WORDS[n]}", rnd(th[f"{n}|Sparse poly"]))
    define("MLPgrowth", f"{th['40|MLP'] / th['20|MLP']:.1f}")

# ---------------------------------------------------------------- KryptoNet-40
km = load("kryptonet_methods.json")
if km:
    a, p = km["accuracy"], km["params"]
    for key, name in (("Zero-shot", "ZeroShotMean"), ("Linear probe", "LPMean"), ("MLP probe", "MLPProbeMean"),
                      ("Scratch (same architecture)", "ScratchMean"), ("Full fine-tuning", "FullFTMean"),
                      ("LoRA, W1, r=1", "LoRAoneMean"), ("Rank-1 swap (analytic, no training)", "SwapMean")):
        define(name, f"{mci(a[key])[0]:.1f}")
    LABEL = {"Zero-shot": "Zero-shot", "Linear probe": "Linear probe (mock)", "MLP probe": "MLP probe (mock)",
             "Scratch (same architecture)": "From scratch", "Full fine-tuning": "Full fine-tuning",
             "LoRA, W1, r=1": r"LoRA on $\mathbf{W}_1$, $r=1$", "LoRA, W1, r=2": r"LoRA on $\mathbf{W}_1$, $r=2$",
             "LoRA, W1, r=4": r"LoRA on $\mathbf{W}_1$, $r=4$", "LoRA, W1, r=8": r"LoRA on $\mathbf{W}_1$, $r=8$",
             "LoRA, W1-W4, r=1": r"LoRA on $\mathbf{W}_{1..4}$, $r=1$", "LoRA, W1-W4, r=4": r"LoRA on $\mathbf{W}_{1..4}$, $r=4$",
             "LoRA, W1-W4, r=8": r"LoRA on $\mathbf{W}_{1..4}$, $r=8$",
             "Rank-1 swap (analytic, no training)": "Analytic rank-1 swap"}
    lines = [r"\begin{tabular}{lrr}", r"\toprule", r"Method & Parameters & Accuracy (\%) \\", r"\midrule"]
    for key, label in LABEL.items():
        lines.append(f"{label} & {p[key]:,} & {pm(a[key], bold_above=90)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_kryptonet.tex").write_text("\n".join(lines) + "\n")
    lora_cfgs = [k for k in a if k.startswith("LoRA")]
    define("LoRAminMean", f"{min(mci(a[k])[0] for k in lora_cfgs):.1f}")
mech = load("kryptonet_mechanism.json")
if mech:
    define("SwapZeroShotPct", f"{100 * mech['zero_shot_swapped_all']:.1f}")
    define("ZeroShotAllPct", f"{100 * mech['zero_shot_all']:.1f}")
    define("ProbeBitZero", f"{100 * mech['probe_bit']['0']:.0f}")
    define("ProbeBitNineteen", f"{100 * mech['probe_bit']['19']:.0f}")
se = load("kryptonet_sample_efficiency.json")
if se:
    for s, w in (("400", "FourHundred"), ("800", "EightHundred"), ("1600", "SixteenHundred")):
        define(f"LoRAat{w}", f"{mci(se['LoRA, W1, r=1'][s])[0]:.1f}")
        define(f"FullFTat{w}", f"{mci(se['Full fine-tuning'][s])[0]:.1f}")
    define("FullFTatMax", f"{mci(se['Full fine-tuning']['2560'])[0]:.1f}")
insp = load("kryptonet_lora_inspection.json")
if insp:
    A = np.array(insp["A"])
    define("LoRAaZero", f"{A[0]:+.2f}")
    define("LoRAaNineteen", f"{A[19]:+.2f}")
    define("LoRAaRest", f"{np.abs(np.delete(A, [0, 19])).max():.2f}")
    define("LoRAcos", f"{insp['cosine']:.2f}")

# ---------------------------------------------------------------- operation counts
fl, net = load("lora_flops.json"), load("lora_network_cost.json")
if fl and net:
    def block(title, key):
        rows = {r[0]: r for r in fl[key]}
        out = [f"\\multicolumn{{4}}{{l}}{{\\emph{{{title}}}}} \\\\"]
        for name, label in (("full fine-tuning", "Full fine-tuning"), ("LoRA, B(AX)", r"LoRA, $\mathbf{B}(\mathbf{A}\mathbf{X})$"),
                            ("LoRA, (BA)X", r"LoRA, $(\mathbf{B}\mathbf{A})\mathbf{X}$")):
            _, f, b, t, prm = rows[name]
            out.append(f"{label} & {f / 1e6:.2f} & {b / 1e6:.2f} & {prm:,} \\\\")
        return out
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Method & Forward & Backward & Trainable \\", r"\midrule"]
    lines += block(r"Hidden layer $\mathbf{W}_2$ ($512\times512$), $r=4$", "hidden layer 512x512|N=64|r=4")
    lines += [r"\midrule"] + block(r"First layer $\mathbf{W}_1$ ($512\times40$), $r=1$", "first layer 512x40|N=64|r=1")
    lines += [r"\midrule", r"\multicolumn{4}{l}{\emph{Whole KryptoNet-40}} \\"]
    for name, label in (("full fine-tuning", "Full fine-tuning"), ("LoRA W1, r=1", r"LoRA on $\mathbf{W}_1$, $r=1$")):
        f, b, prm, _ = net[name]
        lines.append(f"{label} & {f / 1e6:.2f} & {b / 1e6:.2f} & {prm:,} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_ops.tex").write_text("\n".join(lines) + "\n")
    full = sum(net["full fine-tuning"][:2]); lora = sum(net["LoRA W1, r=1"][:2])
    define("NetRatio", f"{lora / full:.2f}")
    define("StepFull", f"{1e3 * net['full fine-tuning'][3]:.1f}")
    define("StepLoRA", f"{1e3 * net['LoRA W1, r=1'][3]:.1f}")

# ---------------------------------------------------------------- hidden predictions
hp = load("hidden_predictions_summary.json")
if hp:
    for n in (20, 30, 40):
        define(f"Agree{WORDS[n]}", f"{100 * hp[str(n)]['agreement_with_MLP']:.1f}")
    define("AgreeSwap", f"{100 * hp['40_adapt']['agreement_with_rank1_swap']:.1f}")

# ---------------------------------------------------------------- sustainability
su = load("sustainability.json")
if su:
    define("TotalHours", f"{su['total_hours']:.1f}")
    define("TotalWh", f"{su['total_Wh']:.0f}")
    define("TotalG", f"{su['total_g']:.0f}")
    define("TotalGhi", f"{su['total_g_hi']:.0f}")
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Notebook & Time (h) & Energy (Wh) & CO$_2$e (g) \\",
             r"\midrule"]
    SHORT = {"01": "01 Anatomy", "02": "02 Polynomials", "03": "03 Comparison", "04": "04 Curves",
             "05": "05 KryptoNet-40", "06": "06 Op.\\ counts", "07": "07 Final models"}
    for name, h, wh, g, g_hi in su["groups"]:
        label = SHORT[name.split("(nb ")[1][:2]]
        lines.append(f"{label} & {h:.2f} & {wh:.1f} & {g:.1f} \\\\")
    lines += [r"\midrule", f"Total & {su['total_hours']:.2f} & {su['total_Wh']:.1f} & {su['total_g']:.1f} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_energy.tex").write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------- write macros (missing ones become ??)
REQUIRED = """etaPct BayesPct LimitTwo LimitSix LimitSeven WeightFourEight MockTwentyTen MockThirtyTen MockFortyTen
SelFixedTwenty SelFixedThirty SelFixedForty SelTunedTwenty SelTunedThirty SelTunedForty MLPTwentyMean MLPThirtyMean
MLPFortyMean SparseTwentyMean SparseThirtyMean SparseFortyMean TimeKernel TimeSparse TimeMLP MLPconfigs NmlpTwenty
NmlpThirty NmlpForty MLPgrowth NsparseTwenty NsparseThirty NsparseForty ZeroShotMean LPMean MLPProbeMean ScratchMean FullFTMean
LoRAoneMean SwapMean LoRAminMean SwapZeroShotPct ZeroShotAllPct ProbeBitZero ProbeBitNineteen LoRAatFourHundred
LoRAatEightHundred LoRAatSixteenHundred FullFTatFourHundred FullFTatEightHundred FullFTatSixteenHundred FullFTatMax
LoRAaZero LoRAaNineteen LoRAaRest LoRAcos NetRatio StepFull StepLoRA AgreeTwenty AgreeThirty AgreeForty AgreeSwap
TotalHours TotalWh TotalG TotalGhi""".split()
with open(OUT / "numbers.tex", "w") as f:
    f.write("% Generated by report/make_numbers.py from results/*.json -- do not edit by hand.\n")
    for name in REQUIRED:
        f.write(f"\\newcommand{{\\{name}}}{{{macros.get(name, '??')}}}\n")
for name in ("table_models", "table_kryptonet", "table_ops", "table_energy"):
    if not (OUT / f"{name}.tex").exists():
        (OUT / f"{name}.tex").write_text("(results pending)\n")
# copy the figures next to the LaTeX source so that report/ is self-contained (e.g. for Overleaf)
(ROOT / "report" / "figures").mkdir(exist_ok=True)
for fig in sorted((ROOT / "figures").glob("fig*.pdf")):
    shutil.copy(fig, ROOT / "report" / "figures" / fig.name)
missing = [n for n in REQUIRED if n not in macros]
print(f"wrote {len(REQUIRED) - len(missing)} numbers; missing: {missing}")

"""Check that your hidden-set predictions are in the format ImpCML 2026 expects.

Run it from the top level of your GitLab repository (or pass the repository's path):
    python check_submission.py [path/to/your/repository]
"""
import os
import sys
import numpy as np

FILES = ["y_predicted_20.npy", "y_predicted_30.npy", "y_predicted_40.npy", "y_predicted_40_adapt.npy"]
N_HIDDEN = 10_000


def problems_with(path):
    if not os.path.isfile(path):
        return ["file is missing"]
    try:
        y = np.load(path, allow_pickle=False)
    except Exception as e:
        return [f"cannot be loaded with numpy.load ({e})"]
    problems = []
    if y.shape != (N_HIDDEN,):
        problems.append(f"has shape {y.shape}, expected ({N_HIDDEN},)")
    if not (np.issubdtype(y.dtype, np.number) or y.dtype == bool):
        problems.append(f"has dtype {y.dtype}, expected integers or booleans")
    elif not np.isin(np.unique(y), [0, 1]).all():
        problems.append("contains values other than 0 and 1 (submit labels, not probabilities)")
    return problems


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    folder = os.path.join(repo, "hiddenlabels")
    if not os.path.isdir(folder):
        print(f"No 'hiddenlabels' directory found in {os.path.abspath(repo)}")
        sys.exit(1)
    ok = True
    for name in FILES:
        problems = problems_with(os.path.join(folder, name))
        ok &= not problems
        print(f"{'OK  ' if not problems else 'FAIL'}  {name}" + (f": {'; '.join(problems)}" if problems else ""))
    extra = sorted(f for f in os.listdir(folder) if f.endswith(".npy") and f not in FILES)
    if extra:
        print(f"Note: these files will be ignored by the grader: {', '.join(extra)}")
    print("All files are in the right format." if ok else "Fix the problems above before submitting.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

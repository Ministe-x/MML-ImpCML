"""Wall-clock logging for the sustainability analysis.

Every experiment is wrapped in `track(name)`, which appends its wall-clock time and the number of
CPU threads used to results/runtime_log.json.  Energy is estimated afterwards (notebook 08) with the
Green Algorithms model (Lannelongue et al., 2021):  E = t * (n_cores * P_core + mem_GB * P_mem) * PUE.
"""
import json
import os
import platform
import time
from contextlib import contextmanager

import torch

from .data import RESULTS

LOG = RESULTS / "runtime_log.json"


@contextmanager
def track(name, notes=""):
    start = time.time()
    yield
    entry = {"name": name, "seconds": round(time.time() - start, 2), "threads": torch.get_num_threads(),
             "cpu": platform.processor() or platform.machine(), "cores_visible": os.cpu_count(), "notes": notes}
    RESULTS.mkdir(exist_ok=True)
    log = json.load(open(LOG)) if LOG.exists() else []
    log = [e for e in log if e["name"] != name] + [entry]
    json.dump(log, open(LOG, "w"), indent=1)
    print(f"[track] {name}: {entry['seconds']:.1f} s")


def estimate(seconds, cores=2, p_core=12.0, mem_gb=4.0, p_mem=0.3725, pue=1.0, grid_g_per_kwh=128.8):
    """Energy (kWh) and emissions (g CO2e) of a computation of the given duration."""
    kwh = seconds / 3600 * (cores * p_core + mem_gb * p_mem) * pue / 1000
    return kwh, kwh * grid_g_per_kwh

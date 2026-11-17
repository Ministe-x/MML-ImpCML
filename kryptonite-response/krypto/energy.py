"""Logging run times so that we can estimate the energy cost of the project (notebook 08).

We have no access to power counters, so we record what we can measure, the wall-clock time and the
number of CPU threads, and convert it to energy afterwards with the Green Algorithms model
(Lannelongue et al., 2021):  E = t * (n_cores * P_core + mem_GB * P_mem) * PUE.
Each experiment is wrapped in `with track(name):`, which writes one entry to results/runtime_log.json.
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
    # re-running an experiment replaces its old entry, so the totals count every experiment once
    log = [e for e in log if e["name"] != name] + [entry]
    json.dump(log, open(LOG, "w"), indent=1)
    print(f"[track] {name}: {entry['seconds']:.1f} s")


def estimate(seconds, cores=2, p_core=12.0, mem_gb=4.0, p_mem=0.3725, pue=1.0, grid_g_per_kwh=128.8):
    """Energy (kWh) and emissions (g CO2e) of a run of the given length; the defaults are our assumptions
    (2 cores at 12 W, 4 GB of memory, PUE 1, the 2025 average carbon intensity of the GB grid)."""
    kwh = seconds / 3600 * (cores * p_core + mem_gb * p_mem) * pue / 1000
    return kwh, kwh * grid_g_per_kwh

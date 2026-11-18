"""CorrDiff normalisation statistics (PhysicsNeMo examples/weather/corrdiff, hrrr_mini / custom datasets).

File layout (JSON):  {"input": {var: {"mean": m, "std": s}}, "output": {...}, "invariant": {...}}

CorrDiff normalises every input channel with these numbers. A network trained on ERA5-like inputs has
learned what "0" and "1" mean for each channel; feeding KIM with the same statistics keeps that meaning
(KIM biases then appear as shifted inputs), while replacing the input statistics with KIM's own
re-centres KIM but also removes real differences. Experiment arms A and C in kcadapt.arms compare
the two. Output statistics must never change for a pretrained model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

GROUPS = ("input", "output", "invariant")


def compute(ds: xr.Dataset, variables=None) -> dict:
    variables = variables or list(ds.data_vars)
    out = {}
    for v in variables:
        x = np.asarray(ds[v].values, dtype=float)
        x = x[np.isfinite(x)]
        s = float(x.std())
        out[v] = {"mean": float(x.mean()), "std": s if s > 0 else 1.0}
    return out


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(stats: dict, path: str | Path) -> Path:
    unknown = set(stats) - set(GROUPS)
    if unknown:
        raise ValueError(f"unknown groups {unknown}; expected {GROUPS}")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return p


def build(input_ds: xr.Dataset, output_ds: xr.Dataset, invariant_ds: xr.Dataset | None = None) -> dict:
    s = {"input": compute(input_ds), "output": compute(output_ds)}
    if invariant_ds is not None:
        s["invariant"] = compute(invariant_ds)
    return s


def replace_input(stats: dict, new_input: dict, variables=None) -> dict:
    """Copy of `stats` with the input statistics of `variables` taken from `new_input` (arm C).
    Output and invariant statistics are kept."""
    variables = variables or list(new_input)
    out = json.loads(json.dumps(stats))
    for v in variables:
        if v not in out.get("input", {}):
            raise KeyError(f"{v} not in the original input statistics")
        out["input"][v] = dict(new_input[v])
    return out


def check_compatible(stats: dict, input_vars, output_vars) -> list[str]:
    problems = []
    for g, vs in (("input", input_vars), ("output", output_vars)):
        missing = [v for v in vs if v not in stats.get(g, {})]
        if missing:
            problems.append(f"{g}: missing {', '.join(missing)}")
        bad = [v for v in vs if v in stats.get(g, {}) and not stats[g][v]["std"] > 0]
        if bad:
            problems.append(f"{g}: non-positive std for {', '.join(bad)}")
    return problems

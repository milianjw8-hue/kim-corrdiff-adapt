"""How different are KIM inputs from the ERA5-like inputs CorrDiff was trained on?

All functions take xarray objects with a ``time`` dimension and spatial dimensions (any names).
Pooled statistics use every time and grid point.

Metrics per variable
--------------------
mean_bias      mean(src) - mean(ref)
std_ratio      std(src) / std(ref)
w1             1-Wasserstein distance of the pooled distributions (mean |q_src(p) - q_ref(p)|)
w1_norm        w1 / std(ref)  (unitless; comparable across variables)
z_shift        (mean(src) - mean_train) / std_train with the CorrDiff training statistics, if given:
               how far the new inputs sit from the centre the network was normalised to
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

PROBS = np.linspace(0.005, 0.995, 199)

# screening thresholds (heuristic, not from a reference): flag a channel for correction when exceeded
THRESHOLDS = {"w1_norm": 0.10, "std_ratio_low": 0.90, "std_ratio_high": 1.10, "z_shift": 0.10}


def _flat(da: xr.DataArray) -> np.ndarray:
    x = np.asarray(da.values, dtype=float).ravel()
    return x[np.isfinite(x)]


def pooled_shift(src: xr.DataArray, ref: xr.DataArray) -> dict:
    a, b = _flat(src), _flat(ref)
    qa, qb = np.quantile(a, PROBS), np.quantile(b, PROBS)
    sd = float(b.std())
    w1 = float(np.mean(np.abs(qa - qb)))
    return {"n_src": a.size, "n_ref": b.size, "mean_src": float(a.mean()), "mean_ref": float(b.mean()),
            "mean_bias": float(a.mean() - b.mean()), "std_ratio": float(a.std() / sd) if sd > 0 else np.nan,
            "w1": w1, "w1_norm": w1 / sd if sd > 0 else np.nan}


def shift_table(src: xr.Dataset, ref: xr.Dataset, variables=None, stats: dict | None = None,
                group: str = "input") -> pd.DataFrame:
    """One row per variable. `stats` is a CorrDiff stats JSON dict (see kcadapt.stats)."""
    variables = variables or [v for v in src.data_vars if v in ref.data_vars]
    rows = []
    for v in variables:
        r = {"variable": v, **pooled_shift(src[v], ref[v])}
        if stats is not None and v in stats.get(group, {}):
            m, s = stats[group][v]["mean"], stats[group][v]["std"]
            r["z_shift"] = (r["mean_src"] - m) / s
            r["z_ref"] = (r["mean_ref"] - m) / s
            r["std_ratio_train"] = float(_flat(src[v]).std() / s)
        rows.append(r)
    t = pd.DataFrame(rows)
    return flag(t)


def flag(t: pd.DataFrame) -> pd.DataFrame:
    """Add `needs_correction` and the reason (screening only)."""
    reasons = []
    for r in t.itertuples():
        why = []
        if getattr(r, "w1_norm", np.nan) > THRESHOLDS["w1_norm"]:
            why.append("distribution")
        sr = getattr(r, "std_ratio", np.nan)
        if sr < THRESHOLDS["std_ratio_low"] or sr > THRESHOLDS["std_ratio_high"]:
            why.append("variance")
        z = getattr(r, "z_shift", np.nan)
        if np.isfinite(z) and abs(z - getattr(r, "z_ref", 0.0)) > THRESHOLDS["z_shift"]:
            why.append("normalisation")
        reasons.append(",".join(why))
    t = t.copy()
    t["reason"] = reasons
    t["needs_correction"] = t["reason"] != ""
    return t


def bias_maps(src: xr.Dataset, ref: xr.Dataset, variables=None, time: str = "time") -> xr.Dataset:
    """Time-mean difference and standard-deviation ratio at every grid point."""
    variables = variables or [v for v in src.data_vars if v in ref.data_vars]
    out = {}
    for v in variables:
        out[f"{v}_mean_bias"] = src[v].mean(time) - ref[v].mean(time)
        out[f"{v}_std_ratio"] = src[v].std(time) / ref[v].std(time)
    return xr.Dataset(out)


def common_times(src: xr.Dataset, ref: xr.Dataset, time: str = "time") -> tuple[xr.Dataset, xr.Dataset]:
    t = np.intersect1d(src[time].values, ref[time].values)
    if t.size == 0:
        raise ValueError("no common times between the two datasets")
    return src.sel({time: t}), ref.sel({time: t})

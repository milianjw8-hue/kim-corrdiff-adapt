"""SYNTHETIC data to test the workflow without KIM, ERA5 or a GPU. Nothing here is real data.

era5   coarse 'ERA5-like' inputs: smooth random fields with an annual cycle
kim    the same weather seen by a different analysis: a spatial bias pattern, inflated wind variance,
       a cold bias in winter and extra small-scale noise
target fine-grid field (factor 4) built from the ERA5 field plus fixed small-scale detail
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

VARS = ("u10m", "v10m", "t2m", "msl")


def _smooth_noise(rng, shape, scale=3):
    x = rng.normal(size=shape)
    k = np.ones(scale) / scale
    for ax in (-2, -1):
        x = np.apply_along_axis(lambda v: np.convolve(np.r_[v[-scale:], v, v[:scale]], k, "same")[scale:-scale],
                                ax, x)
    return x / x.std()


def make(n_years: int = 3, ny: int = 12, nx: int = 12, factor: int = 4, seed: int = 0, start="2022-01-01"):
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, periods=n_years * 365 * 4, freq="6h")
    nt = len(times)
    doy = times.dayofyear.values
    annual = np.cos(2 * np.pi * (doy - 15) / 365.25)          # +1 mid-January
    y = np.linspace(0, 1, ny)[:, None] * np.ones((1, nx))
    base = {
        "u10m": 2.0 + 5.0 * _smooth_noise(rng, (nt, ny, nx)),
        "v10m": 0.5 + 4.0 * _smooth_noise(rng, (nt, ny, nx)),
        "t2m": 288.0 - 10.0 * annual[:, None, None] - 6.0 * y + 2.0 * _smooth_noise(rng, (nt, ny, nx)),
        "msl": 101_300.0 + 400.0 * annual[:, None, None] + 600.0 * _smooth_noise(rng, (nt, ny, nx)),
    }
    coords = {"time": times, "y": np.arange(ny), "x": np.arange(nx)}
    era5 = xr.Dataset({k: (("time", "y", "x"), v.astype("float32")) for k, v in base.items()}, coords=coords,
                      attrs={"title": "SYNTHETIC ERA5-like inputs (not real data)"})

    bias = _smooth_noise(np.random.default_rng(seed + 1), (1, ny, nx), 5)[0]
    winter = np.clip(annual, 0, None)[:, None, None]
    kim = {
        "u10m": 1.25 * (base["u10m"] - 2.0) + 2.0 + 0.8 * bias + 0.5 * rng.normal(size=(nt, ny, nx)),
        "v10m": 1.20 * (base["v10m"] - 0.5) + 0.5 - 0.5 * bias + 0.5 * rng.normal(size=(nt, ny, nx)),
        "t2m": base["t2m"] - 1.5 * winter + 0.7 * bias + 0.3 * rng.normal(size=(nt, ny, nx)),
        "msl": base["msl"] + 80.0 * bias + 30.0 * rng.normal(size=(nt, ny, nx)),
    }
    kim = xr.Dataset({k: (("time", "y", "x"), v.astype("float32")) for k, v in kim.items()}, coords=coords,
                     attrs={"title": "SYNTHETIC KIM-like inputs (not real data)"})

    NY, NX = ny * factor, nx * factor
    detail = _smooth_noise(np.random.default_rng(seed + 2), (1, NY, NX), 2)[0]
    up = era5[["u10m", "v10m", "t2m"]].interp(y=np.linspace(0, ny - 1, NY), x=np.linspace(0, nx - 1, NX))
    up = up.assign_coords(y=np.arange(NY), x=np.arange(NX))
    target = xr.Dataset({"u10m": up["u10m"] * (1 + 0.2 * detail), "v10m": up["v10m"] * (1 + 0.2 * detail),
                         "t2m": up["t2m"] - 3.0 * detail}).astype("float32")
    target.attrs["title"] = "SYNTHETIC fine-grid target (not real data)"
    invariants = xr.Dataset({"orog": (("y", "x"), (500 * np.clip(detail, 0, None)).astype("float32")),
                             "lsm": (("y", "x"), (detail > -0.5).astype("float32"))},
                            coords={"y": np.arange(NY), "x": np.arange(NX)})
    return era5, kim, target, invariants

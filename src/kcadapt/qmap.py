"""Empirical quantile mapping of KIM inputs onto the ERA5-like input distribution.

The mapping is fitted on a period where both KIM and the reference (ERA5) exist, then applied to
KIM fields only. It corrects the marginal distribution of each variable (at each grid point, or
pooled over the domain), optionally separately for each calendar-month group. It does not correct
spatial patterns or the joint distribution between variables.

transfer at probability p:  additive        x' = x + (q_ref(p) - q_src(p))
                            multiplicative  x' = x * q_ref(p) / q_src(p)   (non-negative variables)
Beyond the fitted probability range the correction at the end point is kept (constant extrapolation).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

PROBS = np.linspace(0.01, 0.99, 99)
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
MULTIPLICATIVE_DEFAULT = ("tp", "tcwv", "precip", "rain")
EPS = 1e-6


def _groups(times, by: str | None):
    months = xr.DataArray(times).dt.month.values
    if by is None:
        return {"all": np.ones(len(times), bool)}
    if by == "season":
        return {k: np.isin(months, m) for k, m in SEASONS.items()}
    if by == "month":
        return {f"{m:02d}": months == m for m in range(1, 13)}
    raise ValueError("by must be None, 'season' or 'month'")


def _group_of(times, by):
    months = xr.DataArray(times).dt.month.values
    if by is None:
        return np.array(["all"] * len(times))
    if by == "season":
        lut = {m: k for k, ms in SEASONS.items() for m in ms}
        return np.array([lut[m] for m in months])
    return np.array([f"{m:02d}" for m in months])


def _map_point(x, qs, qr, mult):
    """Map values x with source quantiles qs and reference quantiles qr (1-D, ascending)."""
    # strictly increasing source quantiles for interpolation (ties in bounded variables)
    qs = qs + np.arange(qs.size) * 1e-9 * max(1.0, np.abs(qs).max())
    p = np.interp(x, qs, PROBS)
    if mult:
        ratio = np.interp(p, PROBS, (qr + EPS) / (qs + EPS))
        return np.maximum(x * ratio, 0.0)
    return x + np.interp(p, PROBS, qr - qs)


class QuantileMap:
    """Fit on overlapping (src, ref) data, apply to new src data.

    pointwise=True fits a separate mapping at every grid point (needs enough times: at least a few
    hundred per group); False pools the whole domain (robust with short records, ignores local biases).
    """

    def __init__(self, variables: list[str] | None = None, pointwise: bool = True, by: str | None = "season",
                 multiplicative: tuple[str, ...] = MULTIPLICATIVE_DEFAULT, time: str = "time"):
        self.variables = variables
        self.pointwise = pointwise
        self.by = by
        self.multiplicative = tuple(multiplicative)
        self.time = time
        self.q: xr.Dataset | None = None

    def fit(self, src: xr.Dataset, ref: xr.Dataset, min_samples: int = 30) -> "QuantileMap":
        t = self.time
        common = np.intersect1d(src[t].values, ref[t].values)
        if common.size == 0:
            raise ValueError("src and ref share no times")
        src, ref = src.sel({t: common}), ref.sel({t: common})
        self.variables = self.variables or [v for v in src.data_vars if v in ref.data_vars]
        groups = _groups(common, self.by)
        parts = {}
        for v in self.variables:
            qs, qr = [], []
            for g, mask in groups.items():
                if mask.sum() < min_samples:
                    raise ValueError(f"group {g} has {int(mask.sum())} times (< {min_samples}); use by=None "
                                     "or a longer overlap")
                a, b = src[v].isel({t: mask}), ref[v].isel({t: mask})
                dims = [t] if self.pointwise else list(a.dims)
                qs.append(self._q(a, dims).expand_dims(group=[g]))
                qr.append(self._q(b, dims).expand_dims(group=[g]))
            parts[f"{v}__src"] = xr.concat(qs, "group")
            parts[f"{v}__ref"] = xr.concat(qr, "group")
        q = xr.Dataset(parts)
        q.attrs.update({"pointwise": int(self.pointwise), "by": self.by or "none",
                        "multiplicative": ",".join(v for v in self.variables if v in self.multiplicative),
                        "variables": ",".join(self.variables), "fit_start": str(common.min()),
                        "fit_end": str(common.max()), "n_times": int(common.size)})
        self.q = q
        return self

    @staticmethod
    def _q(da: xr.DataArray, dims) -> xr.DataArray:
        q = da.quantile(PROBS, dim=dims).rename(quantile="prob")
        return q.drop_vars("prob").astype("float64")

    def transform(self, src: xr.Dataset) -> xr.Dataset:
        if self.q is None:
            raise RuntimeError("fit or load first")
        t = self.time
        out = src.copy()
        gid = _group_of(src[t].values, self.by)
        for v in self.variables:
            mult = v in self.multiplicative
            qs_all, qr_all = self.q[f"{v}__src"], self.q[f"{v}__ref"]
            da = src[v].transpose(t, ...)
            x = np.asarray(da.values, dtype=float)
            y = np.empty_like(x)
            for g in np.unique(gid):
                if g not in qs_all["group"].values:
                    raise ValueError(f"no mapping fitted for group {g}")
                sel = gid == g
                qs = qs_all.sel(group=g).transpose("prob", ...).values
                qr = qr_all.sel(group=g).transpose("prob", ...).values
                xs = x[sel]
                if self.pointwise:
                    flat_x = xs.reshape(xs.shape[0], -1)
                    flat_qs, flat_qr = qs.reshape(qs.shape[0], -1), qr.reshape(qr.shape[0], -1)
                    res = np.empty_like(flat_x)
                    for k in range(flat_x.shape[1]):
                        res[:, k] = _map_point(flat_x[:, k], flat_qs[:, k], flat_qr[:, k], mult)
                    y[sel] = res.reshape(xs.shape)
                else:
                    y[sel] = _map_point(xs.ravel(), qs.ravel(), qr.ravel(), mult).reshape(xs.shape)
            out[v] = (da.dims, y.astype(src[v].dtype))
            out[v] = out[v].transpose(*src[v].dims)
            out[v].attrs = {**src[v].attrs, "quantile_mapped": "kcadapt"}
        return out

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.q.to_netcdf(p)
        return p

    @classmethod
    def load(cls, path: str | Path, time: str = "time") -> "QuantileMap":
        q = xr.load_dataset(path)
        mult = tuple(x for x in q.attrs.get("multiplicative", "").split(",") if x)
        by = q.attrs.get("by", "none")
        m = cls(q.attrs["variables"].split(","), bool(q.attrs["pointwise"]), None if by == "none" else by, mult, time)
        m.q = q
        return m


def split_by_year(ds: xr.Dataset, test_years, time: str = "time") -> tuple[xr.Dataset, xr.Dataset]:
    """Train/test split by calendar year (keeps whole seasons together)."""
    years = ds[time].dt.year
    test = np.isin(years, list(test_years))
    return ds.isel({time: ~test}), ds.isel({time: test})

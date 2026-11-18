"""Write paired (coarse input, fine target) data in the layout of the PhysicsNeMo CorrDiff ``hrrr_mini``
dataset, so the stock reader can be used with ``dataset.type: hrrr_mini``.

NetCDF layout (checked against PhysicsNeMo examples/weather/corrdiff/datasets/hrrrmini.py):

  /            time (time), coord (time, 2)  - top-left (i, j) of each sample in the invariant grid
  /input       one variable per channel, (time, y, x) on the coarse grid
  /output      one variable per channel, (time, Y, X) with Y = f*y, X = f*x (integer factor f)
  /invariant   one variable per channel, (Y, X) e.g. orography, land-sea mask

The reader upsamples the input by f with bilinear extrapolation, so the coarse grid must cover exactly
the fine domain (cell-centred, same corners).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def check_pair(inp: xr.Dataset, out: xr.Dataset, ydim="y", xdim="x") -> int:
    """Return the integer upsampling factor or raise."""
    if not np.array_equal(inp["time"].values, out["time"].values):
        raise ValueError("input and output must have the same times in the same order")
    ny, nx = inp.sizes[ydim], inp.sizes[xdim]
    NY, NX = out.sizes[ydim], out.sizes[xdim]
    if NY % ny or NX % nx or NY // ny != NX // nx:
        raise ValueError(f"output {NY}x{NX} is not an integer multiple of input {ny}x{nx} with the same factor")
    return NY // ny


def write_hrrrmini(inp: xr.Dataset, out: xr.Dataset, path: str | Path, invariants: xr.Dataset | None = None,
                   ydim="y", xdim="x") -> Path:
    f = check_pair(inp, out, ydim, xdim)
    NY, NX = out.sizes[ydim], out.sizes[xdim]
    for name, ds in (("input", inp), ("output", out)):
        bad = [v for v in ds.data_vars if not np.isfinite(ds[v].values).all()]
        if bad:
            raise ValueError(f"{name}: non-finite values in {', '.join(bad)}")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    root = xr.Dataset({"coord": (("time", "c"), np.zeros((inp.sizes["time"], 2), dtype=np.int64))},
                      coords={"time": inp["time"].values},
                      attrs={"layout": "PhysicsNeMo CorrDiff hrrr_mini", "upsample_factor": f,
                             "written_by": "kcadapt"})
    root.to_netcdf(p, mode="w")

    def _group(ds, dims):
        g = xr.Dataset({v: (dims, ds[v].transpose(*dims).values.astype("float32")) for v in ds.data_vars})
        return g

    _group(inp, ("time", ydim, xdim)).to_netcdf(p, mode="a", group="input")
    _group(out, ("time", ydim, xdim)).to_netcdf(p, mode="a", group="output")
    if invariants is None:
        invariants = xr.Dataset({"zeros": ((ydim, xdim), np.zeros((NY, NX), "float32"))})
    if any(invariants.sizes[d] != s for d, s in ((ydim, NY), (xdim, NX))):
        raise ValueError("invariants must be on the output grid")
    _group(invariants, (ydim, xdim)).to_netcdf(p, mode="a", group="invariant")
    return p


def read_group(path: str | Path, group: str) -> xr.Dataset:
    return xr.load_dataset(path, group=group)

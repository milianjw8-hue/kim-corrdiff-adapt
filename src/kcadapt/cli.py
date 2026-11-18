"""Command line interface.

  kcadapt shift      --src kim.nc --ref era5.nc [--stats stats.json] --out shift.csv
  kcadapt qmap-fit   --src kim.nc --ref era5.nc --out qmap.nc [--by season] [--pooled] [--test-years 2024]
  kcadapt qmap-apply --qmap qmap.nc --src kim.nc --out kim_qm.nc
  kcadapt stats      --input era5.nc --output target.nc [--invariant inv.nc] --out stats.json
  kcadapt renorm     --stats stats.json --kim kim.nc --out stats_kim.json
  kcadapt export     --input kim.nc --output target.nc [--invariant inv.nc] --out pairs.nc
  kcadapt arms       --setup setup.yaml --out configs/
  kcadapt demo       --out demo/         (synthetic end-to-end, no GPU)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr

from . import arms, export, stats
from .qmap import QuantileMap, split_by_year
from .shift import bias_maps, common_times, shift_table

pd.set_option("display.width", 160)


def _open(p):
    return xr.load_dataset(p)


def cmd_shift(a):
    src, ref = common_times(_open(a.src), _open(a.ref))
    st = stats.load(a.stats) if a.stats else None
    t = shift_table(src, ref, a.variables, st)
    t.to_csv(a.out, index=False)
    if a.maps:
        bias_maps(src, ref, a.variables).to_netcdf(a.maps)
    print(t.round(3).to_string(index=False))


def cmd_qmap_fit(a):
    src, ref = _open(a.src), _open(a.ref)
    by = None if a.by == "none" else a.by
    if a.test_years:
        s_tr, s_te = split_by_year(src, a.test_years)
        r_tr, r_te = split_by_year(ref, a.test_years)
    else:
        s_tr, r_tr, s_te, r_te = src, ref, None, None
    qm = QuantileMap(a.variables, pointwise=not a.pooled, by=by).fit(s_tr, r_tr)
    qm.save(a.out)
    print(f"mapping saved to {a.out} (fitted on {qm.q.attrs['n_times']} times)")
    if s_te is not None:
        s_te, r_te = common_times(s_te, r_te)
        before = shift_table(s_te, r_te, qm.variables).assign(stage="before")
        after = shift_table(qm.transform(s_te), r_te, qm.variables).assign(stage="after")
        t = pd.concat([before, after])
        t.to_csv(Path(a.out).with_suffix(".test.csv"), index=False)
        print("held-out years:", ", ".join(map(str, a.test_years)))
        print(t[["stage", "variable", "mean_bias", "std_ratio", "w1_norm", "needs_correction"]].round(3)
              .to_string(index=False))


def cmd_qmap_apply(a):
    qm = QuantileMap.load(a.qmap)
    qm.transform(_open(a.src)).to_netcdf(a.out)
    print(f"saved {a.out}")


def cmd_stats(a):
    s = stats.build(_open(a.input), _open(a.output), _open(a.invariant) if a.invariant else None)
    stats.save(s, a.out)
    print(f"saved {a.out}")


def cmd_renorm(a):
    s = stats.load(a.stats)
    new = stats.replace_input(s, stats.compute(_open(a.kim), a.variables), a.variables)
    stats.save(new, a.out)
    print(f"saved {a.out} (input statistics from {a.kim}; output statistics unchanged)")


def cmd_export(a):
    p = export.write_hrrrmini(_open(a.input), _open(a.output), a.out, _open(a.invariant) if a.invariant else None)
    print(f"saved {p}")


def cmd_arms(a):
    for p in arms.write_all(arms.load_setup(a.setup), a.out):
        print(f"wrote {p}")


def cmd_demo(a):
    from .demo import run_demo

    run_demo(Path(a.out))


def main(argv=None):
    p = argparse.ArgumentParser(prog="kcadapt")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("shift", help="input distribution shift, KIM vs ERA5")
    s.add_argument("--src", required=True)
    s.add_argument("--ref", required=True)
    s.add_argument("--stats", help="CorrDiff stats JSON of the pretrained model")
    s.add_argument("--variables", nargs="+")
    s.add_argument("--maps", help="NetCDF of time-mean bias and std-ratio maps")
    s.add_argument("--out", default="shift.csv")
    s.set_defaults(func=cmd_shift)

    q = sub.add_parser("qmap-fit", help="fit quantile mapping KIM -> ERA5")
    q.add_argument("--src", required=True)
    q.add_argument("--ref", required=True)
    q.add_argument("--variables", nargs="+")
    q.add_argument("--by", default="season", choices=["season", "month", "none"])
    q.add_argument("--pooled", action="store_true", help="one mapping for the whole domain")
    q.add_argument("--test-years", nargs="+", type=int, help="hold out these years and report before/after")
    q.add_argument("--out", default="qmap.nc")
    q.set_defaults(func=cmd_qmap_fit)

    qa = sub.add_parser("qmap-apply")
    qa.add_argument("--qmap", required=True)
    qa.add_argument("--src", required=True)
    qa.add_argument("--out", required=True)
    qa.set_defaults(func=cmd_qmap_apply)

    st = sub.add_parser("stats", help="CorrDiff stats JSON from data")
    st.add_argument("--input", required=True)
    st.add_argument("--output", required=True)
    st.add_argument("--invariant")
    st.add_argument("--out", default="stats.json")
    st.set_defaults(func=cmd_stats)

    rn = sub.add_parser("renorm", help="replace input statistics with KIM's (arm C)")
    rn.add_argument("--stats", required=True)
    rn.add_argument("--kim", required=True)
    rn.add_argument("--variables", nargs="+")
    rn.add_argument("--out", default="stats_kim.json")
    rn.set_defaults(func=cmd_renorm)

    ex = sub.add_parser("export", help="write pairs in the CorrDiff hrrr_mini layout")
    ex.add_argument("--input", required=True)
    ex.add_argument("--output", required=True)
    ex.add_argument("--invariant")
    ex.add_argument("--out", required=True)
    ex.set_defaults(func=cmd_export)

    ar = sub.add_parser("arms", help="write PhysicsNeMo configs for the experiment arms")
    ar.add_argument("--setup", required=True)
    ar.add_argument("--out", default="configs/generated")
    ar.set_defaults(func=cmd_arms)

    d = sub.add_parser("demo", help="synthetic end-to-end run")
    d.add_argument("--out", default="demo")
    d.set_defaults(func=cmd_demo)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()

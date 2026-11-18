"""Synthetic end-to-end run: diagnose the shift, fit and test quantile mapping, build statistics,
export the data and write the experiment configs. Nothing produced here is real data."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import arms, export, stats  # noqa: E402
from .qmap import PROBS, QuantileMap, split_by_year  # noqa: E402
from .shift import shift_table  # noqa: E402
from .synthetic import make  # noqa: E402


def qq_figure(src, ref, mapped, variables, path):
    fig, axes = plt.subplots(1, len(variables), figsize=(3.2 * len(variables), 3.2))
    for ax, v in zip(np.atleast_1d(axes), variables):
        qr = np.quantile(ref[v].values, PROBS)
        ax.plot(qr, np.quantile(src[v].values, PROBS), label="KIM (raw)")
        ax.plot(qr, np.quantile(mapped[v].values, PROBS), label="KIM (mapped)")
        lo, hi = qr.min(), qr.max()
        ax.plot([lo, hi], [lo, hi], "k:", lw=1)
        ax.set_title(v)
        ax.set_xlabel("ERA5 quantile")
    np.atleast_1d(axes)[0].set_ylabel("KIM quantile")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("Held-out year, pooled quantiles (SYNTHETIC)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_demo(out: Path) -> pd.DataFrame:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    era5, kim, target, inv = make()
    variables = ["u10m", "v10m", "t2m", "msl"]
    test_year = int(era5.time.dt.year.max())

    # statistics of the "pretrained" model: ERA5 inputs of the training years
    e_tr, e_te = split_by_year(era5, [test_year])
    k_tr, k_te = split_by_year(kim, [test_year])
    t_tr, t_te = split_by_year(target, [test_year])
    st = stats.build(e_tr, t_tr, inv)
    stats.save(st, out / "stats_era5.json")

    qm = QuantileMap(variables, pointwise=True, by="season").fit(k_tr, e_tr)
    qm.save(out / "qmap.nc")
    k_qm = qm.transform(k_te)

    before = shift_table(k_te, e_te, variables, st).assign(stage="before")
    after = shift_table(k_qm, e_te, variables, st).assign(stage="after")
    table = pd.concat([before, after], ignore_index=True)
    table.to_csv(out / "shift_test_year.csv", index=False)
    qq_figure(k_te, e_te, k_qm, variables, out / "qq_test_year.png")

    stats.save(stats.replace_input(st, stats.compute(k_tr, variables), variables), out / "stats_kim.json")

    out_vars = ["u10m", "v10m", "t2m"]
    for arm, ds in {"R": e_te, "A": k_te, "B": k_qm}.items():
        export.write_hrrrmini(ds[variables], t_te[out_vars], out / f"pairs_arm{arm}.nc", inv)
    export.write_hrrrmini(k_tr[variables], t_tr[out_vars], out / "pairs_finetune.nc", inv)

    setup = arms.Setup(input_variables=variables, output_variables=out_vars, invariant_variables=["orog", "lsm"],
                       stats=str(out / "stats_era5.json"), stats_kim=str(out / "stats_kim.json"),
                       reg_ckpt="checkpoints/regression.mdlus", res_ckpt="checkpoints/diffusion.mdlus",
                       data={"R": str(out / "pairs_armR.nc"), "A": str(out / "pairs_armA.nc"),
                             "B": str(out / "pairs_armB.nc"), "C": str(out / "pairs_armA.nc")},
                       times=[str(t)[:19] for t in k_te.time.values[:4]],
                       finetune_data=str(out / "pairs_finetune.nc"), pretrained_nimg=10_000_000)
    for p in arms.write_all(setup, out / "configs"):
        print(f"wrote {p}")
    show = table[["stage", "variable", "mean_bias", "std_ratio", "w1_norm", "z_shift", "needs_correction"]]
    print(show.round(3).to_string(index=False))
    print(f"demo written to {out} (SYNTHETIC data)")
    return table

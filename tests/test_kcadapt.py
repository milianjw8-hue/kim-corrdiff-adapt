import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from kcadapt import arms, export, stats
from kcadapt.cli import main
from kcadapt.qmap import QuantileMap, split_by_year
from kcadapt.shift import shift_table
from kcadapt.synthetic import make


@pytest.fixture(scope="module")
def data():
    return make(n_years=3, ny=8, nx=8)


def test_shift_detects_planted_differences(data):
    era5, kim, _, _ = data
    t = shift_table(kim, era5).set_index("variable")
    assert t.loc["u10m", "std_ratio"] > 1.15 and t.loc["u10m", "needs_correction"]
    assert t.loc["t2m", "mean_bias"] < -0.3
    same = shift_table(era5, era5).set_index("variable")
    assert np.allclose(same["w1"], 0) and not same["needs_correction"].any()


def test_qmap_identity_when_same(data):
    era5 = data[0][["t2m"]]
    m = QuantileMap(["t2m"], pointwise=False, by=None).fit(era5, era5)
    y = m.transform(era5)
    inner = (era5.t2m > era5.t2m.quantile(0.02)) & (era5.t2m < era5.t2m.quantile(0.98))
    assert float(np.abs(y.t2m - era5.t2m).where(inner).max()) < 1e-3


@pytest.mark.parametrize("pointwise", [True, False])
def test_qmap_reduces_shift_on_held_out_year(data, pointwise):
    era5, kim, _, _ = data
    yr = int(era5.time.dt.year.max())
    e_tr, e_te = split_by_year(era5, [yr])
    k_tr, k_te = split_by_year(kim, [yr])
    m = QuantileMap(["u10m", "t2m"], pointwise=pointwise, by="season").fit(k_tr, e_tr)
    before = shift_table(k_te, e_te, ["u10m", "t2m"]).set_index("variable")
    after = shift_table(m.transform(k_te), e_te, ["u10m", "t2m"]).set_index("variable")
    assert (after["w1_norm"] < 0.3 * before["w1_norm"]).all()
    assert abs(after.loc["u10m", "std_ratio"] - 1) < 0.05


def test_multiplicative_stays_non_negative():
    t = pd.date_range("2024-01-01", periods=400, freq="6h")
    rng = np.random.default_rng(0)
    ref = xr.Dataset({"tp": (("time", "y"), rng.gamma(0.5, 2.0, (400, 3)))}, coords={"time": t})
    src = xr.Dataset({"tp": (("time", "y"), 1.5 * rng.gamma(0.5, 2.0, (400, 3)))}, coords={"time": t})
    m = QuantileMap(["tp"], pointwise=False, by=None).fit(src, ref)
    y = m.transform(src)
    assert float(y.tp.min()) >= 0 and abs(float(y.tp.mean()) / float(ref.tp.mean()) - 1) < 0.1


def test_qmap_save_load_roundtrip(tmp_path, data):
    era5, kim, _, _ = data
    m = QuantileMap(["v10m"], pointwise=True, by="season").fit(kim, era5)
    m.save(tmp_path / "q.nc")
    m2 = QuantileMap.load(tmp_path / "q.nc")
    sub = kim.isel(time=slice(0, 50))
    assert np.allclose(m.transform(sub).v10m, m2.transform(sub).v10m)


def test_too_short_group_rejected(data):
    era5, kim, _, _ = data
    with pytest.raises(ValueError, match="group"):
        QuantileMap(["t2m"], by="month").fit(kim.isel(time=slice(0, 200)), era5.isel(time=slice(0, 200)))


def test_stats_format_and_renorm(tmp_path, data):
    era5, kim, target, inv = data
    s = stats.build(era5, target, inv)
    p = stats.save(s, tmp_path / "s.json")
    d = json.loads(p.read_text())
    assert set(d) == {"input", "output", "invariant"} and set(d["input"]["t2m"]) == {"mean", "std"}
    new = stats.replace_input(d, stats.compute(kim, ["u10m"]), ["u10m"])
    assert new["input"]["u10m"] != d["input"]["u10m"]
    assert new["input"]["t2m"] == d["input"]["t2m"] and new["output"] == d["output"]
    assert stats.check_compatible(d, ["u10m", "zz"], ["t2m"]) == ["input: missing zz"]


def _hrrrmini_load(path, group, variables, stack_axis=1):
    """Same logic as PhysicsNeMo corrdiff datasets/hrrrmini.py::_load_dataset."""
    with xr.open_dataset(path, group=group) as ds:
        return np.stack([ds[v] for v in variables], axis=stack_axis)


def test_export_matches_corrdiff_reader(tmp_path, data):
    era5, _, target, inv = data
    e = era5.isel(time=slice(0, 10))
    t = target.isel(time=slice(0, 10))
    p = export.write_hrrrmini(e, t[["u10m", "t2m"]], tmp_path / "pairs.nc", inv)
    x = _hrrrmini_load(p, "input", ["u10m", "v10m"])
    y = _hrrrmini_load(p, "output", ["u10m", "t2m"])
    z = _hrrrmini_load(p, "invariant", ["orog", "lsm"], stack_axis=0)
    assert x.shape == (10, 2, 8, 8) and y.shape == (10, 2, 32, 32) and z.shape == (2, 32, 32)
    assert y.shape[-1] // x.shape[-1] == 4
    with xr.open_dataset(p) as root:
        assert root["coord"].shape == (10, 2) and root.attrs["upsample_factor"] == 4
    with pytest.raises(ValueError, match="multiple"):
        export.write_hrrrmini(e, t.isel(x=slice(0, 30)), tmp_path / "bad.nc")


def test_arms_configs(tmp_path):
    s = arms.Setup(input_variables=["u10m"], output_variables=["u10m"], invariant_variables=["orog"],
                   stats="s.json", stats_kim="sk.json", reg_ckpt="reg.mdlus", res_ckpt="res.mdlus",
                   data={"R": "r.nc", "A": "a.nc", "B": "b.nc", "C": "a.nc", "D": "a.nc"},
                   times=["2024-07-01T00:00:00"], finetune_data="ft.nc", pretrained_nimg=1000)
    paths = arms.write_all(s, tmp_path)
    names = {p.name for p in paths}
    assert {"config_generate_armA.yaml", "config_generate_armC.yaml", "config_training_armD_regression.yaml"} <= names
    c = yaml.safe_load((tmp_path / "config_generate_armC.yaml").read_text())
    assert c["dataset"]["stats_path"] == "sk.json" and c["dataset"]["type"] == "hrrr_mini"
    assert c["generation"]["io"]["reg_ckpt_filename"] == "reg.mdlus"
    tr = yaml.safe_load((tmp_path / "config_training_armD_regression.yaml").read_text())
    assert tr["training"]["hp"]["training_duration"] == 1000 + s.finetune_nimg
    assert {"model": "regression"} in tr["defaults"]
    with pytest.raises(ValueError, match="reg_ckpt_finetuned"):
        arms.generation_config(s, "D")


def test_cli_shift_and_qmap(tmp_path, data):
    era5, kim, _, _ = data
    era5.to_netcdf(tmp_path / "e.nc")
    kim.to_netcdf(tmp_path / "k.nc")
    main(["shift", "--src", str(tmp_path / "k.nc"), "--ref", str(tmp_path / "e.nc"), "--out", str(tmp_path / "s.csv"),
          "--maps", str(tmp_path / "m.nc")])
    assert pd.read_csv(tmp_path / "s.csv")["needs_correction"].any()
    assert "u10m_std_ratio" in xr.open_dataset(tmp_path / "m.nc")
    yr = int(era5.time.dt.year.max())
    main(["qmap-fit", "--src", str(tmp_path / "k.nc"), "--ref", str(tmp_path / "e.nc"), "--out",
          str(tmp_path / "q.nc"), "--pooled", "--test-years", str(yr)])
    t = pd.read_csv(tmp_path / "q.test.csv")
    assert set(t["stage"]) == {"before", "after"}
    main(["qmap-apply", "--qmap", str(tmp_path / "q.nc"), "--src", str(tmp_path / "k.nc"),
          "--out", str(tmp_path / "kq.nc")])
    assert xr.open_dataset(tmp_path / "kq.nc")["u10m"].attrs["quantile_mapped"] == "kcadapt"


def test_demo(tmp_path):
    table = main(["demo", "--out", str(tmp_path / "demo")])
    assert (tmp_path / "demo" / "configs" / "config_generate_armB.yaml").exists()
    t = pd.read_csv(tmp_path / "demo" / "shift_test_year.csv")
    assert t[t.stage == "after"]["w1_norm"].max() < t[t.stage == "before"]["w1_norm"].max()

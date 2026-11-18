"""Experiment arms for moving a pretrained CorrDiff from ERA5-like inputs to KIM inputs, written as
Hydra config files for PhysicsNeMo ``examples/weather/corrdiff`` (generate.py / train.py).

  A  raw       KIM inputs, original statistics, original weights          (no adaptation; baseline)
  B  qmap      KIM inputs quantile-mapped to ERA5, original statistics      (input correction only)
  C  renorm    KIM inputs, input statistics recomputed from KIM            (re-centring only)
  D  finetune  KIM inputs, original statistics, regression UNet fine-tuned on (KIM, target) pairs,
               diffusion (residual) network unchanged
  R  era5      ERA5 inputs, original everything                           (reference: what the model
                                                                             does with its own inputs)

Config keys follow the PhysicsNeMo commit checked on 2026-11-18 (conf/config_generate_custom.yaml,
conf/config_training_custom.yaml, conf/base/dataset/hrrr_mini.yaml). Check them against your
PhysicsNeMo version.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ARMS = {
    "R": "ERA5 inputs (reference)",
    "A": "KIM raw",
    "B": "KIM quantile-mapped to ERA5",
    "C": "KIM with KIM input statistics",
    "D": "KIM with fine-tuned regression UNet",
}


@dataclass
class Setup:
    input_variables: list[str]
    output_variables: list[str]
    invariant_variables: list[str]
    stats: str                       # original stats JSON (from ERA5-input training)
    reg_ckpt: str                    # pretrained regression checkpoint (.mdlus)
    res_ckpt: str                    # pretrained diffusion (residual) checkpoint (.mdlus)
    data: dict[str, str]             # arm -> data file (hrrr_mini layout) for generation
    times: list[str]                 # valid times to generate, ISO
    stats_kim: str = ""              # KIM input statistics (arm C)
    finetune_data: str = ""          # (KIM input, target) training pairs for arm D
    finetune_ckpt_dir: str = "checkpoints_finetune"
    reg_ckpt_finetuned: str = ""     # produced by arm D training
    pretrained_nimg: int = 0         # samples the pretrained regression net has seen (train.py resumes from it)
    finetune_nimg: int = 2_000_000   # extra samples for fine-tuning (assumption: tune on validation loss)
    finetune_lr: float = 2e-5        # 10x below the default 2e-4 (assumption)
    num_ensembles: int = 32
    patched: bool = False
    patch_shape: tuple[int, int] = (448, 448)
    extra: dict = field(default_factory=dict)


def _dataset(s: Setup, data: str, stats: str) -> dict:
    return {"type": "hrrr_mini", "data_path": data, "stats_path": stats,
            "input_variables": list(s.input_variables), "output_variables": list(s.output_variables),
            "invariant_variables": list(s.invariant_variables)}


def generation_config(s: Setup, arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    stats = s.stats_kim if arm == "C" else s.stats
    if arm == "C" and not stats:
        raise ValueError("arm C needs stats_kim")
    reg = s.reg_ckpt_finetuned if arm == "D" else s.reg_ckpt
    if arm == "D" and not reg:
        raise ValueError("arm D needs reg_ckpt_finetuned (train it first with training_config)")
    gen = {"num_ensembles": s.num_ensembles, "seed_batch_size": 4, "times": list(s.times),
           "io": {"res_ckpt_filename": s.res_ckpt, "reg_ckpt_filename": reg}}
    if s.patched:
        gen["patch_shape_x"], gen["patch_shape_y"] = s.patch_shape
    return {
        "hydra": {"job": {"chdir": False, "name": f"generate_arm{arm}"},
                  "run": {"dir": f"./outputs/${{hydra:job.name}}"},
                  "searchpath": ["pkg://conf/base"]},
        "defaults": [{"dataset": "hrrr_mini"}, {"generation": "patched" if s.patched else "non_patched"}],
        "dataset": _dataset(s, s.data[arm], stats),
        "generation": gen,
        "wandb": {"mode": "offline"},
        "kcadapt": {"arm": arm, "description": ARMS[arm]},
    }


def training_config(s: Setup) -> dict:
    """Arm D: continue training the regression UNet on KIM pairs.

    PhysicsNeMo train.py resumes from the newest checkpoint in the checkpoint directory (weights,
    optimizer and sample counter). Copy the pretrained regression checkpoint there first; training then
    runs until training_duration = pretrained_nimg + finetune_nimg samples (확인 필요 for your version).
    """
    if not s.finetune_data:
        raise ValueError("arm D needs finetune_data")
    return {
        "hydra": {"job": {"chdir": False, "name": "finetune_regression_kim"},
                  "run": {"dir": "./outputs/${hydra:job.name}"},
                  "searchpath": ["pkg://conf/base"]},
        "defaults": [{"dataset": "hrrr_mini"}, {"model": "regression"}, {"model_size": "normal"},
                     {"training": "${model}"}],
        "dataset": _dataset(s, s.finetune_data, s.stats),
        "training": {"hp": {"training_duration": int(s.pretrained_nimg + s.finetune_nimg), "lr": s.finetune_lr,
                            "lr_rampup": 0},
                     "io": {"checkpoint_dir": s.finetune_ckpt_dir}},
        "wandb": {"mode": "offline"},
        "kcadapt": {"arm": "D", "pretrained_regression": s.reg_ckpt,
                    "note": "copy pretrained_regression into training.io.checkpoint_dir before running train.py"},
    }


def write_all(s: Setup, out_dir: str | Path, arms=("R", "A", "B", "C")) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for a in arms:
        if a not in s.data:
            continue
        if a == "C" and not s.stats_kim:
            continue
        p = out / f"config_generate_arm{a}.yaml"
        p.write_text(yaml.safe_dump(generation_config(s, a), sort_keys=False, allow_unicode=True), encoding="utf-8")
        paths.append(p)
    if s.finetune_data:
        p = out / "config_training_armD_regression.yaml"
        p.write_text(yaml.safe_dump(training_config(s), sort_keys=False, allow_unicode=True), encoding="utf-8")
        paths.append(p)
    if s.reg_ckpt_finetuned and "D" in s.data:
        p = out / "config_generate_armD.yaml"
        p.write_text(yaml.safe_dump(generation_config(s, "D"), sort_keys=False, allow_unicode=True), encoding="utf-8")
        paths.append(p)
    return paths


def load_setup(path: str | Path) -> Setup:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if "patch_shape" in d:
        d["patch_shape"] = tuple(d["patch_shape"])
    return Setup(**d)

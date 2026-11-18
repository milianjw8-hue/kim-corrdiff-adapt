# Changelog

## 0.1.0 (2026-11-18)

- Input-shift diagnosis KIM vs ERA5: mean bias, standard-deviation ratio, 1-Wasserstein distance
  (normalised), shift against the CorrDiff training statistics; bias and std-ratio maps
- Empirical quantile mapping (pointwise or pooled, by season or month, additive or multiplicative),
  held-out-year evaluation, save/load
- CorrDiff statistics JSON (build, replace input statistics, compatibility check)
- Export of (coarse input, fine target, invariants) pairs in the PhysicsNeMo CorrDiff `hrrr_mini` layout
- PhysicsNeMo Hydra configs for experiment arms R, A, B, C, D
- Synthetic data, demo and tests (no GPU)

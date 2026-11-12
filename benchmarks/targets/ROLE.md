# `benchmarks/targets`

Canonical synthetic objects with **shared physical ground truth** for all
configurations A/B/C/D:

- transparent, blocked
- point, narrow slit
- vertical / horizontal knife edge
- checkerboard / calibrated bars
- amplitude step, phase step, phase ramp

Do not fork incompatible per-config object definitions. Rebuild from
`PHYSICAL_TRUTH` onto the desired `(n, pixel_um)` sampling instead.

Paper artifacts go under `evidence/taskXX_<name>/` via `output_layout`.

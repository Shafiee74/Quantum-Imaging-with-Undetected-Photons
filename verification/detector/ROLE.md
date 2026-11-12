# `verification/detector`

U12 validation of the **existing** `detector.py` EMCCD chain (Task 12).

Controlled isolations: ideal / shot / read / dark / CIC / EM gain / combined.
Analytical mean/variance checks only — no second detector model and no
experimental-image tuning.

Paper artifacts go under `evidence/taskXX_<name>/` via `output_layout`.

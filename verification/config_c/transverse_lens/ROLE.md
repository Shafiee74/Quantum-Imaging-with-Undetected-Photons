# `verification/config_c/transverse_lens`

Independent derivation and reference implementation of the **Configuration C**
far-field transverse idler-lens (lateral shift Δx/Δy) phase law.

**Critical rule:** `derivation.py` / `reference.py` must not import production
`optics`, `setup_params`, or `simulate`. Comparison against production happens
only in `compare.py` / `runner.py`.

Created by Task 17. Evidence artifacts go to `evidence/task17_transverse_lens/`.

# `validation/config_c/c1`

C1 axial idler-mirror phase-step evaluation (held-out predictions).

**Task 18** implements identification → no-object calibration → freeze →
with-object evaluation. Experimental TIFF tree is **READ-ONLY**.

Stage-position (µm) values are **not** encoded in filenames. Folder tokens
(`0`, `pi2`, `pi`, `3pi2`, `2pi`, `dark`, `ref`) are used as acquisition
labels. Any µm assignment is taken only from the Configuration-C freeze plan
experimenter report (25 µm increments ≈ quarter-wave sequence) and is labeled
with explicit provenance — never the ideal geometric mirror law.

Evidence: `evidence/task18_c1_validation/`.

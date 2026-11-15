# Figure 6(a) — public reproducibility package

## What is shipped (non-restricted)

| File | Role |
|---|---|
| `Experimental_Frequency_Comparison.csv` | Derived experimental fringe-center frequencies used by the manuscript plot |
| `Figure6a_Published_Frequencies.csv` | Locked simulation/experiment plotted frequencies (MAE/RMSE check) |
| `README.md` | This note |

These CSVs are **derived analysis records**, not raw EMCCD frames.

## What is not shipped

Original laboratory EMCCD TIFF stacks under `exp data/` remain private by agreement with the experimental team. They are listed in `.gitignore` and are **not** required for the public numerical Figure 6(a) frequency curve.

Controlled access: contact the corresponding author / laboratory PI. Repository maintainers do not redistribute the raw TIFF archive.

## Public clean-run command

From the repository root:

```bash
python tools/reproduce_figure6a.py --public
```

This mode:

1. Regenerates the **simulation** image strip from production `optics.py` + `C_lab`.
2. Loads experimental frequencies from `reproducibility/figure6a/Experimental_Frequency_Comparison.csv`.
3. Plots the frequency panel with a **hollow circular marker** at the analytical 20 µm simulation point (image-derived points remain filled squares).
4. Writes outputs under `reproducibility/figure6a/out/` (not the gitignored `article figures/` tree).

Optional laboratory mode (requires private TIFFs + local archive baseline):

```bash
python tools/reproduce_figure6a.py --laboratory
```

## Expected numerical summary

Manuscript / reviewer convention: MAE/RMSE over **image-derived** nonzero points
(30, 40, 60, 180 µm), **excluding** the analytical 20 µm simulation marker.

- MAE ≈ **0.05215** cycles/mm  
- RMSE ≈ **0.05465** cycles/mm  

(Matches the supervisor recalculation quoted in the review.)

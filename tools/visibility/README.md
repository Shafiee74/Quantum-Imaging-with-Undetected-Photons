# C_lab visibility analysis (`tools.visibility`)

Phase-resolved visibility for **experimental RAW TIFF** series and **simulated** intensity stacks.

## Public imports

```python
from tools.visibility import (
    phase_from_mirror_um,
    fit_integrated,
    fit_pixelwise,
    run_experiment,
    run_simulated_stack,
    moments_to_phase_stack,
    figure5_four_step_map,
    BackgroundMode,
)
```

## Metrics (always label separately)

| Key | Meaning |
|-----|---------|
| `visibility_raw` / `visibility_raw_unweighted` | Integrated sinusoid LS, **unweighted** (canonical) |
| `visibility_weighted_sem` | Inverse-variance weighted LS (legacy pilot style) |
| `pixelwise_mean_raw` | Mean of per-pixel fitted V in ROI |
| `visibility_from_summed_coefficients` | √(B²+C²)/A from summed pixel coefficients |
| `statistically_valid` | `False` if residuals/χ² fail — **not** a validated measurement |

**Figure 5:** exact four-step quadrature remains `tools.z1.estimator.compute_four_step_visibility`.
Use `figure5_four_step_map` only for exact 0, π/2, π, 3π/2 frames.

## CLI

```powershell
python -m tools.visibility.c_lab_shared `
  --experimental-root "exp data/mirror drift" `
  --roi "evidence/c_lab_a6_12e_figure5_visibility_recovery/20261015T082901Z/FIGURE5_ROI_90PCT.npy" `
  --positions "tools/visibility/data/mirror_drift_5phase_um.json" `
  --output "evidence/my_visibility_run"
```

## Background modes

Default: `BackgroundMode.RAW_NO_DARK`. Future calibrated bias uses
`prepare_intensity_frame(..., mode=BackgroundMode.CALIBRATED_BIAS, bias_adu=map)`
when a validated bias map exists (no artificial offsets).

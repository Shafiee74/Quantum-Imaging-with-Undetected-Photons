# Figure 6(a) — root-cause analysis and validation (Comment A2)

## A. Root cause

The active public path in `tools/reproduce_figure6a.py` previously contained an
`enforce_published=True` fallback inside `run_public()` / `build_rows()`.

When independently computed simulation frequencies differed from a hard-coded
`PUBLISHED` table (or when a rounding mismatch was detected), the code
**silently replaced calculated values with manuscript numbers** before plotting.

That violates scientific reproducibility: the figure could look correct without
proving that the physical model produced the submitted frequencies.

Additional issues:

- Top-level `Experimental_Frequency_Comparison.csv` used incorrect experimental
  center counts (conflating historical **simulated** counts 5/7/21 with Table S3).
- Reference CSV/JSON lived mixed with generated outputs, blurring compute vs validate.

## B. Fix

1. Removed all manuscript-frequency overwrite / `enforce_published` behavior.
2. Split pipeline: `--compute` (model only) → `--verify` (read-only comparison).
3. Moved validation targets to `reproducibility/figure6a/reference/`.
4. Write newly calculated rows only to `reproducibility/figure6a/computed/`.
5. Corrected derived experimental Table S3 center counts (1/2/3/4/5/16).
6. Documented historical obsolete tables under `historical/`.

## C. Independent reproduction status

| Quantity | Status |
|---|---|
| Simulated images (C_lab, production optics) | **Independently reproduced** |
| Simulated centers 1/1/3/3/3/11 | **Independently reproduced** |
| Simulated frequencies (incl. analytical 20 µm) | **Independently reproduced** |
| Experimental frequencies | **Derived public record** (Table S3); not re-extracted from TIFFs |
| Experimental image strip | **Not reproduced publicly** (original EMCCD TIFFs not public) |

## D. Numerical validation (compute vs reference)

Verification report: `Figure6a_Validation_Report.md` / `.json`

- Max |Δk_sim| vs reference Final_Data: **≈ 4.3×10⁻¹⁰** cycles/mm → **PASS**
- Center counts exact match → **PASS**
- MAE (30/40/60/180 vs derived experiment) ≈ **0.05215** → **PASS**
- RMSE ≈ **0.05465** → **PASS**

## E. Commands

```bash
python tools/reproduce_figure6a.py --compute
python tools/reproduce_figure6a.py --verify
```

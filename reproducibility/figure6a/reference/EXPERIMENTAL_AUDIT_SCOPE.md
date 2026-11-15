# Experimental center audit scope (Table S3 / Word §4)

## What is public

`Experimental_Frequency_Comparison.csv` is the **derived** Supplementary Table S3
record used by the public Figure 6(a) pipeline:

| Δx (µm) | Accepted experimental centers | Frequency (cycles/mm) |
|---:|---:|---:|
| 0 | 1 | 0 (aligned-reference convention) |
| 20 | 2 | 0.429938017 |
| 30 | 3 | 0.549472354 |
| 40 | 4 | 0.785463685 |
| 60 | 5 | 0.968635059 |
| 180 | 16 | 3.043021563 |

These are **not** historical simulated counts (5/7/21).

## Fixed analysis settings (laboratory re-extraction)

When original EMCCD TIFFs are available locally, the documented algorithm is:

- Group TIFFs by stage position; remove duplicate SHA-256 hashes
- Average distinct 411×453 frames (no dark subtraction / crop / per-frame norm)
- 12% illumination mask; FFT fringe-normal; 720-bin directional profile
- Adaptive smoothing σ = min(6, max(1, P/8)) pixels
- Prominence ≥ max(0.06 × range, 0.02 × std); min separation 0.45 × broad period
- Empty bins interpolated only for spacing on the uniform grid; accepted centers
  must belong to measured bins
- Physical pixel pitch 13 µm

Entry point (private data required):
`python tools/figure6a_experimental_fringe_center_correction.py`

## What cannot be shipped publicly

Full accepted/rejected candidate-audit CSV rows from the original EMCCD TIFF
recordings are **not** in the public repository because the original TIFF
images themselves are restricted. Public numerical reproducibility therefore
uses the derived Table S3 frequencies above, clearly labelled as a derived
record rather than a fresh TIFF re-extraction.

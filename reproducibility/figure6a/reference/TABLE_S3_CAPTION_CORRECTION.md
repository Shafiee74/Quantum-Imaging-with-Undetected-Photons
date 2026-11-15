# Supplementary Table S3 — corrected data-availability caption

**Use this wording in the Supplementary Information** (replaces any caption that
claims full-precision fringe-center positions, individual spacings, or
peak-selection audit records are publicly available on GitHub).

## Corrected caption (recommended)

> **Table S3.** Derived experimental fringe-frequency summary for the transverse
> idler-lens displacements used in Figure 6(a). Reported quantities are the
> accepted bright-fringe center counts and the corresponding image-derived
> spatial frequencies (cycles/mm) obtained with the fixed laboratory analysis
> settings documented in Section S5.6. These **derived** numerical records are
> publicly available in the companion GitHub repository under
> `reproducibility/figure6a/reference/Experimental_Frequency_Comparison.csv`.
> The original experimental EMCCD TIFF recordings, and the associated
> full-precision fringe-center positions, individual adjacent spacings, and
> accepted/rejected peak-selection audit tables, are **not** publicly available.

## What the public repository actually ships

| Record | Public? | Path |
|---|---|---|
| Accepted-center counts + plotted frequencies | Yes | `reproducibility/figure6a/reference/Experimental_Frequency_Comparison.csv` |
| Analysis-settings documentation | Yes | `reproducibility/figure6a/reference/EXPERIMENTAL_AUDIT_SCOPE.md` |
| Full-precision crest positions / spacing audits | No | Requires restricted EMCCD TIFFs |
| Original EMCCD TIFF stacks | No | Restricted (see `DATA_AVAILABILITY.md`) |

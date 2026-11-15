# Data and code availability (Comment A / Technologies)

**Package:** Physics-Informed Reduced-Order SPDC-QIUP Forward Simulator Prototype  
**Version:** 3.4.2  
**Repository:** https://github.com/Shafiee74/Quantum-Imaging-with-Undetected-Photons  
**Stable tag:** [`v3.4.2-comment-a2`](https://github.com/Shafiee74/Quantum-Imaging-with-Undetected-Photons/releases/tag/v3.4.2-comment-a2)

## Publicly available

- Full simulator source (`setup_params.py`, `optics.py`, `simulate.py`, …) and registered YAML profiles A–D / `C_lab`
- Publication scripts for Figures 3–6 under `tools/`
- Figure 6(a) **derived** frequency tables under `reproducibility/figure6a/`
- Architecture identity helpers (`verification/configuration_identity.py`)
- Hash manifest `SHA256SUMS.txt` and audit `COMMENT_A2_TECHNICAL_AUDIT.md`

## Restricted (not published on GitHub)

- Original laboratory **EMCCD TIFF** stacks used for Figures 5–6 experimental panels  
  Path conventionally used on lab machines: `exp data/` (gitignored)

**Why:** Authors and supervisors agreed not to publish the original EMCCD datasets.

**Controlled access:** Available on reasonable request to the corresponding author / laboratory PI for editorial or reproducibility review. The public repository intentionally does **not** redistribute raw frames.

## What a clean checkout can reproduce without EMCCD

```bash
python tools/reproduce_figure6a.py --public
```

Produces simulation image strips + frequency panel (hollow circle = analytical 20 µm)
and writes MAE/RMSE under `reproducibility/figure6a/out/`.

Figures 3–4 are numerical-only and do not require EMCCD data. Figures 5–6
experimental **image** panels require the restricted TIFF archive; analysis
scripts remain public.

## Scope claim (do not overstate)

This package is a **prototype / pre-validation** physics-informed forward simulator.
Software verification and semi-quantitative experimental correspondence are
documented; the work does **not** claim absolute experimental calibration or a
fully validated closed-loop digital twin.

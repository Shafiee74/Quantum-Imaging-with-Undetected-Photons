# Manuscript refinements (Comment B / handover D) — author-owned

LaTeX / PDF / SI packaging is **outside this simulator repository**. Repo-side
software and data-availability items are complete (see `SUPERVISOR_COMMENT_STATUS.md`).

## B. Text to confirm in the manuscript/SI (authors)

1. **Sensitivity / uncertainty (main text)** — if not already present: 30 µm
   fringe-frequency estimate changes by up to **15.74%** under tested
   peak-selection/smoothing perturbations (S5.6); retain **n=2 at 180 µm**.

2. **Scope of validation** — describe the software as a **physics-informed
   digital-twin prototype**; do not imply complete experimental calibration or
   independent instrument-level validation. Repo wording already follows this.

3. **Table S3 caption** — replace any claim that full-precision crest positions /
   spacings / peak-selection audits are on GitHub. Use:
   `reproducibility/figure6a/reference/TABLE_S3_CAPTION_CORRECTION.md`.

## D. Author submission package (not in this repo)

- Final main manuscript PDF  
- Supplementary PDF  
- Complete cleaned LaTeX ZIP  

## Already available from this repository

- Stable tag: `v3.4.2-comment-a2`  
- Tip commit on `main` (`git rev-parse HEAD`)  
- Clean-run log: `reproducibility/figure6a/out/CLEAN_RUN_LOG.txt`  
- Software verification: `python -m verification.config_d_checks` → **264/264 PASS**

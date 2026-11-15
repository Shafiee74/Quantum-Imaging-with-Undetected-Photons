# Supervisor comment status — mandatory A / recommended B / handover D

**Date:** 2026-11-15  
**Public tip:** `origin/main`  
**Tag:** `v3.4.2-comment-a2` (must match tip after close-out push)  
**Repo:** https://github.com/Shafiee74/Quantum-Imaging-with-Undetected-Photons

## A. Mandatory corrections

| Item | Status | Evidence |
|---|---|---|
| **A1** Architecture A–D naming | **DONE** | Public README matches manuscript; YAML/`SETUP_PROFILE_NAME_ALIASES`/`tests/test_configuration_identifier_standardization.py` verify implementation (not labels alone). Legacy B↔D evidence map documented. |
| **A2** Figure 6(a) public workflow | **DONE** | Independent `--compute` / `--verify`; no overwrite; hollow 20 µm; MAE≈0.05215, RMSE≈0.05465; legacy Corrected scripts quarantined; `--replot-archived` explicitly labelled (exit 3). |
| **A3** Data/code availability accuracy | **DONE** | `DATA_AVAILABILITY.md` Restricted section uses the required wording; no TIFF-on-request promise. Table S3 public scope corrected (`TABLE_S3_CAPTION_CORRECTION.md`). |

## A′. Follow-up items (2026-11-15 supervisor note)

| Item | Status | Evidence |
|---|---|---|
| Table S3 caption vs shipped records | **DONE (repo)** | Caption correction + audit-scope docs; only derived counts/frequencies are public |
| Remaining verification failure (N1 throughput) | **DONE** | Expectations corrected for NF `mean(S_cam)` on absolute moments + documented 532/810/1550 nm rounding residual; `python -m verification.config_d_checks` → **264/264 PASS** |
| Reproducibility docs / checklist consistency | **DONE (repo)** | This file + `COMMENT_A2_TECHNICAL_AUDIT.md` + figure6a package |
| LaTeX / PDF / SI package polish | **OUT OF SCOPE here** | Authors own manuscript PDFs and LaTeX ZIP (not stored in this simulator repo) |

## B. Recommended manuscript refinements

| Item | Status |
|---|---|
| Main-text 30 µm sensitivity (15.74%) + 180 µm n=2 limitation | **AUTHOR MANUSCRIPT** — apply/confirm in SI/main text if not already present (`MANUSCRIPT_REFINEMENTS_CHECKLIST.md`) |
| Scope-of-validation wording (prototype, not fully calibrated DT) | **DONE in repo**; authors confirm identical wording in manuscript PDF |

## C. Already resolved (acknowledged)

No further code action. Supervisor’s MAE/RMSE recalculation matches the public package.

## D. Handover checklist

| Deliverable | Status |
|---|---|
| Updated repository commit/tag | **DONE** — tip + tag `v3.4.2-comment-a2` |
| Reproducibility clean-run log + Fig. 6(a) plot | **DONE** — `reproducibility/figure6a/out/CLEAN_RUN_LOG.txt`, `Figure6a_Final.png/.pdf` |
| Software verification 264/264 | **DONE** — `verification/config_d_checks.py` |
| Final manuscript PDF / SI PDF / LaTeX ZIP | **AUTHORS** (outside this repository) |

## Public reproduction command

```bash
python tools/reproduce_figure6a.py --compute
python tools/reproduce_figure6a.py --verify
# or: python tools/reproduce_figure6a.py --public
python -m verification.config_d_checks
```

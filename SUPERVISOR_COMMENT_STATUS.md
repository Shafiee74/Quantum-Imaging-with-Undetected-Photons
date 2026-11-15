# Supervisor comment status — mandatory A / recommended B / handover D

**Date:** 2026-11-15  
**Public tip:** see `git rev-parse HEAD` / `origin/main`  
**Tag:** `v3.4.2-comment-a2`  
**Repo:** https://github.com/Shafiee74/Quantum-Imaging-with-Undetected-Photons

## A. Mandatory corrections

| Item | Status | Evidence |
|---|---|---|
| **A1** Architecture A–D naming | **DONE** | Public README matches manuscript; YAML/`SETUP_PROFILE_NAME_ALIASES`/`tests/test_configuration_identifier_standardization.py` verify implementation (not labels alone). Legacy B↔D evidence map documented. |
| **A2** Figure 6(a) public workflow | **DONE (numerical public path)** | `tools/figure6a_experimental_fringe_center_correction.py` present; derived CSVs in `reproducibility/figure6a/`; `python tools/reproduce_figure6a.py --public` regenerates plot with **hollow circle at analytical 20 µm**; MAE=0.05215, RMSE=0.05465 (image-derived convention). Experimental **image strip** still requires private EMCCD (placeholders labeled “restricted”). |
| **A3** Data/code availability accuracy | **DONE** | `DATA_AVAILABILITY.md` + README section: EMCCD not published; controlled access via corresponding author/PI; clean numerical checkout path documented. |

## B. Recommended manuscript refinements

| Item | Status |
|---|---|
| Main-text 30 µm sensitivity (15.74%) + 180 µm n=2 limitation | **OPEN — manuscript edit** (see `MANUSCRIPT_REFINEMENTS_CHECKLIST.md`) |
| Scope-of-validation wording | **DONE in repo**; confirm identical wording in main manuscript PDF |

## C. Already resolved (acknowledged)

No further code action. Supervisor’s MAE/RMSE recalculation matches the public package.

## D. Handover checklist

| Deliverable | Status |
|---|---|
| Updated repository commit/tag | **DONE** — push after this status commit; tag `v3.4.2-comment-a2` |
| Reproducibility clean-run log + Fig. 6(a) plot | **DONE** — `reproducibility/figure6a/out/CLEAN_RUN_LOG.txt`, `Figure6a_Final.png/.pdf` |
| Final manuscript PDF / SI PDF / LaTeX ZIP / change log | **OPEN — authors** (not stored in this simulator repo) |

## Public reproduction command

```bash
python tools/reproduce_figure6a.py --public
```

# Comment A2 — Technical Audit (Digital Twin v3.4.2)

**Repository:** https://github.com/Shafiee74/Quantum-Imaging-with-Undetected-Photons  
**Package label:** Physics-Informed Reduced-Order SPDC-QIUP Forward Simulator Prototype  
**Reported manuscript version:** Digital Twin / QIUP Simulator **v3.4.2**  
**Audit date:** 2026-11-14

## 1. Verified A/B/C/D mapping

Resolved via `SETUP_PROFILE_NAME_ALIASES` → registered YAML → `resolve_architecture_profile`.

| Letter | Registry / YAML | `spdc_architecture` | `imaging_mode` | Physical crystals | Object type | Object interactions |
|---|---|---|---|---|---|---|
| A | `two_crystal_2014` → `configs/setup_two_crystal_2014.yaml` | `two_crystal` | `far_field` | 2 | transmissive | 1 |
| B | `two_crystal_near_field` → `configs/setup_two_crystal_near_field.yaml` | `two_crystal` | `near_field` | 2 | transmissive | 1 |
| C | `single_crystal_far_field` → `configs/setup_single_crystal_far_field.yaml` | `single_crystal_double_pass` | `far_field` | 1 | reflective | 1 |
| D | `single_crystal_near_field` → `configs/setup_single_crystal_near_field.yaml` | `single_crystal_double_pass` | `near_field` | 1 | reflective | 1 |

`benchmarks/four_config/definitions.py` and `tools/qiup_figure4_final_article.py` use the same manuscript labels.

## 2. B/D inconsistency verdict

**Documentation-only** relative to the manuscript.

- Code (`setup_params.py`, YAML profiles, Figure 3/4 workflows, four-config benchmark) already matched the manuscript.
- `README.md` previously listed legacy swapped B/D descriptions; corrected in this release.
- Legacy evidence letter swap (`legacy B` ↔ manuscript `D`) remains documented in `verification/configuration_identity.py` and is covered by `tests/test_configuration_identifier_standardization.py`. Physical YAML identities were never swapped; numerical arrays must not be reordered by letter alone.

## 3. Figures 3–4 architecture evidence

- Figure 3 assembly: `tools/qiup_figure3_final_article.py` (uses manuscript A–D panel generators).
- Figure 3(b) operating points: NL1 (A/B) & NL (C/D) = 83.7 °C; NL2 (A/B) = 84.7 °C; Δk=0 ≈ 75.59 °C (`tools/figure3b_manuscript_candidate.py`).
- Obsolete caption fragment “A/D at 83.7 °C and B/C at 84.3 °C” in `CAPTION_PANEL_B_NOTE` was **exported into** `FIGURE3_CAPTION_FINAL.md` but did **not** drive marker coordinates; string corrected; numerical operating points unchanged.
- Figure 4: `tools/qiup_figure4_final_article.py` → `resolve_setup_selection("A"|"B"|"C"|"D")` with manuscript architecture titles.

## 4. Supplementary Table S2 — path checklist (publication-relevant)

| Path | Status | Notes |
|---|---|---|
| `tools/qiup_figure3_final_article.py` | Present | Fig. 3 assembly |
| `tools/figure3a_manuscript_final.py` … `figure3e_fixed_detector_evidence.py` | Present | Panel generators |
| `tools/qiup_figure4_final_article.py` | Present | Fig. 4 assembly |
| `tools/figure4_stage*.py`, `tools/taskX6_U09_Figure4.py` | Present | Fig. 4 intermediates (regenerable) |
| `tools/c_lab_figure5_*.py`, `tools/generate_c_lab_figure5*.py` | Present | Fig. 5 analysis (needs private EMCCD for full exp. replot) |
| `tools/c_lab_figure6_*.py`, `tools/generate_c_lab_figure6*.py` | Present | Fig. 6 analysis |
| `tools/figure6a_simulated_fringe_frequency_correction.py` | Present | Simulation frequency correction |
| `tools/figure6a_experimental_fringe_center_correction.py` | **Restored** | Was missing at the Table S2 path; now the documented entry point |
| `tools/reproduce_figure6a.py` | Present | Public Fig. 6(a) reproduction entry (simulation path) |
| `configs/setup_{two_crystal_2014,two_crystal_near_field,single_crystal_far_field,single_crystal_near_field}.yaml` | Present | Generic A–D |
| `configs/setup_config_c_lab_v1.yaml` | Present | Laboratory C overlay (Figs. 5–6) |
| `benchmarks/four_config/` | Present | Matched A–D spatial benchmark |
| `verification/configuration_identity.py` | Present | Legacy↔manuscript letter map |
| `VERSION` | Present | `3.4.2` |

**Distinguish missing vs regenerable:** Figure 4 stage NPZ/CSV under `evidence/` are regenerable by the stage scripts / `qiup_figure4_final_article.py` and are gitignored. Absence of frozen evidence hashes locally is not a missing code dependency.

## 5. Version consistency

| Artifact | Before audit | After audit |
|---|---|---|
| `VERSION` | 3.4.2 | 3.4.2 |
| `README.md` header | 3.4.2 | 3.4.2 |
| `provenance.PACKAGE_VERSION` | 3.4.1 | **3.4.2** |
| Generated `qiup_digital_twin_full.py` | 3.4.1 (stale rebuild) | rebuilt to 3.4.2 |

Historical inventory filenames such as `configs/PARAMETER_INVENTORY_3.4.1.md` remain as prior audit artifacts; they are not the package release label.

## 6. Remaining reproducibility limitations (accurate)

1. **Private EMCCD datasets** for Figures 5–6 are intentionally not published; full independent experimental replot requires laboratory access.
2. `tools/figure6a_experimental_fringe_center_correction.py` regenerates experimental frequencies only when the local TIFF tree is present; it does not fabricate numbers.
3. Package remains a **prototype / pre-validation** simulator — not an experimentally calibrated digital twin.
4. Regenerable `evidence/` outputs and optional UI `node_modules` are excluded from the public hash-critical surface as documented in `verify_sha256_manifest.py`.

## 7. Corrected files (this Comment A2 close-out)

| File | Change |
|---|---|
| `README.md` | Fixed B/D architecture table; documented legacy letter map |
| `tools/qiup_figure3_final_article.py` | Corrected `CAPTION_PANEL_B_NOTE` temperatures |
| `provenance.py` | `PACKAGE_VERSION` / calibration statement → 3.4.2 |
| `qiup_digital_twin_full.py` | Deterministic rebuild after provenance update |
| `tools/figure6a_experimental_fringe_center_correction.py` | Restored Table S2 public path |
| `qiup-ui/scripts/smoke_test.py` | Manuscript B uses NL1/NL2 temps (not legacy single-crystal 84.3 °C) |
| `verification/config_d_checks.py` | Target registry `two_crystal_near_field` (manuscript B); document historical module name |
| `verification/__init__.py`, `verification/replay_helpers.py`, `repo_paths.py` | Version label comments → 3.4.2 |
| `COMMENT_A2_TECHNICAL_AUDIT.md` | This audit (repo root; `docs/` is gitignored) |
| `SHA256SUMS.txt` / `configs/single_file_build_hashes.json` | Regenerated after rebuild |

## 8. Regression / parity commands and results

Commands run on 2026-11-14 (local Windows, Python 3.10):

```text
python build_single_file.py --check
  -> OK qiup_digital_twin_full.py matches clean rebuild (18923 lines)

python -m verification.parity
  -> A/B/C/D: PASS (max abs error = 0 for all four profiles)

python -B -m pytest tests/test_configuration_identifier_standardization.py \
  tests/test_figure3b_manuscript_candidate.py \
  tests/test_figure3c_manuscript_candidate.py \
  tests/test_generic_architecture_reference_parameters.py \
  tests/test_generic_uniform_collection.py -q -p no:cacheprovider
  -> 32 passed

python -m verification.config_d_checks
  -> After A2 fix: module targets registry two_crystal_near_field (manuscript B).
  -> 262/263 PASS; 1 FAIL retained and documented (not silenced):
     "detected N1 carries throughput exactly once"
     Expectation N1=throughput*nbar is naive for two-crystal path-1 signal
     losses (N2≈throughput*nbar; N1 is lower). No production physics changed.

python tools/figure6a_experimental_fringe_center_correction.py
  -> exit 2 without private EMCCD TIFF root (expected; no fabricated CSV)

python verify_sha256_manifest.py
  -> OK after --write regeneration
```

## 9. Immutable release pointer

- **Public tip (`main`):** `0590ce8073c53f812166a19219aff15c560319c1`
- **Primary close-out commit:** `56df7f960bb2be1052307593173d5730e2ee560d`
- **Branch:** https://github.com/Shafiee74/Quantum-Imaging-with-Undetected-Photons/tree/main
- **Tag:** `v3.4.2-comment-a2` — https://github.com/Shafiee74/Quantum-Imaging-with-Undetected-Photons/releases/tag/v3.4.2-comment-a2
- **Anonymous checks (2026-11-14):** public README shows corrected A/B/C/D; `tools/figure6a_experimental_fringe_center_correction.py` reachable; `VERSION` = 3.4.2.
- EMCCD datasets remain unpublished by design.

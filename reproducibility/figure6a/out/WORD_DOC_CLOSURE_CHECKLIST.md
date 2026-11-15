# Word-doc Comment A2 closure checklist

Source: `Figure6a_A2_GitHub_Reproducibility_Instructions.docx`

| § | Requirement | Status |
|---|---|---|
| 1 | Distinguish historical vs final Figure 6(a) generations | **DONE** — `historical/` + quarantine notes |
| 2 | Preserve Φ_tr / κ_tr / 20% prominence / hollow 20 µm | **DONE** — `tools/reproduce_figure6a.py` |
| 3 | No silent published-value substitution; `--verify` vs archived replot | **DONE** — no overwrite; `--replot-archived` exit 3 |
| 4 | Table S3 experimental centers + algorithm documentation | **DONE** — `reference/Experimental_Frequency_Comparison.csv` + `EXPERIMENTAL_AUDIT_SCOPE.md` (full TIFF candidate audits remain laboratory-only) |
| 5 | Clean obsolete Corrected assets from active path | **DONE** — legacy `__main__` gated; historical outputs only |
| 6 | Clean-run independent compute + MAE/RMSE | **DONE** — `--public` PASS |
| 7 | Stable tag/SHA matching verified chain | **DONE** after retag of `v3.4.2-comment-a2` |

## Outside this repository (authors)

- Main manuscript PDF / SI PDF / LaTeX ZIP wording (supervisor B/D)
- Confirmation that SI §S5.6 text matches the public derived Table S3 record

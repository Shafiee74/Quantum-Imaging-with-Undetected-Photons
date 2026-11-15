# Figure 6(a) — independent computation and validation

## Pipeline (mandatory order)

1. **Compute** — production model → new simulated images → extracted centers → computed frequencies  
2. **Plot** — figure drawn only from computed simulation + derived experimental frequency record  
3. **Verify** — compare computed simulation against read-only reference CSV (never overwrite)

```bash
python tools/reproduce_figure6a.py --compute
python tools/reproduce_figure6a.py --verify
# equivalent:
python tools/reproduce_figure6a.py --public
```

Explicit non-reproduction mode (labelled; exits nonzero):

```bash
python tools/reproduce_figure6a.py --replot-archived
```

## Directory roles

| Path | Role |
|---|---|
| `reference/` | Validation targets only (`Figure6a_Final_Data.csv`, Table S3 experimental frequencies) |
| `computed/` | Newly calculated simulation results (`Figure6a_Computed_Data.csv`) |
| `out/` | Figures and validation reports generated from computed data |
| `historical/` | Quarantined obsolete / diagnostic outputs (not on the active path) |

## Model summary

- Transverse phase: \(\Phi_{\mathrm{tr}}=[4\pi/(\lambda_s f_c)](\Delta x_\ell x_c+\Delta y_\ell y_c)\)
- Analytical frequency: \(\kappa_{\mathrm{tr}}=2|\Delta x_\ell|/(\lambda_s f_c)\) (cycles/mm)
- \(\lambda_s=810\,\mathrm{nm}\), \(f_c=150\,\mathrm{mm}\) (`C_lab`)
- Image-derived simulated frequencies: uniform **20%** relative prominence gate
- At 20 µm (one resolved simulated crest): analytical hollow-circle marker

## Integrity rules

- `reference/Figure6a_Final_Data.csv` is **never** used as a computational substitute for `--compute`.
- There is **no** fallback that copies manuscript frequencies into the plot when mismatch occurs.
- On verification mismatch the process exits nonzero and reports calculated vs reference values.
- Legacy `Figure6a_Corrected*` generators are gated behind `--allow-historical`.

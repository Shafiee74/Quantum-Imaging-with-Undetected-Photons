# Integration manifest — `tools.visibility` (2026-10-15)

## Added

| Path | Role |
|------|------|
| `tools/visibility/__init__.py` | Public API exports |
| `tools/visibility/c_lab_shared.py` | Core estimators + `run_experiment` / `run_simulated_stack` |
| `tools/visibility/intensity.py` | `BackgroundMode`, `prepare_intensity_frame` |
| `tools/visibility/data/mirror_drift_5phase_um.json` | Default mirror positions (µm) |
| `tools/visibility/README.md` | Interface documentation |
| `tests/test_c_lab_visibility_shared.py` | Promoted unit tests |
| `examples/c_lab_visibility_minimal.py` | Minimal usage |

## Shim (compatibility)

| Path | Role |
|------|------|
| `tests/isolated/c_lab_visibility_shared_prototype/c_lab_visibility_shared.py` | Re-exports `tools.visibility.c_lab_shared` |

## Unchanged (by design)

- `tools/z1/estimator.py` — original Figure 5 four-step
- SPDC / `optics.py` / C_lab YAML / rabbit frozen data / experimental TIFFs
- Prior `evidence/` timestamps

## Validation

```powershell
python -m pytest tests/test_c_lab_visibility_shared.py -q
python -m pytest tests/isolated/c_lab_visibility_shared_prototype/test_prototype_shim_imports.py -q
python -m pytest tests/test_c_lab_visibility_shared_prototype_validation.py -q
python tools/c_lab_visibility_shared_prototype_validation.py
```

# Physics-Informed Reduced-Order SPDC-QIUP Forward Simulator Prototype

## QIUP Simulator v3.4.2 — Prototype / Pre-validation QIUP Simulator

**Release classification:** Prototype / Pre-validation QIUP Simulator (software freeze candidate for formal experimental validation)  
**Calibration status:** NOT YET EXPERIMENTALLY CALIBRATED

> Version 3.4.2 is a prototype/pre-validation physics-based QIUP simulator. The implemented physical model and software verification tests do not constitute experimental calibration or validation. Calibration against the actual laboratory pump, nonlinear crystals, optical relays, SLM, and detector is required before the system can be classified as a validated digital twin.

Validated or verified digital-twin status is **not claimed** for Version 3.4.2.

### Registered configurations

Manuscript / code convention (authoritative; see `SETUP_PROFILE_NAME_ALIASES` in `setup_params.py` and `verification/configuration_identity.py`):

| Profile | Architecture | Imaging | Object |
|---|---|---|---|
| A | two crystal (NL1, NL2) | far field | transmissive, single pass |
| B | two crystal (NL1 → relay → object → NL2) | near field | transmissive, single pass |
| C | single crystal, double pass | far field | reflective |
| D | single crystal, double pass | near field | reflective |

**Legacy letter note:** Pre-2026-10-19 evidence archives used swapped letters for the near-field profiles (`legacy B` ↔ manuscript `D`, `legacy D` ↔ manuscript `B`). Physical YAML registry names were never swapped. Use `verification.configuration_identity` helpers when reading frozen legacy CSVs; do not reorder numerical arrays by letter alone.

### Source of truth and build

The modular files `setup_params.py`, `latents.py`, `optics.py`, `detector.py`,
`provenance.py`, `simulate.py` are authoritative. `qiup_digital_twin_full.py` is
generated deterministically by `python build_single_file.py` and must never be
edited by hand (the file name is historical; the package is not a digital twin).

### Project structure

| Path | Role |
|---|---|
| `setup_params.py`, `latents.py`, `optics.py`, `detector.py`, `provenance.py`, `simulate.py` | Authoritative modular simulator (kept at repository root for freeze stability) |
| `qiup_digital_twin_full.py` | Generated single-file distribution (`build_single_file.py`) |
| `configs/` | Registered YAML profiles and generated inventory/audit JSON |
| `verification/` | Independent D physics, parity, and physics-change audit |
| `tests/` | Complete regression suite |
| `diagnostics/` | Collection-envelope / FWHM helpers used by optics and tests (required) |
| `evidence/` | Local regenerable article/verification outputs (gitignored). Produce with `tools/*.py` |
| `object_maps/` | C_lab rabbit object-map loaders used by production and tests |
| `examples/` | Notebooks and demonstration generators |
| `data/` | Shipped MNIST train subset (100 random idx samples under `data/MNIST/raw/`) |
| `qiup_web/` | Django HTTP API for the prototype UI (`/api/defaults`, `/api/preview`, `/api/run`) |
| `qiup-ui/` | Optional Next.js frontend (proxies `/api/*` to Django; no Node simulator bridge) |
| `qiup_ui_backend.py` | Simulator service used by Django (and optional CLI) |

### Version 3.4.2 verification

```bash
python build_single_file.py --check                       # single file == clean rebuild
python -m verification.parity                             # modular vs single-file parity, A/B/C/D
python -m verification.config_d_checks                    # two-crystal near-field checks (historical name; manuscript B / legacy D)
python parameter_inventory_audit.py                       # trace parameter consumption per profile (slow)
python build_parameter_spec.py                            # regenerate parameter inventory JSON
python -B -m pytest tests -q -p no:cacheprovider          # complete test suite
python qiup_web/manage.py test api -v 2                   # Django UI API tests
python verify_sha256_manifest.py                          # verify SHA256SUMS.txt
```

Prototype UI (Django API + optional Next.js frontend):

```bash
python qiup_web/manage.py runserver 8000                  # API on :8000
cd qiup-ui && npm run dev                                 # UI on :3000 (rewrites /api → Django)
python qiup-ui/scripts/smoke_test.py http://127.0.0.1:8000
```


- Parameter inventory: `configs/three_configuration_parameter_spec.json`
- Frozen evidence hashes: `tests/test_figures3_6_graphics_only.py` and `SHA256SUMS.txt`
- Hash manifest: `SHA256SUMS.txt`

**Forbidden claims:**
- Verified Digital Twin
- Validated Digital Twin
- Experimentally calibrated or experimentally validated simulator
- Official / release dataset generation for training or verification from this package

---

## Historical notes (Version 3.3.x – 3.4)

The notes below describe earlier development stages and are kept for traceability.
Where they mention only A/B/C, they predate the registration of profile D.

Package label (Comment 8.2, retained in Version 3.4.2): `Physics-Informed Reduced-Order SPDC-QIUP Forward Simulator Prototype`.

Undetected-photon imaging (QIUP) simulator for the **single-crystal double-pass** setup. The manuscript path uses an analog EMCCD model.

The five modular files are authoritative: `setup_params.py`, `latents.py`, `optics.py`, `detector.py`, and `simulate.py`. `qiup_digital_twin_full.py` is generated and must not be edited by hand (the filename is historical; this package is a prototype, not a validated digital twin).

Finding status (Code / Test / Evidence / Review / Closed) is tracked only in the configs and evidence registers listed above.

### Code work completed

- Gain moved from empirical coefficients (`s_gain` and visibility penalties) to the physical chain `P → r → n̄ → (N1, N2, C12) → optical intensity → EMCCD`.
- KTP refractive-index and thermo-optic tables updated to Kato & Takaoka (20 °C reference).
- Hidden poling-period calibration removed from Physical Mode; `Δk` enters the gain, not a separate flux scale.
- `d_eff` from square QPM (`m=1`, `D=0.5`) and `k_SPDC,0` in SI; mW→W conversion applied once.
- Geometric overlap `O_geom` from pump waist and focus; QPM phase is not folded into it.
- Physical Mode default: one crystal, two interactions. `n_crystals` is hardware count, not architecture. Architecture is selected by `spdc_architecture` (`single_crystal_double_pass` or `two_crystal`).
- Double-pass pump path on the manuscript route: `Pp1 = ηp,pre Pp,laser` then `Pp2 = ηp,return Pp1`. In `two_crystal`, NL2 pump is taken separately from the laser: `Pp2 = η_NL2 Pp,laser` (not the return path). `s_pump` instability applies only to frame power, not again to flux.
- Idler seed applied once: `Tseed = ηmode · ηi,path · To`; `ηmode` and pump overlap are not multiplied again onto visibility.
- Double-pass moments use `n̄1` and `n̄2`; signal losses after the source; interference only from `N1, N2, C12`.
- Path-length coherence (`γpath`) only on `C12`. Induced emission and pedestal are not repeated in the detector.
- Run and sweep metadata apply physical quantities for real.
- Naming updated for single-crystal double-pass. The combined file is rebuilt after modular edits.
- Intensity-level `imaging_mode`: `near_field` (position correlation, σ from L) or `far_field` (momentum correlation, σ from \(w_p,\lambda_i,f_u\)).
- Optional Fabry–Pérot / etalon fringe model (intensity level, default off): `enable_etalon_fringes` with fringe angle relative to horizontal.

### Not implemented / outside prototype scope

- Item 7.1 / Finding 1: crystal length `crystal_length_mm` is a user physics variable in `[2, 20]` mm (`physics_team_user_variable_v1` — replaces the older fixed 10 mm crystal requirement in the critical report). Defaults: `single_blur`=10 mm; `double_blur`=2 mm per interaction. Decision record: `configs/physics_team_crystal_length_decision_v1.json`. L is not among the frozen CFG-P0-00 fields.

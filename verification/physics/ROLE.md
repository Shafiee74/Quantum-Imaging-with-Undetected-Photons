# `verification/physics`

**INDEPENDENT REFERENCE** equations for U1–U9 style physics controls
(energy conservation, SPDC gain, phase matching, transparent/blocked,
amplitude/phase/visibility, spatial mapping, correlation widths,
longitudinal idler-lens phase).

- Implementation: `independent_equations.py`
- Metadata catalog: `equation_meta.py`
- Reuses (does not duplicate) `verification.independent_reference` where that
  module already encodes the same independent law.

**PRODUCTION IMPLEMENTATION** remains in `setup_params.py` / `optics.py` /
`simulate.py` and must not be called to generate expected values here.

Created by Task 02; populated by Task 04. Paper artifacts go under
`evidence/taskXX_<name>/` via `output_layout`.

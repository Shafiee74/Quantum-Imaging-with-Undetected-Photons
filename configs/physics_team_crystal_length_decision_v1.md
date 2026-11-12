# Physics-team crystal-length decision (supersedes fixed 10 mm)

**Decision ID:** `physics_team_crystal_length_decision_v1`  
**Policy ID:** `physics_team_user_variable_v1`  
**Machine-readable:** `configs/physics_team_crystal_length_decision_v1.json`

## Superseded requirement (historical traceability)

The **Digital Twin v3.3.1 Critical Problems Team Report** (15 September 2026) and the
September 2026 action plan described **Critical Problem 1 / CFG-P0-00** as requiring a
**fixed canonical crystal length of 10 mm** frozen into the approved system identity.

That fixed-length requirement is **superseded** by this physics-team-approved policy.
Do **not** revert code or tests to enforce a frozen 10 mm canonical field.

## Approved specification (current)

| Rule | Value |
|------|--------|
| `crystal_length_mm` | User-selectable physical simulation variable |
| Allowed range | **2.0 ≤ crystal_length_mm ≤ 20.0 mm** (fail-closed outside) |
| `single_blur` default | **10.0 mm** |
| `double_blur` default | **2.0 mm** for interaction 1 **and** interaction 2 |
| Explicit user value | Overrides the blur-linked default |
| Canonical identity | `crystal_length_mm` is **not** a frozen CFG-P0-00 hash field |

## Runtime facts (verified by tests)

- `setup_params.crystal_length_policy()` → `policy_id = physics_team_user_variable_v1`
- Canonical lab YAML (`double_blur`) ships with **2.0 mm** default L (not 10 mm)
- Evidence: `tests/test_canonical_config.py`, fixture `configs/canonical_config_hash.json`

## Governance (not release closure)

- **Release is not independently approved.**
- Finding **CFG-P0-00**: Code Implemented = Yes, Test Passed = Yes; Evidence / Review / Closed remain **Pending / No** until formal sign-off.

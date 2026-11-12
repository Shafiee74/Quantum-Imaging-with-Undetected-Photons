"""Generic A/B/C/D reference crystal parameters (Lemos NL1 benchmark for C/D)."""
from __future__ import annotations

from verification.generic_architecture_reference_parameters import (
    assert_generic_expectations,
    collect_runtime_table,
)


def test_generic_architecture_runtime_parameters() -> None:
    table = collect_runtime_table()
    errors = assert_generic_expectations(table)
    assert not errors, errors


def test_c_and_d_match_nl1_delta_k_of_a() -> None:
    table = collect_runtime_table()
    by_cfg = {r["configuration"]: r for r in table["generic_rows"] if r["interaction"] in ("NL1", "NL")}
    dk_a_nl1 = next(r["delta_k_qpm_per_m"] for r in table["generic_rows"] if r["configuration"] == "A" and r["interaction"] == "NL1")
    for letter in ("C", "D"):
        dk = by_cfg[letter]["delta_k_qpm_per_m"]
        assert abs(dk - dk_a_nl1) < 1e-3, f"{letter} Δk should match A NL1 at 83.7 °C"

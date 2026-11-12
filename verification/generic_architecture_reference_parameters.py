"""Runtime verification of generic A/B/C/D and C_lab crystal reference parameters."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from setup_params import (
    crystal_length_m_from_setup,
    delta_k_qpm_per_m_from_setup,
    is_two_crystal_architecture,
    resolve_setup_selection,
)
from verification.independent_reference import phase_matching_temperature_C

ROOT = Path(__file__).resolve().parents[1]

GENERIC_PROFILES = {
    "A": "two_crystal_2014",
    "B": "two_crystal_near_field",
    "C": "single_crystal_far_field",
    "D": "single_crystal_near_field",
}

EXPECTED = {
    "lambda_p_nm": 532.0,
    "lambda_s_nm": 810.0,
    "lambda_i_nm": 1550.0,
    "poling_period_um": 9.675,
    "crystal_length_mm": 2.0,
    "A": {"NL1_C": 83.7, "NL2_C": 84.7},
    "B": {"NL1_C": 83.7, "NL2_C": 84.7},
    "C": {"NL_C": 83.7},
    "D": {"NL_C": 83.7},
}


def _sinc2_envelope(delta_k_qpm_per_m: float, length_m: float) -> float:
    x = 0.5 * delta_k_qpm_per_m * length_m
    if abs(x) < 1e-15:
        return 1.0
    return (math.sin(x) / x) ** 2


def _row_for_interaction(
    letter: str,
    profile: str,
    setup: Any,
    *,
    interaction: int,
    label: str,
    t_c: float,
) -> dict[str, Any]:
    length_m = crystal_length_m_from_setup(setup, interaction=interaction)
    dk = delta_k_qpm_per_m_from_setup(setup, interaction=interaction)
    return {
        "configuration": letter,
        "profile": profile,
        "interaction": label,
        "T_operating_C": t_c,
        "L_mm": length_m * 1000.0,
        "lambda_p_nm": float(setup.lambda_p_nm),
        "lambda_s_nm": float(setup.lambda_s_nm),
        "lambda_i_nm": float(setup.lambda_i_nm),
        "poling_period_um": float(setup.poling_period_um),
        "delta_k_qpm_per_m": dk,
        "delta_k_L_over_2pi": dk * length_m / (2.0 * math.pi),
        "sinc2_L_dk_half": _sinc2_envelope(dk, length_m),
    }


def collect_runtime_table() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for letter, profile in GENERIC_PROFILES.items():
        setup = resolve_setup_selection(profile)
        if is_two_crystal_architecture(setup):
            rows.append(
                _row_for_interaction(
                    letter,
                    profile,
                    setup,
                    interaction=1,
                    label="NL1",
                    t_c=float(setup.T_crystal_NL1_C),
                )
            )
            rows.append(
                _row_for_interaction(
                    letter,
                    profile,
                    setup,
                    interaction=2,
                    label="NL2",
                    t_c=float(setup.T_crystal_NL2_C),
                )
            )
        else:
            rows.append(
                _row_for_interaction(
                    letter,
                    profile,
                    setup,
                    interaction=1,
                    label="NL",
                    t_c=float(setup.T_crystal_C),
                )
            )

    c_lab = resolve_setup_selection("C_lab")
    cl_len = crystal_length_m_from_setup(c_lab, interaction=1)
    cl_dk = delta_k_qpm_per_m_from_setup(c_lab, interaction=1)
    cl_t = float(c_lab.T_crystal_C)

    t_pm0 = phase_matching_temperature_C(
        532.0,
        810.0,
        1550.0,
        9.675,
        cl_t,
    )

    return {
        "generic_rows": rows,
        "C_lab": {
            "profile": "config_c_lab_v1",
            "L_mm": cl_len * 1000.0,
            "T_operating_C": cl_t,
            "delta_k_qpm_per_m": cl_dk,
            "sinc2_L_dk_half": _sinc2_envelope(cl_dk, cl_len),
            "model_T_pm0_C": t_pm0,
        },
        "expected": EXPECTED,
    }


def assert_generic_expectations(table: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    by_cfg: dict[str, list[dict[str, Any]]] = {}
    for row in table["generic_rows"]:
        by_cfg.setdefault(row["configuration"], []).append(row)

    for letter in ("A", "B"):
        exp = EXPECTED[letter]
        for row in by_cfg.get(letter, []):
            if row["interaction"] == "NL1" and abs(row["T_operating_C"] - exp["NL1_C"]) > 1e-6:
                errors.append(f"{letter} NL1 T={row['T_operating_C']} expected {exp['NL1_C']}")
            if row["interaction"] == "NL2" and abs(row["T_operating_C"] - exp["NL2_C"]) > 1e-6:
                errors.append(f"{letter} NL2 T={row['T_operating_C']} expected {exp['NL2_C']}")
            if abs(row["L_mm"] - 2.0) > 1e-6:
                errors.append(f"{letter} L={row['L_mm']} mm expected 2 mm")

    for letter in ("C", "D"):
        exp_t = EXPECTED[letter]["NL_C"]
        for row in by_cfg.get(letter, []):
            if abs(row["T_operating_C"] - exp_t) > 1e-6:
                errors.append(f"{letter} T={row['T_operating_C']} expected {exp_t}")
            if abs(row["L_mm"] - 2.0) > 1e-6:
                errors.append(f"{letter} L={row['L_mm']} expected 2 mm")
            if abs(row["poling_period_um"] - 9.675) > 1e-6:
                errors.append(f"{letter} poling period {row['poling_period_um']} expected 9.675")

    if abs(table["C_lab"]["L_mm"] - 2.0) > 1e-6:
        errors.append(f"C_lab L={table['C_lab']['L_mm']} expected 2 mm")

    return errors


def main() -> int:
    table = collect_runtime_table()
    errors = assert_generic_expectations(table)
    out_dir = ROOT / "evidence" / "configuration_parameter_correction_20251019"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runtime_parameter_table.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(json.dumps(table, indent=2))
    if errors:
        print("ASSERTION FAILURES:", errors)
        return 1
    print("All generic/C_lab length and temperature checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

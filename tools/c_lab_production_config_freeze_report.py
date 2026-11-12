#!/usr/bin/env python3
"""Emit evidence for the frozen C_lab production configuration (Mi=2.0 freeze)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from setup_params import (
    crystal_length_m_from_setup,
    delta_k_qpm_per_m_from_setup,
    resolve_setup_selection,
    resolved_spatial_mapping,
    setup_selection_info,
)

EVIDENCE_ROOT = ROOT / "evidence" / "c_lab_production_config_freeze"
YAML_PATH = ROOT / "configs" / "setup_config_c_lab_v1.yaml"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_report() -> dict:
    setup = resolve_setup_selection("C_lab")
    mapping = resolved_spatial_mapping(setup)
    dk1 = float(delta_k_qpm_per_m_from_setup(setup, interaction=1))
    dk2 = float(delta_k_qpm_per_m_from_setup(setup, interaction=2))
    return {
        "setup_id": setup.setup_id,
        "yaml_path": str(YAML_PATH.relative_to(ROOT)),
        "yaml_selection": setup_selection_info(setup),
        "frozen_parameters": {
            "crystal_material": "ppKTP",
            "spdc_type": setup.spdc_type,
            "crystal_length_mm": float(setup.crystal_length_mm),
            "crystal_length_m_ix1": float(crystal_length_m_from_setup(setup, interaction=1)),
            "T_crystal_C": float(setup.T_crystal_C),
            "poling_period_um": float(setup.poling_period_um),
            "lambda_nm": [float(setup.lambda_p_nm), float(setup.lambda_s_nm), float(setup.lambda_i_nm)],
            "pump_waist_um": float(setup.pump_waist_um),
            "Ms": float(setup.far_field_signal_relay_magnification),
            "Mi_effective_idler_relay": float(setup.far_field_idler_relay_magnification),
            "f_signal_mm": float(setup.F2_mm),
            "f_idler_mm": float(mapping["far_field_idler_focal_mm"]),
            "pixel_um": float(setup.pixel_um),
            "object_map_pixel_um": float(setup.object_map_pixel_um),
            "object_type": setup.object_type,
            "imaging_mode": setup.imaging_mode,
            "architecture": setup.spdc_architecture,
        },
        "delta_k0_rad_per_m": {
            "interaction_1_forward_sign": dk1,
            "interaction_2_backward_sign": dk2,
            "computed_by": "setup_params.delta_k_qpm_per_m_from_setup (Sellmeier + QPM; not hardcoded)",
            "Mi_affects_delta_k0": False,
            "Mi_affects_delta_k0_note": (
                "Δk(q_s,q_i) uses wavelengths, temperature, poling period and propagation sign only; "
                "far_field_idler_relay_magnification enters spatial mapping, not the axial QPM mismatch formula."
            ),
        },
        "spatial_mapping": {
            "signed_object_to_camera_M": float(mapping["signed_object_to_camera_magnification"]),
            "M_magnitude": float(mapping["magnification_magnitude"]),
            "Ms": float(mapping["far_field_signal_relay_magnification"]),
            "Mi": float(mapping["far_field_idler_relay_magnification"]),
            "formula": "M = -(Ms/Mi)*lambda_s*f_s/(lambda_i*f_i)",
            "michelson_phase": "phi = 4*pi*delta_z_um/1.55 (idler wavelength in um); independent of Ms/Mi",
        },
        "superseded_assumptions_reconciled": [
            "Mi=1.5 production freeze (2026-10-15 morning) is SUPERSEDED; authoritative Mi=2.0.",
            "L=10 mm references in isolated pilots/cache audits are not the production C_lab crystal length (L=2 mm).",
            "Reflective SLM placeholder in legacy YAML replaced by transmissive binary_mask + interaction_count=2.",
        ],
    }


def main() -> None:
    out = EVIDENCE_ROOT / utc_stamp()
    out.mkdir(parents=True, exist_ok=True)
    report = build_report()
    (out / "C_LAB_PRODUCTION_FREEZE.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "C_LAB_PRODUCTION_FREEZE.md").write_text(
        "\n".join(
            [
                "# C_lab production configuration freeze",
                "",
                f"- **YAML:** `{report['yaml_path']}`",
                f"- **L:** {report['frozen_parameters']['crystal_length_mm']} mm",
                f"- **Ms / Mi:** {report['frozen_parameters']['Ms']} / {report['frozen_parameters']['Mi_effective_idler_relay']}",
                f"- **M (object→camera):** {report['spatial_mapping']['signed_object_to_camera_M']:.6f}",
                f"- **Δk₀ ix1 / ix2:** {report['delta_k0_rad_per_m']['interaction_1_forward_sign']:.4f} / "
                f"{report['delta_k0_rad_per_m']['interaction_2_backward_sign']:.4f} rad/m",
                f"- **Mi changes Δk₀:** {report['delta_k0_rad_per_m']['Mi_affects_delta_k0']}",
                "",
                "Transverse mismatch Δk(q_s,q_i) remains the existing paraxial implementation in optics/setup_params.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"evidence={out}")
    print("PRODUCTION_FILES_MODIFIED = YES (configs/setup_config_c_lab_v1.yaml)")


if __name__ == "__main__":
    main()

"""Task 16 definitions — lab Config C + experimental audit policy."""
from __future__ import annotations

from pathlib import Path
from typing import Any

PROVENANCE_LABELS: tuple[str, ...] = (
    "MEASURED",
    "LITERATURE",
    "ASSUMED",
    "CALIBRATED",
    "UNKNOWN",
)

# Z5 physics-team correction. These values describe distinct locations and
# must not be treated as competing measurements of the same quantity.
PUMP_POWER_CANDIDATES_MW: tuple[float, ...] = (250.0, 70.0)
PUMP_POWER_STATUS: str = "RESOLVED_DISTINCT_SOURCE_AND_CRYSTAL_POWERS_Z5"

EXPERIMENTAL_DATA_ROOT_DEFAULT = Path(
    r"D:\Quantum\Quantum\QIUP_Synthetic_Dataset"
    r"\Ver1.3.4(Digital_Twin_Verification)"
    r"\Report\Received_Verification\for generate results\exp data"
)

# New lab profile name — must not collide with stock "C" / single_crystal_far_field.
LAB_PROFILE_ID: str = "config_c_lab_v1"
LAB_PROFILE_YAML_REL: str = "configs/setup_config_c_lab_v1.yaml"
STOCK_C_PROFILE: str = "single_crystal_far_field"

DEFINITIONS_MANIFEST: dict[str, Any] = {
    "task": "task16_config_c_lab",
    "read_only_experimental_data": True,
    "forbidden_ops": [
        "rename_source",
        "move_source",
        "overwrite_source",
        "resize_source_tiff",
        "crop_source_in_place",
        "normalize_source_in_place",
    ],
    "stock_config_c": STOCK_C_PROFILE,
    "lab_profile_id": LAB_PROFILE_ID,
    "lab_profile_yaml": LAB_PROFILE_YAML_REL,
    "pump_power_policy": {
        "status": PUMP_POWER_STATUS,
        "candidates_mW": list(PUMP_POWER_CANDIDATES_MW),
        "rule": (
            "250 mW is the nominal source rating; 70 mW is the reported "
            "power at the crystal. Do not conflate their locations."
        ),
    },
    "provenance_labels": list(PROVENANCE_LABELS),
    "acquisition_group_policy": (
        "Infer groups ONLY from explicit directory/file naming tokens; "
        "do not invent experimental meaning."
    ),
}

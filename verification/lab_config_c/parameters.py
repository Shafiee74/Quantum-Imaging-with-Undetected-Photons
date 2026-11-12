"""Laboratory parameter manifest for config_c_lab_v1 (provenance labels)."""
from __future__ import annotations

from typing import Any

from verification.lab_config_c.definitions import (
    PUMP_POWER_CANDIDATES_MW,
    PUMP_POWER_STATUS,
)


def laboratory_parameter_rows() -> list[dict[str, Any]]:
    """One row per lab-relevant parameter with explicit provenance label."""
    rows: list[dict[str, Any]] = [
        {
            "parameter": "spdc_architecture",
            "value": "single_crystal_double_pass",
            "unit": "",
            "provenance": "ASSUMED",
            "notes": "Inherited from stock Configuration C (far-field Michelson).",
        },
        {
            "parameter": "imaging_mode",
            "value": "far_field",
            "unit": "",
            "provenance": "ASSUMED",
            "notes": "Inherited from stock Configuration C.",
        },
        {
            "parameter": "crystal_material",
            "value": "ppKTP",
            "unit": "",
            "provenance": "ASSUMED",
            "notes": "Stock C default; not yet confirmed against this lab run sheet.",
        },
        {
            "parameter": "crystal_length_mm",
            "value": "2.0",
            "unit": "mm",
            "provenance": "ASSUMED",
            "notes": "Stock C default; laboratory confirmation pending.",
        },
        {
            "parameter": "lambda_p_nm",
            "value": "532.0",
            "unit": "nm",
            "provenance": "LAB_NOMINAL",
            "notes": (
                "Physics-team nominal pump wavelength. "
                "Energy-exact mathematical value from 810/1550 is "
                "≈531.991525 nm (ENERGY_EXACT_LAMBDA_P_FROM_810_1550_NM); "
                "do not substitute it to force U01 residual to zero."
            ),
        },
        {
            "parameter": "lambda_s_nm",
            "value": "810.0",
            "unit": "nm",
            "provenance": "LAB_NOMINAL",
            "notes": "Physics-team nominal signal wavelength.",
        },
        {
            "parameter": "lambda_i_nm",
            "value": "1550.0",
            "unit": "nm",
            "provenance": "LAB_NOMINAL",
            "notes": "Physics-team nominal idler wavelength.",
        },
        {
            "parameter": "pump_power_laser_mW",
            "value": "250.0",
            "unit": "mW",
            "provenance": "LAB_CONFIRMED",
            "notes": "Nominal laser/source rating; not the power at the crystal.",
        },
        {
            "parameter": "pump_power_at_crystal_mW",
            "value": "70.0",
            "unit": "mW",
            "provenance": "LAB_CONFIRMED",
            "notes": "Reported local pump power at the crystal; independent of beam width.",
        },
        {
            "parameter": "pump_target_power_NL1_mW",
            "value": "70.0",
            "unit": "mW",
            "provenance": "LAB_CONFIRMED",
            "notes": "Target records the reported local crystal power.",
        },
        {
            "parameter": "pump_target_power_NL2_mW",
            "value": "70.0",
            "unit": "mW",
            "provenance": "LAB_CONFIRMED",
            "notes": "Unit-return model retains 70 mW for the second interaction.",
        },
        {
            "parameter": "spdc_source_mode",
            "value": "fixed_nbar",
            "unit": "",
            "provenance": "ASSUMED",
            "notes": "Kept because absolute pump-to-gain calibration remains unavailable.",
        },
        {
            "parameter": "camera_filter_preset",
            "value": "laboratory_10nm",
            "unit": "",
            "provenance": "ASSUMED",
            "notes": "Stock C laboratory_10nm preset; not verified against these TIFFs.",
        },
        {
            "parameter": "detector_mode",
            "value": "analog_emccd",
            "unit": "",
            "provenance": "ASSUMED",
            "notes": "Stock C detector mode.",
        },
        {
            "parameter": "adc_bit_depth",
            "value": "16",
            "unit": "bit",
            "provenance": "ASSUMED",
            "notes": "Consistent with measured TIFF dtype uint16 / mode I;16.",
        },
        {
            "parameter": "experimental_image_dtype",
            "value": "uint16",
            "unit": "",
            "provenance": "MEASURED",
            "notes": "From READ-ONLY TIFF audit of experimental tree.",
        },
        {
            "parameter": "experimental_image_mode",
            "value": "I;16",
            "unit": "",
            "provenance": "MEASURED",
            "notes": "PIL mode for all audited .tif files.",
        },
        {
            "parameter": "experimental_n_files",
            "value": "see_inventory",
            "unit": "",
            "provenance": "MEASURED",
            "notes": "Counted during Task 16 audit.",
        },
        {
            "parameter": "object_map_pixel_um",
            "value": "null",
            "unit": "um",
            "provenance": "UNKNOWN",
            "notes": "Multiple TIFF crop sizes present; pixel pitch not confirmed.",
        },
        {
            "parameter": "camera_pixel_um",
            "value": "null",
            "unit": "um",
            "provenance": "UNKNOWN",
            "notes": "Camera model / pixel pitch not in file metadata.",
        },
        {
            "parameter": "roi",
            "value": "null",
            "unit": "",
            "provenance": "UNKNOWN",
            "notes": "Not present in filenames or TIFF metadata.",
        },
        {
            "parameter": "binning",
            "value": "null",
            "unit": "",
            "provenance": "UNKNOWN",
            "notes": "Not present in filenames or TIFF metadata.",
        },
        {
            "parameter": "crop_origin",
            "value": "null",
            "unit": "px",
            "provenance": "UNKNOWN",
            "notes": "Varying HxW across files implies cropping, origin unknown.",
        },
        {
            "parameter": "preprocessing",
            "value": "null",
            "unit": "",
            "provenance": "UNKNOWN",
            "notes": "No preprocessing recipe shipped with the experimental tree.",
        },
        {
            "parameter": "camera_metadata",
            "value": "null",
            "unit": "",
            "provenance": "UNKNOWN",
            "notes": "EXIF / camera tags absent on audited TIFFs.",
        },
        {
            "parameter": "far_field_focal_mm",
            "value": "150.0",
            "unit": "mm",
            "provenance": "LAB_CONFIRMED",
            "notes": (
                "Idler-path / undetected-arm Fourier lens f_i "
                "(resolved_far_field_focal_mm). Physics-team: 150 mm. "
                "Stock Configuration C retains 75 mm Lemos placeholder."
            ),
        },
        {
            "parameter": "F1_mm",
            "value": "150.0",
            "unit": "mm",
            "provenance": "LAB_CONFIRMED",
            "notes": (
                "Synced with far_field_focal_mm for lab-matched C so "
                "M_QIUP and far-field signed M remain consistent."
            ),
        },
        {
            "parameter": "F2_mm",
            "value": "150.0",
            "unit": "mm",
            "provenance": "LAB_CONFIRMED",
            "notes": (
                "Signal-path Fourier lens f_s; also used as camera/front "
                "imaging f_c in optics far-field phase (no separate field)."
            ),
        },
        {
            "parameter": "poling_period_um",
            "value": "9.675",
            "unit": "um",
            "provenance": "ASSUMED",
            "notes": "Stock C QPM period.",
        },
        {
            "parameter": "spdc_gain_per_sqrt_mW",
            "value": "null",
            "unit": "mW^-0.5",
            "provenance": "UNKNOWN",
            "notes": "Not calibrated for this laboratory dataset.",
        },
    ]
    return rows


def unresolved_parameter_rows() -> list[dict[str, Any]]:
    return [
        {
            "parameter": "roi",
            "status": "UNRESOLVED",
            "candidates": "",
            "reason": "Not available in directory names or TIFF metadata.",
        },
        {
            "parameter": "binning",
            "status": "UNRESOLVED",
            "candidates": "",
            "reason": "Not available in directory names or TIFF metadata.",
        },
        {
            "parameter": "crop_origin",
            "status": "UNRESOLVED",
            "candidates": "",
            "reason": "Multiple image dimensions; crop origin not documented.",
        },
        {
            "parameter": "preprocessing",
            "status": "UNRESOLVED",
            "candidates": "",
            "reason": "No preprocessing specification with the data package.",
        },
        {
            "parameter": "camera_metadata",
            "status": "UNRESOLVED",
            "candidates": "",
            "reason": "No EXIF/camera tags on audited TIFFs.",
        },
        {
            "parameter": "object_map_pixel_um",
            "status": "UNRESOLVED",
            "candidates": "",
            "reason": "Pixel pitch not measured for these recordings.",
        },
        {
            "parameter": "camera_pixel_um",
            "status": "UNRESOLVED",
            "candidates": "",
            "reason": "Camera model / pitch not recorded with files.",
        },
        {
            "parameter": "spdc_gain_per_sqrt_mW",
            "status": "UNRESOLVED",
            "candidates": "",
            "reason": "Gain coefficient not yet calibrated for this lab.",
        },
    ]

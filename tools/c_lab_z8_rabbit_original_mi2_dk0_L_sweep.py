"""TEST ONLY — 8-case sweep: L×Δk₀×object with Mi=2 and original-size physical rabbit."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from benchmarks.targets.geometry import centered_axis_um
from object_maps.rabbit_loader import load_object_map
from optics import spdc_crystal_state
from scipy.ndimage import map_coordinates
import setup_params
from setup_params import (
    OBJECT_DEVICE_MASK,
    OBJECT_MODE_GENERIC_RESEARCH,
    OBJECT_TYPE_TRANSMISSIVE,
    resolved_spatial_mapping,
)
from tools.c_lab_a6_2_calibrated_1024_audit import (
    APERTURE_H_MM,
    APERTURE_W_MM,
    N,
    axis_um,
    hash_protected,
    measures,
    project_once,
    sample_camera,
    save_json,
    signal_widths,
)
from tools.c_lab_constants import C_LAB_CRYSTAL_LENGTH_MM
from tools.c_lab_isolated_pilot import (
    DeltaKCalibrationState,
    build_pilot_setup,
    run_forward_case,
)
from tools.z2.physics import ref_from_T

Y1_MASK = ROOT / "evidence" / "taskY1_rabbit_ground_truth" / "data" / "frozen" / "mask_rabbit.npy"
Y1_META = ROOT / "evidence" / "taskY1_rabbit_ground_truth" / "data" / "frozen" / "rabbit_mask_metadata.json"

PITCH_UM = 13.0
SPAN_MM = N * PITCH_UM * 1e-3
MS = 1.0
MI = 2.0
K_PHYSICAL = 1.0
TARGET_W_MM = APERTURE_W_MM
TARGET_H_MM = APERTURE_H_MM
OBJECT_LABEL = "original_rabbit"
DK_HIGH = 1090.5510343926662
LENGTHS_MM = (C_LAB_CRYSTAL_LENGTH_MM,)
OBSERVABLE = "I_plus_REDUCED_ORDER_EXPLORATORY"
DEFAULT_OUT = ROOT / "tests" / "z8_rabbit_original_Mi2_dk0_L_sweep"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def object_mask_physical(k: float) -> np.ndarray:
    loaded = load_object_map(Y1_MASK, Y1_META)
    native = np.asarray(loaded["mask"], dtype=np.float64)
    src_x = np.asarray(loaded["x_um"], dtype=np.float64)
    src_y = np.asarray(loaded["y_um"], dtype=np.float64)
    x_obj = axis_um(N, PITCH_UM)
    y_obj = axis_um(N, PITCH_UM)
    xx, yy = np.meshgrid(x_obj, y_obj)
    inv = 1.0 / float(k)
    col = (xx * inv - float(src_x[0])) / float(src_x[1] - src_x[0])
    row = (yy * inv - float(src_y[0])) / float(src_y[1] - src_y[0])
    sampled = map_coordinates(native, [row, col], order=0, mode="constant", cval=0.0)
    return (sampled >= 0.5).astype(np.float64)


def build_setup(L_mm: float | None = None):
    L_mm = float(C_LAB_CRYSTAL_LENGTH_MM if L_mm is None else L_mm)
    if abs(L_mm - C_LAB_CRYSTAL_LENGTH_MM) > 1e-9:
        raise ValueError(f"C_lab workflow supports only L={C_LAB_CRYSTAL_LENGTH_MM} mm, got {L_mm}")
    bundle = build_pilot_setup()
    setup = bundle.effective.overridden(
        crystal_length_mm=L_mm,
        object_map_pixel_um=PITCH_UM,
        pixel_um=PITCH_UM,
        far_field_signal_relay_magnification=MS,
        far_field_idler_relay_magnification=MI,
        object_type=OBJECT_TYPE_TRANSMISSIVE,
        object_mode=OBJECT_MODE_GENERIC_RESEARCH,
        object_device=OBJECT_DEVICE_MASK,
        object_interaction_count=2,
    )
    mapping = resolved_spatial_mapping(setup)
    m_signed = float(mapping["signed_object_to_camera_magnification"])
    return setup, mapping, m_signed


def dk_state_for_case(setup, mode: Literal["zero", "high"]) -> DeltaKCalibrationState:
    dk1 = float(setup_params.delta_k_qpm_per_m_from_setup(setup, interaction=1))
    dk2 = float(setup_params.delta_k_qpm_per_m_from_setup(setup, interaction=2))
    if mode == "zero":
        return DeltaKCalibrationState(mode="calibrated", dk0_theory_ix1=dk1, dk0_theory_ix2=dk2)
    return DeltaKCalibrationState(
        mode="calibrated",
        dk0_theory_ix1=dk1 - DK_HIGH,
        dk0_theory_ix2=dk2 + DK_HIGH,
    )


def rabbit_vs_spot_class(width_ratio: float, height_ratio: float) -> str:
    mx = max(width_ratio, height_ratio)
    if mx > 1.15:
        return "larger_than_spot"
    if mx < 0.85:
        return "smaller_than_spot"
    return "comparable_to_spot"


def radial_marginal_quadrature(
    *,
    n_radial: int,
    q_s_max: float,
    waist_m: float,
    length_m: float,
    delta_k0: float,
    k_p: float,
    k_s: float,
    k_i: float,
    n_pump: int,
    n_theta: int,
) -> np.ndarray:
    q_s = np.linspace(0.0, q_s_max, int(n_radial), dtype=np.float64)
    q_p = np.linspace(0.0, 6.0 / waist_m, int(n_pump), dtype=np.float64)
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False, dtype=np.float64)
    cos_theta = np.cos(theta)[None, None, :]
    qp = q_p[None, :, None]
    qp2 = qp * qp
    pump = np.exp(-0.5 * (waist_m * q_p) ** 2)[None, :, None]
    profile = np.empty_like(q_s)
    batch = 64
    for start in range(0, q_s.size, batch):
        qs = q_s[start : start + batch, None, None]
        qs2 = qs * qs
        qi2 = qp2 + qs2 - 2.0 * qp * qs * cos_theta
        delta_k = delta_k0 - qp2 / (2.0 * k_p) + qs2 / (2.0 * k_s) + qi2 / (2.0 * k_i)
        phase = np.sinc((0.5 * length_m * delta_k) / np.pi) ** 2
        angular = 2.0 * np.pi * np.mean(pump * phase, axis=2)
        profile[start : start + batch] = np.trapz(angular * q_p[None, :], q_p, axis=1)
    peak = float(np.max(profile))
    if not np.isfinite(peak) or peak <= 0.0:
        raise RuntimeError("signal marginal is non-finite or zero")
    return np.clip(profile / peak, 0.0, None)


def generate_scam(setup, L_mm: float, dk0_axial: float) -> np.ndarray:
    crystal = spdc_crystal_state(setup, interaction=1)
    mapping = resolved_spatial_mapping(setup)
    waist_um = float(setup_params.resolved_pump_waist_um(setup, interaction=1))
    lambda_s_m = 810.0e-9
    lambda_p_m = 532.0e-9
    lambda_i_m = 1550.0e-9
    k_p = 2.0 * np.pi * float(crystal["n_pump"]) / lambda_p_m
    k_s = 2.0 * np.pi * float(crystal["n_signal"]) / lambda_s_m
    k_i = 2.0 * np.pi * float(crystal["n_idler"]) / lambda_i_m
    k_s_air = 2.0 * np.pi / lambda_s_m
    f_s_m = float(mapping["far_field_signal_focal_mm"]) * 1e-3
    m_s = abs(float(mapping["far_field_signal_relay_magnification"]))
    q_per_pixel = k_s_air * (PITCH_UM * 1e-6) / (m_s * f_s_m)
    q_s_max = float(0.5 * np.hypot(N - 1, N - 1) * q_per_pixel)
    profile = radial_marginal_quadrature(
        n_radial=N,
        q_s_max=q_s_max,
        waist_m=waist_um * 1e-6,
        length_m=float(L_mm) * 1e-3,
        delta_k0=float(dk0_axial),
        k_p=k_p,
        k_s=k_s,
        k_i=k_i,
        n_pump=72,
        n_theta=96,
    )
    return sample_camera(profile, q_s_max, q_per_pixel)


def ring_stats(img: np.ndarray) -> dict[str, Any]:
    flat = int(np.argmax(img))
    row, col = divmod(flat, img.shape[1])
    yy, xx = np.indices(img.shape, dtype=np.float64)
    r = np.rint(np.hypot(xx - col, yy - row)).astype(np.int32)
    prof = np.zeros(int(r.max()) + 1, dtype=np.float64)
    a = np.asarray(img, dtype=np.float64)
    for ri in range(prof.size):
        m = r == ri
        if np.any(m):
            prof[ri] = float(a[m].mean())
    pk = float(prof.max()) if prof.size else 0.0
    rings = []
    if pk > 0:
        n = prof / pk
        for i in range(2, n.size - 1):
            if n[i] >= n[i - 1] and n[i] >= n[i + 1] and n[i] >= 0.05:
                rings.append({"radius_px": float(i), "height_norm": float(n[i])})
    return {
        "n_rings_ge_5pct": len(rings),
        "strongest_side_ring_height_norm": max((r["height_norm"] for r in rings), default=0.0),
    }


def write_gray16(path: Path, img: np.ndarray, scale: float) -> None:
    u16 = np.clip(img / max(scale, 1e-30) * 65535.0, 0, 65535).astype(np.uint16)
    try:
        import tifffile

        tifffile.imwrite(path, u16, photometric="minisblack")
    except ImportError:
        from PIL import Image

        Image.fromarray(u16).save(path.with_suffix(".png"))


def run_sweep(output: Path | None = None) -> Path:
    hashes_before = hash_protected()
    base = output or DEFAULT_OUT / utc_stamp()
    base.mkdir(parents=True, exist_ok=True)

    mask_rb = object_mask_physical(K_PHYSICAL)
    axis = centered_axis_um(N, PITCH_UM)
    obj_m = measures(mask_rb, PITCH_UM, axis, axis)

    cases: list[dict[str, Any]] = []
    images: dict[str, np.ndarray] = {}
    scam_store: dict[str, np.ndarray] = {}

    for L_mm in LENGTHS_MM:
        for dk_mode, dk_label, dk_axial in (
            ("zero", "dk0_0", 0.0),
            ("high", "dk0_1090", DK_HIGH),
        ):
            setup, mapping, m_signed = build_setup(L_mm)
            scam_key = f"L{int(L_mm)}_{dk_label}"
            dk_state = dk_state_for_case(setup, dk_mode)
            scam_store[scam_key] = generate_scam(setup, L_mm, dk_axial)
            ref_clear = ref_from_T(np.ones((N, N), dtype=np.float32))
            ref_clear["phi"] = np.zeros_like(ref_clear["T"], dtype=np.float32)
            ref_rb = ref_from_T(mask_rb.astype(np.float32))
            ref_rb["phi"] = np.zeros_like(ref_rb["T"], dtype=np.float32)
            no = run_forward_case(setup, ref_clear, dk_state, grid_n=N)
            rb = run_forward_case(setup, ref_rb, dk_state, grid_n=N)
            for tag, img, has_rb in (("no", no["I_plus"], False), ("rabbit", rb["I_plus"], True)):
                key = f"L{int(L_mm)}_{dk_label}_{tag}"
                arr = np.asarray(img, dtype=np.float64)
                images[key] = arr
                w = signal_widths(arr, PITCH_UM)
                rings = ring_stats(arr)
                row: dict[str, Any] = {
                    "case_id": key,
                    "crystal_length_mm": L_mm,
                    "delta_k0_axial_rad_per_m": dk_axial if dk_mode == "high" else 0.0,
                    "delta_k0_mode": dk_mode,
                    "object": "none" if tag == "no" else OBJECT_LABEL,
                    "camera_observable": OBSERVABLE,
                    "Ms": MS,
                    "Mi": MI,
                    "M_signed": m_signed,
                    "object_grid_pitch_um": PITCH_UM,
                    "fwhm_x_px": w["fwhm_x_px"],
                    "fwhm_y_px": w["fwhm_y_px"],
                    "fwhm_radial_diameter_px": w["fwhm_radial_diameter_px"],
                    "fwhm_x_mm": w["fwhm_x_mm"],
                    "fwhm_y_mm": w["fwhm_y_mm"],
                    "fwhm_radial_diameter_mm": w["fwhm_radial_diameter_mm"],
                    **rings,
                }
                if has_rb:
                    cam = project_once(mask_rb, m_signed)
                    cm = measures(cam, PITCH_UM, axis, axis)
                    row["rabbit_projected_width_px"] = cm["aperture_width_px"]
                    row["rabbit_projected_height_px"] = cm["aperture_height_px"]
                    row["rabbit_projected_width_mm"] = cm["aperture_width_mm"]
                    row["rabbit_projected_height_mm"] = cm["aperture_height_mm"]
                    row["rabbit_width_over_fwhm_x"] = cm["aperture_width_px"] / max(w["fwhm_x_px"], 1e-30)
                    row["rabbit_height_over_fwhm_y"] = cm["aperture_height_px"] / max(w["fwhm_y_px"], 1e-30)
                    inside = (
                        cm["aperture_width_px"] <= w["fwhm_x_px"]
                        and cm["aperture_height_px"] <= w["fwhm_y_px"]
                    )
                    row["rabbit_bbox_inside_fwhm_box"] = inside
                    row["rabbit_fits_mostly_inside_spot"] = (
                        row["rabbit_width_over_fwhm_x"] < 1.05 and row["rabbit_height_over_fwhm_y"] < 1.05
                    )
                    row["rabbit_vs_spot_size_class"] = rabbit_vs_spot_class(
                        row["rabbit_width_over_fwhm_x"], row["rabbit_height_over_fwhm_y"]
                    )
                cases.append(row)

    scale = max(float(im.max()) for im in images.values())
    for key, arr in images.items():
        write_gray16(base / f"{key}_GRAY16.tif", arr, scale)
    for sk, scam in scam_store.items():
        np.save(base / f"S_CAM_{sk}.npy", scam)

    montage_order = [
        ("L2_dk0_0_no", "L2_dk0_0_rabbit"),
        ("L2_dk0_1090_no", "L2_dk0_1090_rabbit"),
    ]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
    row_titles = [
        "L=2 mm, dk0=0",
        "L=2 mm, dk0=1090.55",
    ]
    half = 0.5 * SPAN_MM
    extent = [-half, half, -half, half]
    for ri, (pair, rtitle) in enumerate(zip(montage_order, row_titles)):
        for ci, k in enumerate(pair):
            ax = axes[ri, ci]
            ax.imshow(images[k], origin="upper", extent=extent, cmap="gray", vmin=0, vmax=scale, interpolation="nearest")
            ax.set_title(f"{rtitle}\n{'no object' if ci==0 else 'original rabbit'}", fontsize=8)
            ax.set_aspect("equal")
    fig.savefig(base / "MONTAGE_8CASE_I_PLUS_GRAY.png", dpi=140)
    plt.close(fig)

    def pick(substr: str, field: str) -> float:
        for c in cases:
            if c["case_id"].startswith(substr) and c["object"] == "none":
                return float(c[field])
        return float("nan")

    comparisons = {
        "dk0_zero_suppresses_rings_vs_1090": pick("L2_dk0_0", "n_rings_ge_5pct")
        <= pick("L2_dk0_1090", "n_rings_ge_5pct"),
        "original_rabbit_vs_spot_Mi2_L2_dk0_0": next(
            c for c in cases if c["case_id"] == "L2_dk0_0_rabbit"
        )["rabbit_vs_spot_size_class"],
        "crystal_length_mm": C_LAB_CRYSTAL_LENGTH_MM,
    }

    summary = {
        "test_only": True,
        "production_modified": False,
        "physical_rabbit_scale_k": K_PHYSICAL,
        "target_aperture_mm": [TARGET_W_MM, TARGET_H_MM],
        "measured_object_aperture_mm": [obj_m["aperture_width_mm"], obj_m["aperture_height_mm"]],
        "cases": cases,
        "comparisons": comparisons,
        "shared_grayscale_scale_max": scale,
        "invalid_legacy_cache_not_used": True,
    }
    save_json(base / "SWEEP_SUMMARY.json", summary)
    (base / "CONCLUSION.md").write_text(
        "\n".join(
            [
                "# Z8 Mi=2 original rabbit L×Δk₀ sweep (TEST ONLY)",
                "",
                f"- Observable: **{OBSERVABLE}** on 1024×1024 @ {PITCH_UM} µm, Ms={MS}, Mi={MI}.",
                f"- Rabbit: **original physical size** (k={K_PHYSICAL}, ~{TARGET_W_MM:.3f}×{TARGET_H_MM:.3f} mm).",
                f"- Δk₀=0 uses calibrated patch; Δk₀≈{DK_HIGH:.2f} rad/m forced on consumers.",
                "",
                "## Yes/no",
                f"- Δk₀=0 suppresses rings vs 1090? **{comparisons['dk0_zero_suppresses_rings_vs_1090']}**",
                f"- Original rabbit vs spot (L=2 mm, dk0=0): **{comparisons['original_rabbit_vs_spot_Mi2_L2_dk0_0']}**",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    if hash_protected() != hashes_before:
        raise RuntimeError("protected hashes changed")

    print("PRODUCTION_FILES_MODIFIED = NO")
    try:
        ev_disp = base.relative_to(ROOT)
    except ValueError:
        ev_disp = base
    print(f"evidence={ev_disp}")
    print(f"cases={len(cases)} scale_max={scale:.4f}")
    print(f"rings_L10_dk0_0={pick('L10_dk0_0','n_rings_ge_5pct')} L10_dk1090={pick('L10_dk0_1090','n_rings_ge_5pct')}")
    rb = next(c for c in cases if c["case_id"] == "L10_dk0_0_rabbit")
    print(
        f"rabbit/fwhm L10 dk0=0: {rb['rabbit_width_over_fwhm_x']:.3f} x {rb['rabbit_height_over_fwhm_y']:.3f}"
    )
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run_sweep(args.output)


if __name__ == "__main__":
    main()

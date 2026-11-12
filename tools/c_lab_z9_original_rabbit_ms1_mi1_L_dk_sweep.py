"""TEST ONLY — Z9: original Y1 rabbit, Ms=Mi=1, L×Δk₀ sweep with full image products."""
from __future__ import annotations

import argparse
import json
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
MI = 1.0
M_SIGNED_EXPECTED = -(810.0 / 1550.0)
DK_HIGH = 1090.5510343926662
OBSERVABLE = "REDUCED_ORDER_EXPLORATORY_I_PLUS"
EVIDENCE_ROOT = ROOT / "evidence" / "c_lab_z9_original_rabbit_ms1_mi1_L_dk_sweep"

CASE_SPECS: tuple[tuple[str, float, Literal["zero", "high"], bool], ...] = (
    ("A1", C_LAB_CRYSTAL_LENGTH_MM, "zero", False),
    ("A2", C_LAB_CRYSTAL_LENGTH_MM, "zero", True),
    ("A3", C_LAB_CRYSTAL_LENGTH_MM, "high", False),
    ("A4", C_LAB_CRYSTAL_LENGTH_MM, "high", True),
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def object_mask_original() -> np.ndarray:
    loaded = load_object_map(Y1_MASK, Y1_META)
    native = np.asarray(loaded["mask"], dtype=np.float64)
    src_x = np.asarray(loaded["x_um"], dtype=np.float64)
    src_y = np.asarray(loaded["y_um"], dtype=np.float64)
    x_obj = axis_um(N, PITCH_UM)
    y_obj = axis_um(N, PITCH_UM)
    xx, yy = np.meshgrid(x_obj, y_obj)
    col = (xx - float(src_x[0])) / float(src_x[1] - src_x[0])
    row = (yy - float(src_y[0])) / float(src_y[1] - src_y[0])
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
        object_forward_transform="identity",
        object_backward_transform="identity",
    )
    mapping = resolved_spatial_mapping(setup)
    m_signed = float(mapping["signed_object_to_camera_magnification"])
    if abs(m_signed - M_SIGNED_EXPECTED) > 1e-9:
        raise RuntimeError(f"M_signed {m_signed} != expected {M_SIGNED_EXPECTED}")
    if abs(float(mapping["far_field_signal_relay_magnification"]) - MS) > 1e-9:
        raise RuntimeError("Ms != 1")
    if abs(float(mapping["far_field_idler_relay_magnification"]) - MI) > 1e-9:
        raise RuntimeError("Mi != 1")
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
    for start in range(0, q_s.size, 64):
        qs = q_s[start : start + 64, None, None]
        qs2 = qs * qs
        qi2 = qp2 + qs2 - 2.0 * qp * qs * cos_theta
        delta_k = delta_k0 - qp2 / (2.0 * k_p) + qs2 / (2.0 * k_s) + qi2 / (2.0 * k_i)
        phase = np.sinc((0.5 * length_m * delta_k) / np.pi) ** 2
        angular = 2.0 * np.pi * np.mean(pump * phase, axis=2)
        profile[start : start + 64] = np.trapz(angular * q_p[None, :], q_p, axis=1)
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


def signal_peak_px(signal: np.ndarray) -> tuple[float, float]:
    flat = int(np.argmax(signal))
    row, col = divmod(flat, signal.shape[1])
    return float(col), float(row)


def min_enclosing_radius_px(mask: np.ndarray, peak_col: float, peak_row: float) -> float:
    rows, cols = np.where(np.asarray(mask) >= 0.5)
    if cols.size == 0:
        return 0.0
    return float(np.max(np.hypot(cols - peak_col, rows - peak_row)))


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
                rings.append(float(n[i]))
    return {
        "n_rings_ge_5pct": len(rings),
        "strongest_side_ring_height_norm": max(rings, default=0.0),
    }


def write_gray16(path: Path, img: np.ndarray, scale: float) -> None:
    u16 = np.clip(np.asarray(img, dtype=np.float64) / max(scale, 1e-30) * 65535.0, 0, 65535).astype(np.uint16)
    import tifffile

    tifffile.imwrite(path, u16, photometric="minisblack")


def write_preview_png(path: Path, img: np.ndarray, vmax: float, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    half = 0.5 * SPAN_MM
    extent = [-half, half, -half, half]
    fig, ax = plt.subplots(figsize=(5.2, 5.2), constrained_layout=True)
    ax.imshow(img, origin="upper", extent=extent, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=8)
    ax.set_aspect("equal")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_geometry_png(path: Path, signal: np.ndarray, rabbit_cam: np.ndarray, vmax: float, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    half = 0.5 * SPAN_MM
    extent = [-half, half, -half, half]
    fig, ax = plt.subplots(figsize=(5.2, 5.2), constrained_layout=True)
    ax.imshow(signal, origin="upper", extent=extent, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
    ax.contour(rabbit_cam, levels=[0.5], colors="0.85", linewidths=0.7, origin="upper", extent=extent)
    ax.set_title(title, fontsize=8)
    ax.set_aspect("equal")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fits_inside(diameter_px: float, limit_px: float) -> bool:
    return float(diameter_px) <= float(limit_px) + 1e-6


def case_metrics(
    scam: np.ndarray,
    i_plus: np.ndarray,
    rabbit_cam: np.ndarray,
    axis: np.ndarray,
) -> dict[str, Any]:
    sw = signal_widths(scam, PITCH_UM)
    iw = signal_widths(i_plus, PITCH_UM)
    rings = ring_stats(i_plus)
    rm = measures(rabbit_cam, PITCH_UM, axis, axis)
    pk_c, pk_r = signal_peak_px(i_plus)
    enc_r = min_enclosing_radius_px(rabbit_cam, pk_c, pk_r)
    enc_d = 2.0 * enc_r
    cx_mm = rm["centroid_x_mm"]
    cy_mm = rm["centroid_y_mm"]
    center = 0.5 * (N - 1)
    rab_c_px = center + cx_mm * 1e3 / PITCH_UM
    rab_r_px = center - cy_mm * 1e3 / PITCH_UM
    offset_px = float(np.hypot(rab_c_px - pk_c, rab_r_px - pk_r))
    fwhm_r = sw["fwhm_radial_diameter_px"]
    d10 = sw["contour_10pct_diameter_px"]
    d01 = sw["contour_1pct_diameter_px"]
    return {
        **sw,
        "I_plus_fwhm_radial_diameter_px": iw["fwhm_radial_diameter_px"],
        **rings,
        "rabbit_max_row_span_px": rm["aperture_width_px"],
        "rabbit_max_col_span_px": rm["aperture_height_px"],
        "rabbit_bbox_width_px": rm["bbox_width_px"],
        "rabbit_bbox_height_px": rm["bbox_height_px"],
        "rabbit_enclosing_circle_diameter_px": enc_d,
        "rabbit_over_S_cam_FWHM_radial": enc_d / max(fwhm_r, 1e-30),
        "rabbit_over_S_cam_10pct_diameter": enc_d / max(d10, 1e-30),
        "rabbit_over_S_cam_1pct_diameter": enc_d / max(d01, 1e-30),
        "rabbit_centroid_offset_from_I_plus_peak_px": offset_px,
        "rabbit_inside_S_cam_FWHM_radial": fits_inside(enc_d, fwhm_r),
        "rabbit_inside_S_cam_10pct_contour": fits_inside(enc_d, d10),
        "rabbit_inside_S_cam_1pct_contour": fits_inside(enc_d, d01),
        "rabbit_bbox_inside_S_cam_FWHM_box": (
            rm["bbox_width_px"] <= sw["fwhm_x_px"] and rm["bbox_height_px"] <= sw["fwhm_y_px"]
        ),
    }


def run_z9(output: Path | None = None) -> Path:
    hashes_before = hash_protected()
    out = output or EVIDENCE_ROOT / utc_stamp()
    out.mkdir(parents=True, exist_ok=True)

    mask_obj = object_mask_original()
    axis = centered_axis_um(N, PITCH_UM)
    obj_m = measures(mask_obj, PITCH_UM, axis, axis)
    rabbit_cam = project_once(mask_obj, M_SIGNED_EXPECTED)

    scam_cache: dict[str, np.ndarray] = {}
    forward_cache: dict[tuple[float, str, bool], np.ndarray] = {}
    case_rows: list[dict[str, Any]] = []
    i_plus_all: dict[str, np.ndarray] = {}
    scam_by_case: dict[str, np.ndarray] = {}
    scam_vmax = 1.0

    for case_id, L_mm, dk_mode, with_rabbit in CASE_SPECS:
        setup, mapping, m_signed = build_setup(L_mm)
        dk_axial = 0.0 if dk_mode == "zero" else DK_HIGH
        scam_key = f"L{int(L_mm)}_{dk_mode}"
        if scam_key not in scam_cache:
            scam_cache[scam_key] = generate_scam(setup, L_mm, dk_axial)
        scam = scam_cache[scam_key]

        dk_state = dk_state_for_case(setup, dk_mode)
        fkey = (L_mm, dk_mode, with_rabbit)
        if fkey not in forward_cache:
            ref = ref_from_T((mask_obj if with_rabbit else np.ones((N, N))).astype(np.float32))
            ref["phi"] = np.zeros_like(ref["T"], dtype=np.float32)
            out_fwd = run_forward_case(setup, ref, dk_state, grid_n=N)
            forward_cache[fkey] = np.asarray(out_fwd["I_plus"], dtype=np.float64)
        i_plus = forward_cache[fkey]
        i_plus_all[case_id] = i_plus
        scam_by_case[case_id] = scam

        met = case_metrics(scam, i_plus, rabbit_cam, axis)
        row = {
            "case_id": case_id,
            "crystal_length_mm": L_mm,
            "delta_k0_axial_rad_per_m": dk_axial,
            "object": "original_rabbit" if with_rabbit else "none",
            "camera_observable": OBSERVABLE,
            "Ms": MS,
            "Mi": MI,
            "M_signed": m_signed,
            "object_grid_pitch_um": PITCH_UM,
            "camera_pitch_um": PITCH_UM,
            "S_cam_source": f"regenerated_quadrature_{scam_key}",
            **met,
        }
        case_rows.append(row)

    i_vmax = max(float(im.max()) for im in i_plus_all.values())
    for case_id in i_plus_all:
        scam = scam_by_case[case_id]
        i_plus = i_plus_all[case_id]
        np.save(out / f"S_CAM_{case_id}.npy", scam)
        np.save(out / f"I_PLUS_{case_id}.npy", i_plus)
        np.save(out / f"RABBIT_CAM_{case_id}.npy", rabbit_cam)
        write_gray16(out / f"S_CAM_{case_id}.tif", scam, scam_vmax)
        write_gray16(out / f"I_PLUS_{case_id}.tif", i_plus, i_vmax)
        write_preview_png(out / f"S_CAM_{case_id}_preview.png", scam, scam_vmax, case_id)
        write_preview_png(out / f"I_PLUS_{case_id}_preview.png", i_plus, i_vmax, case_id)
        write_geometry_png(
            out / f"GEOMETRY_{case_id}.png",
            i_plus,
            rabbit_cam,
            i_vmax,
            f"{case_id} I_plus + rabbit outline",
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    half = 0.5 * SPAN_MM
    extent = [-half, half, -half, half]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (cid, _, _, _) in zip(axes.ravel(), CASE_SPECS):
        ax.imshow(i_plus_all[cid], origin="upper", extent=extent, cmap="gray", vmin=0, vmax=i_vmax, interpolation="nearest")
        ax.set_title(cid, fontsize=9)
        ax.set_aspect("equal")
    fig.savefig(out / "Z9_ALL_CASES_MONTAGE.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
    pairs = (("A1", "A2"), ("A3", "A4"))
    for axrow, (no_id, rb_id) in zip(axes, pairs):
        for ax, cid in zip(axrow, (no_id, rb_id)):
            ax.imshow(i_plus_all[cid], origin="upper", extent=extent, cmap="gray", vmin=0, vmax=i_vmax, interpolation="nearest")
            ax.set_title(cid, fontsize=8)
            ax.set_aspect("equal")
    fig.savefig(out / "Z9_NO_OBJECT_VS_RABBIT_BY_CASE.png", dpi=130)
    plt.close(fig)

    def row(case_id: str) -> dict[str, Any]:
        return next(r for r in case_rows if r["case_id"] == case_id)

    a2 = row("A2")
    decisions = {
        "crystal_length_mm": C_LAB_CRYSTAL_LENGTH_MM,
        "dk0_0_vs_1090_S_cam_L2": {
            "rings_A1": row("A1")["n_rings_ge_5pct"],
            "rings_A3": row("A3")["n_rings_ge_5pct"],
            "fwhm_radial_A1_px": row("A1")["fwhm_radial_diameter_px"],
            "fwhm_radial_A3_px": row("A3")["fwhm_radial_diameter_px"],
        },
        "original_rabbit_vs_spot_A2": {
            "rabbit_over_FWHM": a2["rabbit_over_S_cam_FWHM_radial"],
            "class": (
                "larger_than_spot"
                if a2["rabbit_over_S_cam_FWHM_radial"] > 1.15
                else "smaller_than_spot"
                if a2["rabbit_over_S_cam_FWHM_radial"] < 0.85
                else "comparable_to_spot"
            ),
        },
        "rabbit_enclosing_fit_by_case": {
            c["case_id"]: {
                "FWHM": c["rabbit_inside_S_cam_FWHM_radial"],
                "10pct": c["rabbit_inside_S_cam_10pct_contour"],
                "1pct": c["rabbit_inside_S_cam_1pct_contour"],
            }
            for c in case_rows
        },
    }

    summary = {
        "test_only": True,
        "production_modified": False,
        "pilot": "c_lab_z9_original_rabbit_ms1_mi1_L_dk_sweep",
        "measured_object_aperture_mm": [obj_m["aperture_width_mm"], obj_m["aperture_height_mm"]],
        "target_object_aperture_mm": [APERTURE_W_MM, APERTURE_H_MM],
        "M_signed_expected": M_SIGNED_EXPECTED,
        "projection": "project_once single inverse map",
        "invalid_legacy_cache_not_used": True,
        "shared_I_plus_scale_max": i_vmax,
        "cases": case_rows,
        "decisions": decisions,
    }
    save_json(out / "Z9_SUMMARY.json", summary)

    (out / "FINAL_DECISION.md").write_text(
        "\n".join(
            [
                "# Z9 FINAL DECISION (TEST ONLY)",
                "",
                f"## a) Signal spot (S_cam, L={C_LAB_CRYSTAL_LENGTH_MM} mm, dk0=0, no object)",
                f"- A1 radial FWHM {row('A1')['fwhm_radial_diameter_px']:.1f} px.",
                "",
                "## b) dk0=0 vs dk0=1090.55 (L=2 mm, no object)",
                f"- I_plus rings ≥5%: A1={row('A1')['n_rings_ge_5pct']}, A3={row('A3')['n_rings_ge_5pct']}.",
                f"- S_cam radial FWHM: A1={row('A1')['fwhm_radial_diameter_px']:.1f} px, A3={row('A3')['fwhm_radial_diameter_px']:.1f} px.",
                "",
                "## c) Ms=Mi=1, original rabbit vs main spot (A2)",
                f"- Enclosing / S_cam radial FWHM = **{a2['rabbit_over_S_cam_FWHM_radial']:.3f}** → **{decisions['original_rabbit_vs_spot_A2']['class']}**.",
                "",
                "## d) Rabbit inside contours (enclosing circle vs S_cam)",
                *[f"- {cid}: FWHM={v['FWHM']}, 10%={v['10pct']}, 1%={v['1pct']}" for cid, v in decisions["rabbit_enclosing_fit_by_case"].items()],
                "",
                f"Observable: **{OBSERVABLE}** — not full experimental crystal→rabbit validation.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    hashes_after = hash_protected()
    if hashes_before != hashes_after:
        raise RuntimeError("protected file hashes changed")
    (out / "PROTECTED_SHA256.json").write_text(
        json.dumps({"before": hashes_before, "after": hashes_after, "unchanged": True}, indent=2) + "\n",
        encoding="utf-8",
    )

    print("PRODUCTION_FILES_MODIFIED = NO")
    try:
        print(f"evidence={out.relative_to(ROOT)}")
    except ValueError:
        print(f"evidence={out}")
    print("case  L  dk0     obj   S_FWHM_r  rab/FWHM  rings")
    for c in case_rows:
        dk = "0" if c["delta_k0_axial_rad_per_m"] == 0 else "1090"
        obj = "rb" if c["object"] != "none" else "no"
        print(
            f"{c['case_id']:3} {int(c['crystal_length_mm']):2} {dk:>4} {obj:>4} "
            f"{c['fwhm_radial_diameter_px']:8.1f} {c['rabbit_over_S_cam_FWHM_radial']:8.3f} "
            f"{c['n_rings_ge_5pct']:5}"
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run_z9(args.output)


if __name__ == "__main__":
    main()

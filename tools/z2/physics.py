"""Z2 physics helpers: moments, registration, metrics, ablations."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates, zoom

from tools.z1.estimator import compute_four_step_visibility, visibility_stats
from tools.z2.common import (
    GRID_N,
    LAMBDA_I_NM,
    NBAR,
    PHASE_LABELS,
    PHASE_RAD,
    PIXEL_UM,
    SEED,
    SENSOR_N,
    STAGE_UM_PROVISIONAL,
)


def ref_from_T(T: np.ndarray) -> dict[str, Any]:
    T = np.asarray(T, dtype=np.float32)
    phi = np.zeros_like(T, dtype=np.float32)
    amp = np.sqrt(np.clip(T, 0, None)).astype(np.float32)
    return {
        "M": T,
        "T": T,
        "phi": phi,
        "O_i": (amp * np.exp(1j * phi)).astype(np.complex64),
        "D": np.zeros_like(T, dtype=np.float32),
        "target_size": int(T.shape[0]),
    }


def physical_geometry(setup, exp_shape: tuple[int, int] | None = None) -> dict[str, float]:
    """Camera-plane 13 um pitch; object pitch via M_QIUP (not YAML 16 um).

    If ``exp_shape`` is given (analysis crop), the simulation object FOV is matched
    to that crop's physical camera-plane extent — not the full 1024 sensor —
    because Figure-5 TIFFs are cropped. Crop is assumed centered on the optical axis.
    """
    from setup_params import qiup_magnification

    M = float(qiup_magnification(setup, lambda_s_nm=810.0, lambda_i_nm=1550.0))
    cam_pitch = float(PIXEL_UM)
    obj_pitch = cam_pitch / max(M, 1e-12)
    if exp_shape is None:
        h = w = SENSOR_N
    else:
        h, w = int(exp_shape[0]), int(exp_shape[1])
    # Match the larger crop dimension so both axes are covered after centering
    cam_fov_um = max(h, w) * cam_pitch
    obj_fov_um = cam_fov_um / max(M, 1e-12)
    sim_dx = obj_fov_um / GRID_N
    return {
        "M_QIUP": M,
        "camera_pitch_um": cam_pitch,
        "object_pitch_for_camera_um": obj_pitch,
        "camera_fov_um": float(cam_fov_um),
        "camera_fov_x_um": float(w * cam_pitch),
        "camera_fov_y_um": float(h * cam_pitch),
        "exp_shape_px": [h, w],
        "object_fov_matching_camera_um": obj_fov_um,
        "sim_grid_n": float(GRID_N),
        "sim_object_dx_um": sim_dx,
        "sim_half_fov_um": 0.5 * obj_fov_um,
        "crop_origin_assumption": "ASSUMED_CENTERED_ON_OPTICAL_AXIS",
    }


def extract_moment_maps(T: np.ndarray, setup_g, state) -> dict[str, Any]:
    """Recompute spatial N1/N2/C12 via the same Michelson+loss path as optics."""
    from optics import (
        apply_signal_path_losses,
        idler_object_amplitude,
        idler_path_transmission_from_setup,
        idler_seed_from_parts,
        michelson_signal_return_amplitude,
        signal_path_transmissions_from_setup,
        spdc_michelson_return_moments,
        spdc_observables_from_moments,
        _optical_intensity,
    )

    ref = ref_from_T(T)
    opt = _optical_intensity(ref, state, 0, np.random.default_rng(SEED))
    vis = dict(opt["vis"])
    gamma = float(vis.get("gamma_path") or vis.get("V_coh") or 1.0)
    T_eff = np.asarray(opt["T_eff"], dtype=np.float64)
    phi_eff = np.asarray(opt["phi_eff"], dtype=np.float64)
    # Prefer idler_return from transfer if available via recompute seed from T_eff
    t_object = idler_object_amplitude(T_eff, phi_eff)
    setup = state.get_setup()
    eta_mode = float(vis.get("eta_mode") or 1.0)
    eta_i = float(vis.get("eta_i_path") or idler_path_transmission_from_setup(setup))
    seed = idler_seed_from_parts(eta_mode, eta_i, t_object)
    nbar1 = float(vis.get("mean_photons_per_mode_NL1") or NBAR)
    nbar2 = float(vis.get("mean_photons_per_mode_NL2") or NBAR)
    src = spdc_michelson_return_moments(
        nbar1,
        nbar2,
        seed["t_seed"],
        michelson_signal_return_amplitude(setup),
        signal_seeds_second_interaction=bool(setup.signal_return_seeds_second_interaction),
    )
    eta_s1, eta_s2 = signal_path_transmissions_from_setup(setup)
    moments = apply_signal_path_losses(
        src["N1_source"], src["N2_source"], src["C12_source"],
        eta_s1=eta_s1, eta_s2=eta_s2, gamma_path=gamma,
    )
    thr = float(vis.get("spectral_filter_throughput", 1.0))
    N1 = np.asarray(moments["N1"], dtype=np.float64) * thr
    N2 = np.asarray(moments["N2"], dtype=np.float64) * thr
    C12 = np.asarray(moments["C12"], dtype=np.complex128) * thr
    obs = spdc_observables_from_moments(N1, N2, C12)
    with np.errstate(divide="ignore", invalid="ignore"):
        B = 2.0 * np.sqrt(np.clip(N1 * N2, 0, None)) / np.where((N1 + N2) > 0, N1 + N2, np.nan)
        Vth = 2.0 * np.abs(C12) / np.where((N1 + N2) > 0, N1 + N2, np.nan)
    return {
        "N1": N1,
        "N2": N2,
        "C12": C12,
        "abs_C12": np.abs(C12),
        "arg_C12": np.angle(C12),
        "g1": obs["intrinsic_g1"],
        "V_theory": Vth,
        "balance": B,
        "N1_raw": np.asarray(src["N1_source"], dtype=np.float64),
        "N2_raw": np.asarray(src["N2_source"], dtype=np.float64),
        "gamma_path": gamma,
        "throughput": thr,
        "opt": opt,
        "I_plus": np.asarray(opt["I_plus"], dtype=np.float64),
    }


def simulate_stacks(T: np.ndarray, setup_g, *, detector_noise: bool = False, seed: int = SEED) -> dict[str, Any]:
    from benchmarks.four_config.runner import quiet_state
    from simulate import simulate_forward

    st = quiet_state(setup_g, nbar=NBAR, num_frames=4, enable_phase_steps=True)
    res = simulate_forward(ref_from_T(T), st, seed=seed, detector_noise=detector_noise)
    frames = [np.asarray(res.frames_plus[i], dtype=np.float64) for i in range(4)]
    vo = dict(res.meta.get("visibility_observables") or {})
    moments = extract_moment_maps(T, setup_g, st)
    four = compute_four_step_visibility(*frames)
    return {
        "frames": frames,
        "vo": vo,
        "moments": moments,
        "four": four,
        "stats": visibility_stats(four["V"], four["valid_mask"]),
        "detector_noise": detector_noise,
    }


def camera_axes_mm(n: int = SENSOR_N, pitch_um: float = PIXEL_UM) -> tuple[np.ndarray, np.ndarray]:
    c = (n - 1) / 2.0
    x = (np.arange(n) - c) * pitch_um / 1000.0
    return x, x.copy()


def object_axes_mm(n: int, dx_um: float) -> tuple[np.ndarray, np.ndarray]:
    c = (n - 1) / 2.0
    x = (np.arange(n) - c) * dx_um / 1000.0
    return x, x.copy()


def exp_camera_axes_mm(shape: tuple[int, int], pitch_um: float = PIXEL_UM) -> tuple[np.ndarray, np.ndarray]:
    """Centered camera-plane axes for an experimental crop (unknown origin → optical-center assumption)."""
    h, w = int(shape[0]), int(shape[1])
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    x = (np.arange(w) - cx) * pitch_um / 1000.0
    y = (np.arange(h) - cy) * pitch_um / 1000.0
    return x, y


def sim_to_exp_camera_grid(
    arr: np.ndarray,
    geom: dict[str, float],
    exp_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample object-plane sim onto the experimental crop camera-plane grid (mm).

    Assumes the experimental analysis crop is centered on the optical axis
    (crop origin not independently available; documented as ASSUMED_CENTERED).
    """
    from scipy.interpolate import RegularGridInterpolator

    M = geom["M_QIUP"]
    dx_obj = geom["sim_object_dx_um"]
    xo, yo = object_axes_mm(arr.shape[1], dx_obj)
    # object -> camera: x_cam = M * x_obj
    xc_src = M * xo
    yc_src = M * yo
    # Ensure increasing axes for interpolator
    if xc_src[0] > xc_src[-1]:
        xc_src = xc_src[::-1]
        arr = arr[:, ::-1]
    if yc_src[0] > yc_src[-1]:
        yc_src = yc_src[::-1]
        arr = arr[::-1, :]
    xc, yc = exp_camera_axes_mm(exp_shape)
    f = RegularGridInterpolator(
        (yc_src, xc_src), arr.astype(np.float64), bounds_error=False, fill_value=np.nan
    )
    YY, XX = np.meshgrid(yc, xc, indexing="ij")
    out = f(np.stack([YY.ravel(), XX.ravel()], axis=-1)).reshape(exp_shape)
    return out, xc, yc


def common_roi_mask(V_exp: np.ndarray, V_sim_cam: np.ndarray, *, edge_frac: float = 0.08) -> np.ndarray:
    """Frozen ROI: overlap of finite pixels, excluding edge margin."""
    m = np.isfinite(V_exp) & np.isfinite(V_sim_cam)
    h, w = m.shape
    ey, ex = max(1, int(h * edge_frac)), max(1, int(w * edge_frac))
    m[:ey, :] = False
    m[-ey:, :] = False
    m[:, :ex] = False
    m[:, -ex:] = False
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r2 = ((yy - cy) / (0.45 * h)) ** 2 + ((xx - cx) / (0.45 * w)) ** 2
    m &= r2 <= 1.0
    return m


def map_metrics(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        m &= mask
    aa, bb = a[m], b[m]
    if aa.size < 10:
        return {
            "NRMSE": float("nan"),
            "NCC": float("nan"),
            "SSIM": float("nan"),
            "RMSE": float("nan"),
            "residual_mean": float("nan"),
            "n": 0,
        }
    rmse = float(np.sqrt(np.mean((aa - bb) ** 2)))
    nrmse = rmse / (float(np.sqrt(np.mean(aa ** 2))) + 1e-15)
    aa0, bb0 = aa - aa.mean(), bb - bb.mean()
    ncc = float(np.sum(aa0 * bb0) / (np.sqrt(np.sum(aa0 ** 2) * np.sum(bb0 ** 2)) + 1e-15))
    c1, c2 = 1e-4, 9e-4
    ssim = float(
        (2 * aa.mean() * bb.mean() + c1)
        * (2 * np.cov(aa, bb)[0, 1] + c2)
        / ((aa.mean() ** 2 + bb.mean() ** 2 + c1) * (aa.var() + bb.var() + c2) + 1e-15)
    )
    return {
        "NRMSE": nrmse,
        "NCC": ncc,
        "SSIM": ssim,
        "RMSE": rmse,
        "residual_mean": float(np.mean(aa - bb)),
        "n": int(aa.size),
    }


def estimate_background_ratio(frames: list[np.ndarray], *, edge: int = 40) -> dict[str, float]:
    """Estimate peripheral ADU / central ADU from experimental frames (no rabbit fit)."""
    ratios = []
    bg_vals = []
    sig_vals = []
    for I in frames:
        I = np.asarray(I, float)
        h, w = I.shape
        border = np.concatenate(
            [
                I[:edge, :].ravel(),
                I[-edge:, :].ravel(),
                I[:, :edge].ravel(),
                I[:, -edge:].ravel(),
            ]
        )
        cy0, cy1 = h // 2 - h // 8, h // 2 + h // 8
        cx0, cx1 = w // 2 - w // 8, w // 2 + w // 8
        center = I[cy0:cy1, cx0:cx1].ravel()
        bg = float(np.median(border))
        sig = float(np.median(center))
        bg_vals.append(bg)
        sig_vals.append(sig)
        ratios.append(bg / (sig + 1e-15))
    return {
        "bg_median_adu": float(np.mean(bg_vals)),
        "signal_median_adu": float(np.mean(sig_vals)),
        "bg_to_signal_ratio": float(np.mean(ratios)),
        "method": "peripheral_median_over_central_median",
        "rabbit_used": False,
    }


def apply_background_to_frames(frames: list[np.ndarray], bg_frac: float) -> list[np.ndarray]:
    """Add incoherent DC = bg_frac * mean(frame) to each phase frame."""
    out = []
    for I in frames:
        I = np.asarray(I, float)
        B = bg_frac * float(np.nanmean(I[np.isfinite(I)]))
        out.append(I + B)
    return out


def four_step_phase_schedule_rows(series: str) -> list[dict[str, Any]]:
    rows = []
    for lab, phi, dz in zip(PHASE_LABELS, PHASE_RAD, STAGE_UM_PROVISIONAL):
        dphi = 4.0 * np.pi * dz / LAMBDA_I_NM  # um/um? lambda in nm: convert
        # Delta_phi = 4*pi*dz/lambda with consistent units: dz_um, lambda_um
        lam_um = LAMBDA_I_NM / 1000.0
        dphi = 4.0 * np.pi * (dz / lam_um)
        rows.append(
            {
                "case": series,
                "frame": lab,
                "stage_position_um": dz,
                "relative_displacement_um": dz,
                "assigned_phase_rad": phi,
                "assigned_phase_wrapped_rad": float(np.mod(phi, 2 * np.pi)),
                "double_pass_phase_from_dz_rad": dphi,
                "double_pass_phase_wrapped_rad": float(np.mod(dphi, 2 * np.pi)),
                "phase_source": "PROVISIONAL_folder_label_mapped_to_25um_steps",
                "status": "PROVISIONAL",
            }
        )
    return rows

"""Independent reference equations for Configuration D verification.

Every function here is written from the governing physical equations and
uses only the Python standard library and NumPy. The module must never import
``setup_params``, ``optics``, ``simulate``, ``latents``, ``detector``,
``provenance`` or the generated ``qiup_digital_twin_full`` module; this is
enforced by ``tests/test_independent_physics_validation_d.py``.

Conventions
-----------
* Wavelengths in nm unless the name says ``_um``; lengths in µm or m as named.
* Two-crystal induced coherence (Zou-Wang-Mandel): NL1 emits (s1, i1); i1 is
  transmitted once through the object t (and relay losses) and seeds NL2,
  which emits s2. ``C12 = <s2^dagger s1>``.
* Combiner: out± = (d1 ± exp(-iθ) d2)/√2, so
  I± = (N1+N2)/2 ± Re[C12 exp(iθ)].
"""
from __future__ import annotations

import cmath
import math
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# Material constants (literature values, transcribed independently)
# --------------------------------------------------------------------------
# Kato & Takaoka, Appl. Opt. 41, 5040 (2002): KTP n_z Sellmeier at 20 °C and
# the z-axis thermo-optic coefficient. λ in µm.
KTP_NZ_SELLMEIER = (4.59423, 0.06206, 0.04763, 110.80672, 86.12171)
KTP_DNZ_DT = (0.9221, -2.9220, 3.6677, -0.1897)  # × 1e-5 /°C, powers λ^-3..λ^0
KTP_SELLMEIER_REFERENCE_C = 20.0
# Linear thermal expansion of KTP along the poling (x) axis.
KTP_THERMAL_EXPANSION_PER_C = 6.7e-6


# --------------------------------------------------------------------------
# 7.1 Pump-power distribution
# --------------------------------------------------------------------------
def fluctuation_factor(relative_fluctuation: float) -> float:
    """Pump power cannot become negative: (1 + dP)_+."""
    return max(1.0 + float(relative_fluctuation), 0.0)


def branched_pump_split(
    laser_mW: float,
    reflectance: float,
    eta_pre: float = 1.0,
    eta_nl2: float = 1.0,
    relative_fluctuation: float = 0.0,
) -> tuple[float, float]:
    """Passive beam splitter: NL1 receives R·η_pre, NL2 receives (1−R)·η_NL2."""
    p = float(laser_mW) * fluctuation_factor(relative_fluctuation)
    return (
        float(reflectance) * float(eta_pre) * p,
        (1.0 - float(reflectance)) * float(eta_nl2) * p,
    )


def sequential_pump_split(
    laser_mW: float,
    eta_pre: float = 1.0,
    eta_between: float = 1.0,
    relative_fluctuation: float = 0.0,
) -> tuple[float, float]:
    """Pump transmitted through NL1 and then NL2: P2 = η_between·P1."""
    p1 = float(eta_pre) * float(laser_mW) * fluctuation_factor(relative_fluctuation)
    return p1, float(eta_between) * p1


# --------------------------------------------------------------------------
# 7.2 SPDC generation and two-source moments
# --------------------------------------------------------------------------
def gain_from_pump(k_per_sqrt_mW: float, pump_mW: float) -> float:
    """Phase-matched single-pass gain r0 = k·√P (k in 1/√mW, P in mW)."""
    return float(k_per_sqrt_mW) * math.sqrt(max(float(pump_mW), 0.0))


def closed_form_nbar(kappa_per_m: float, delta_k_per_m: float, length_m: float) -> float:
    """Mean photon number per mode of a phase-mismatched parametric amplifier.

    nbar = κ² |sinh(gL)/g|²,   g = √(κ² − (Δk/2)²)  (complex square root).
    The complex root covers the hyperbolic (g real) and oscillatory
    (g imaginary) branches with a single expression; g → 0 gives (κL)².
    """
    k = float(kappa_per_m)
    L = float(length_m)
    g = cmath.sqrt(k * k - (0.5 * float(delta_k_per_m)) ** 2)
    if abs(g) * L < 1e-9:
        return (k * L) ** 2
    return float(abs(k * cmath.sinh(g * L) / g) ** 2)


def coupled_mode_bogoliubov(
    kappa_per_m: float,
    delta_k_per_m: float,
    length_m: float,
    steps: int = 4000,
) -> dict[str, float]:
    """Integrate the undepleted-pump coupled-mode equations with RK4.

    d/dz [a_s, a_i†]ᵀ = [[0, iκ e^{iΔk z}], [−iκ e^{−iΔk z}, 0]] [a_s, a_i†]ᵀ.
    With a_s(L) = U a_s(0) + V a_i†(0), nbar = |V|² and |U|² − |V|² = 1.
    """
    k = float(kappa_per_m)
    dk = float(delta_k_per_m)
    L = float(length_m)
    n = int(steps)
    h = L / n

    def rhs(z: float, phi: np.ndarray) -> np.ndarray:
        m = np.array(
            [[0.0, 1j * k * np.exp(1j * dk * z)],
             [-1j * k * np.exp(-1j * dk * z), 0.0]],
            dtype=np.complex128,
        )
        return m @ phi

    phi = np.eye(2, dtype=np.complex128)
    z = 0.0
    for _ in range(n):
        k1 = rhs(z, phi)
        k2 = rhs(z + 0.5 * h, phi + 0.5 * h * k1)
        k3 = rhs(z + 0.5 * h, phi + 0.5 * h * k2)
        k4 = rhs(z + h, phi + h * k3)
        phi = phi + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        z += h
    U, V = phi[0, 0], phi[0, 1]
    return {
        "nbar": float(abs(V) ** 2),
        "unitarity": float(abs(U) ** 2 - abs(V) ** 2),
    }


def _mode_vector(size: int, index: int, value: complex) -> np.ndarray:
    out = np.zeros(size, dtype=np.complex128)
    out[index] = value
    return out


def two_crystal_bogoliubov_moments(
    n1: float,
    n2: float,
    t_seed: complex,
    eta_s1: float = 1.0,
    eta_s2: float = 1.0,
) -> dict[str, Any]:
    """Two-crystal moments from vacuum-mode Bogoliubov coefficients.

    Vacuum input modes: [s1, i1, s2, idler_loss, det1_loss, det2_loss].
    Each output operator is written as A = α·m + β·m†. For the vacuum input,
    <A† B> = Σ conj(β_A) β_B and [A, B†] = Σ α_A conj(α_B) − β_A conj(β_B).
    """
    size = 6
    S1, I1, S2, LI, D1, D2 = range(size)
    u1, v1 = math.sqrt(1.0 + n1), math.sqrt(n1)
    u2, v2 = math.sqrt(1.0 + n2), math.sqrt(n2)
    t = complex(t_seed)
    leak = math.sqrt(max(0.0, 1.0 - abs(t) ** 2))

    # NL1: a_s1 = u1 s1 + v1 i1†,  a_i1 = u1 i1 + v1 s1†
    a_s1 = (_mode_vector(size, S1, u1), _mode_vector(size, I1, v1))
    a_i1 = (_mode_vector(size, I1, u1), _mode_vector(size, S1, v1))
    # Object (single pass) + idler loss: i' = t a_i1 + √(1−|t|²) l
    i_alpha = t * a_i1[0] + _mode_vector(size, LI, leak)
    i_beta = t * a_i1[1]
    # i'† swaps and conjugates the coefficient vectors.
    idag_alpha, idag_beta = np.conjugate(i_beta), np.conjugate(i_alpha)
    # NL2 seeded by the returning idler: a_s2 = u2 s2 + v2 i'†
    a_s2 = (
        _mode_vector(size, S2, u2) + v2 * idag_alpha,
        v2 * idag_beta,
    )
    # Detector-side signal transmissions with vacuum admixture.
    d1 = (
        math.sqrt(eta_s1) * a_s1[0] + _mode_vector(size, D1, math.sqrt(1.0 - eta_s1)),
        math.sqrt(eta_s1) * a_s1[1],
    )
    d2 = (
        math.sqrt(eta_s2) * a_s2[0] + _mode_vector(size, D2, math.sqrt(1.0 - eta_s2)),
        math.sqrt(eta_s2) * a_s2[1],
    )

    def commutator(a, b) -> complex:
        return complex(np.vdot(b[0], a[0]) - np.vdot(b[1], a[1]))

    return {
        "N1": float(np.vdot(d1[1], d1[1]).real),
        "N2": float(np.vdot(d2[1], d2[1]).real),
        "C12": complex(np.vdot(d2[1], d1[1])),
        "commutator_d1": commutator(d1, d1),
        "commutator_d2": commutator(d2, d2),
        "commutator_d1_d2": commutator(d1, d2),
        "operators": {"d1": d1, "d2": d2},
    }


def combiner_intensities_operator(moments: dict[str, Any], theta: float) -> tuple[float, float]:
    """I± from the operator form out± = (d1 ± e^{−iθ} d2)/√2."""
    d1 = moments["operators"]["d1"]
    d2 = moments["operators"]["d2"]
    phase = cmath.exp(-1j * float(theta))
    plus = (d1[1] + phase * d2[1]) / math.sqrt(2.0)
    minus = (d1[1] - phase * d2[1]) / math.sqrt(2.0)
    return float(np.vdot(plus, plus).real), float(np.vdot(minus, minus).real)


def combiner_intensities(N1: float, N2: float, C12: complex, theta: float) -> tuple[float, float]:
    cross = (complex(C12) * cmath.exp(1j * float(theta))).real
    mid = 0.5 * (float(N1) + float(N2))
    return mid + cross, mid - cross


def fixed_combiner_visibility(N1: float, N2: float, C12: complex) -> float:
    total = float(N1) + float(N2)
    return 0.0 if total <= 0.0 else 2.0 * abs(complex(C12)) / total


def reduced_order_detected_moments(
    n1: float,
    n2: float,
    t_seed: complex,
    *,
    eta_s1: float = 1.0,
    eta_s2: float = 1.0,
    coherence: float = 1.0,
    throughput: float = 1.0,
) -> dict[str, float | complex]:
    """Bogoliubov moments followed by the declared reduced-order scalars.

    The spectral filter throughput multiplies N1, N2 and C12 once and the
    spectral coherence magnitude multiplies C12 once.
    """
    m = two_crystal_bogoliubov_moments(n1, n2, t_seed, eta_s1, eta_s2)
    return {
        "N1": throughput * m["N1"],
        "N2": throughput * m["N2"],
        "C12": throughput * coherence * m["C12"],
    }


# --------------------------------------------------------------------------
# Spectral filter (Gaussian source × Gaussian filter)
# --------------------------------------------------------------------------
def sigma_from_fwhm(fwhm: float) -> float:
    return float(fwhm) / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def gaussian_filter_throughput_analytic(
    source_center_nm: float,
    source_fwhm_nm: float,
    filter_center_nm: float,
    filter_fwhm_nm: float,
) -> float:
    """∫S·T / ∫S for a unit-peak Gaussian filter T and Gaussian spectrum S."""
    ss = sigma_from_fwhm(source_fwhm_nm)
    sf = sigma_from_fwhm(filter_fwhm_nm)
    var = ss * ss + sf * sf
    d = float(filter_center_nm) - float(source_center_nm)
    return sf / math.sqrt(var) * math.exp(-d * d / (2.0 * var))


def spectral_coherence_quadrature(
    source_center_nm: float,
    source_fwhm_nm: float,
    filter_center_nm: float,
    filter_fwhm_nm: float,
    opd_nm: float,
    samples: int = 40001,
) -> dict[str, float]:
    """Filtered spectral coherence ∫S T e^{i2π OPD(1/λ − 1/λ0)} / ∫S T.

    Independent wavelength grid (±12σ of the filtered spectrum) and a
    hand-written trapezoid rule.
    """
    ss = sigma_from_fwhm(source_fwhm_nm)
    sf = sigma_from_fwhm(filter_fwhm_nm)
    lo = min(source_center_nm, filter_center_nm) - 12.0 * max(ss, sf)
    hi = max(source_center_nm, filter_center_nm) + 12.0 * max(ss, sf)
    lam = np.linspace(max(lo, 1e-3), hi, int(samples))
    w = np.full(lam.size, lam[1] - lam[0])
    w[0] *= 0.5
    w[-1] *= 0.5
    src = np.exp(-0.5 * ((lam - source_center_nm) / ss) ** 2)
    filt = np.exp(-0.5 * ((lam - filter_center_nm) / sf) ** 2)
    weighted = src * filt
    phase = 2.0 * math.pi * float(opd_nm) * (1.0 / lam - 1.0 / float(source_center_nm))
    coh = np.sum(w * weighted * np.exp(1j * phase)) / np.sum(w * weighted)
    return {
        "throughput": float(np.sum(w * weighted) / np.sum(w * src)),
        "coherence_magnitude": float(abs(coh)),
        "coherence_phase_rad": float(cmath.phase(coh)),
    }


def carrier_phase_rad(opd_mm: float, lambda_s_nm: float) -> float:
    """Interferometer carrier phase 2π·OPD/λ_s."""
    return 2.0 * math.pi * float(opd_mm) * 1.0e6 / float(lambda_s_nm)


# --------------------------------------------------------------------------
# 7.5 Quasi-phase matching (Type-0 ppKTP, collinear)
# --------------------------------------------------------------------------
def ktp_nz(lambda_um: float, temperature_C: float) -> float:
    a, b, c, d, e = KTP_NZ_SELLMEIER
    l2 = float(lambda_um) ** 2
    n20 = math.sqrt(a + b / (l2 - c) + d / (l2 - e))
    p3, p2, p1, p0 = KTP_DNZ_DT
    lam = float(lambda_um)
    dndt = (p3 / lam ** 3 + p2 / lam ** 2 + p1 / lam + p0) * 1.0e-5
    return n20 + dndt * (float(temperature_C) - KTP_SELLMEIER_REFERENCE_C)


def poling_period_at_temperature(period0_um: float, temperature_C: float, reference_C: float) -> float:
    return float(period0_um) * (
        1.0 + KTP_THERMAL_EXPANSION_PER_C * (float(temperature_C) - float(reference_C))
    )


def idler_wavelength_nm(lambda_p_nm: float, lambda_s_nm: float) -> float:
    """Energy conservation 1/λ_i = 1/λ_p − 1/λ_s."""
    return 1.0 / (1.0 / float(lambda_p_nm) - 1.0 / float(lambda_s_nm))


def delta_k_per_m(
    lambda_p_nm: float,
    lambda_s_nm: float,
    lambda_i_nm: float,
    temperature_C: float,
    period_um: float,
    order: int = 1,
    backward: bool = False,
) -> float:
    """Δk = k_p − k_s − k_i − 2πm/Λ  [rad/m]; backward propagation flips sign."""
    lp, ls, li = (float(x) * 1.0e-3 for x in (lambda_p_nm, lambda_s_nm, lambda_i_nm))
    T = float(temperature_C)
    dk_um = 2.0 * math.pi * (
        ktp_nz(lp, T) / lp - ktp_nz(ls, T) / ls - ktp_nz(li, T) / li
        - int(order) / float(period_um)
    )
    return (-1.0 if backward else 1.0) * dk_um * 1.0e6


def phase_matched_period_um(
    lambda_p_nm: float, lambda_s_nm: float, lambda_i_nm: float,
    temperature_C: float, order: int = 1,
) -> float:
    lp, ls, li = (float(x) * 1.0e-3 for x in (lambda_p_nm, lambda_s_nm, lambda_i_nm))
    T = float(temperature_C)
    return int(order) / (ktp_nz(lp, T) / lp - ktp_nz(ls, T) / ls - ktp_nz(li, T) / li)


def phase_matching_temperature_C(
    lambda_p_nm: float, lambda_s_nm: float, lambda_i_nm: float,
    period0_um: float, period_reference_C: float,
    lo_C: float = 0.0, hi_C: float = 200.0, order: int = 1,
) -> float:
    """Bisection for Δk(T) = 0 including thermal expansion of Λ."""
    def f(T: float) -> float:
        return delta_k_per_m(
            lambda_p_nm, lambda_s_nm, lambda_i_nm, T,
            poling_period_at_temperature(period0_um, T, period_reference_C), order,
        )
    a, b = float(lo_C), float(hi_C)
    fa, fb = f(a), f(b)
    if fa * fb > 0.0:
        raise ValueError("phase-matching temperature is not bracketed")
    for _ in range(200):
        m = 0.5 * (a + b)
        fm = f(m)
        if fa * fm <= 0.0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


# --------------------------------------------------------------------------
# 7.4 Near-field relay mapping
# --------------------------------------------------------------------------
def near_field_index_scale(m_u: float, m_s: float, p_object_um: float, p_camera_um: float) -> float:
    """ρ_c = (M_s/M_u) ρ_o expressed in array indices."""
    return (float(m_s) / float(m_u)) * float(p_object_um) / float(p_camera_um)


def bilinear_edge_clamped(
    field: np.ndarray,
    input_factor: float,
    center: tuple[float, float] | None = None,
) -> np.ndarray:
    """Sample ``field`` at input = center + input_factor·(output − center).

    Separable linear interpolation; coordinates outside the array are
    clamped to the nearest edge sample.
    """
    src = np.asarray(field)
    h, w = src.shape
    cy, cx = ((h - 1) / 2.0, (w - 1) / 2.0) if center is None else center
    f = float(input_factor)
    ys = np.clip(cy + f * (np.arange(h) - cy), 0.0, h - 1.0)
    xs = np.clip(cx + f * (np.arange(w) - cx), 0.0, w - 1.0)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    top = src[np.ix_(y0, x0)] * (1.0 - wx) + src[np.ix_(y0, x1)] * wx
    bottom = src[np.ix_(y1, x0)] * (1.0 - wx) + src[np.ix_(y1, x1)] * wx
    return top * (1.0 - wy) + bottom * wy


def gaussian_blur_reflect(field: np.ndarray, sigma_px: float, truncate: float = 4.0) -> np.ndarray:
    """Separable normalized Gaussian with half-sample symmetric boundaries."""
    out = np.asarray(field, dtype=np.complex128)
    s = float(sigma_px)
    if s <= 0.0:
        return out.copy()
    radius = int(truncate * s + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / s) ** 2)
    kernel /= kernel.sum()
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="symmetric")
        acc = np.zeros_like(out)
        n = out.shape[axis]
        for j, kj in enumerate(kernel):
            sl = [slice(None), slice(None)]
            sl[axis] = slice(j, j + n)
            acc += kj * padded[tuple(sl)]
        out = acc
    return out


def p07_visibility_half_width_um(length_um: float, lambda_d_um: float, lambda_u_um: float, waist_um: float) -> float:
    """P07 Eq. (4.9): near-field visibility 1/e half-width at the crystal."""
    lsum = lambda_d_um + lambda_u_um
    a = 2.0 * math.pi * waist_um ** 2 * lsum
    return (
        math.sqrt(length_um * lsum / (4.0 * math.pi))
        * a * math.sqrt(1.0 + lambda_d_um ** 2 * length_um / a)
        / abs(a - lambda_d_um * lambda_u_um * length_um)
    )


def p10_joint_gaussian(length_um: float, lambda_d_um: float, lambda_u_um: float, waist_um: float) -> dict[str, float]:
    """P10 Eq. (9): P ∝ exp[−A|λu ρd + λd ρu|² − B|ρd − ρu|²]."""
    lsum = lambda_d_um + lambda_u_um
    A = 2.0 / (waist_um ** 2 * lsum ** 2)
    B = 4.0 * math.pi / (length_um * lsum)
    alpha = A * lambda_d_um ** 2 + B
    beta = A * lambda_u_um ** 2 + B
    gamma = A * lambda_d_um * lambda_u_um - B
    return {
        "A": A, "B": B, "alpha": alpha, "beta": beta, "gamma": gamma,
        "conditional_scale": -gamma / alpha,
        "envelope_coefficient": beta - gamma * gamma / alpha,
        "sigma_um": 1.0 / math.sqrt(2.0 * alpha),
    }


def p10_near_field_transfer(
    object_field: np.ndarray,
    *,
    m_u: float,
    m_s: float,
    p_object_um: float,
    p_camera_um: float,
    length_um: float,
    lambda_d_um: float,
    lambda_u_um: float,
    waist_um: float,
    pump_x_um: float = 0.0,
    pump_y_um: float = 0.0,
) -> dict[str, np.ndarray]:
    """Relay mapping followed by the P10 conditional joint-Gaussian response.

    t_V = G/∫P with G = ∫P t dρo; the marginal ∫P (peak-normalized) weights
    the detected intensity. A unit object therefore gives t_V = 1.
    """
    field = np.asarray(object_field, dtype=np.complex128)
    scale = near_field_index_scale(m_u, m_s, p_object_um, p_camera_um)
    mapped = bilinear_edge_clamped(field, 1.0 / scale)
    jg = p10_joint_gaussian(length_um, lambda_d_um, lambda_u_um, waist_um)
    p_x = p_camera_um / abs(m_s)
    sigma_px = jg["sigma_um"] / p_x
    h, w = field.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    yd = (yy - (h - 1) / 2.0) * p_x - pump_y_um
    xd = (xx - (w - 1) / 2.0) * p_x - pump_x_um
    envelope = np.exp(-jg["envelope_coefficient"] * (xd * xd + yd * yd))
    center = ((h - 1) / 2.0 + pump_y_um / p_x, (w - 1) / 2.0 + pump_x_um / p_x)

    def joint(src: np.ndarray) -> np.ndarray:
        return bilinear_edge_clamped(
            gaussian_blur_reflect(src, sigma_px), jg["conditional_scale"], center
        ) * envelope

    num = joint(mapped)
    den = np.real(joint(np.ones_like(mapped)))
    return {
        "mapped": mapped,
        "t_V": num / den,
        "marginal_weight": den / den.max(),
        "sigma_px": np.asarray(sigma_px),
    }


# --------------------------------------------------------------------------
# Object / SLM helpers and harmonic analysis
# --------------------------------------------------------------------------
def slm_grid_phase(level: int, levels: int = 256, phi_max: float = 2.0 * math.pi) -> float:
    """Phase exactly representable by an L-level SLM (no quantization error)."""
    return float(level) * float(phi_max) / float(levels)


def harmonic_fit(phases: np.ndarray, values: np.ndarray) -> dict[str, float]:
    """Least-squares a0 + Σ_{n=1,2} (a_n cos nφ + b_n sin nφ)."""
    p = np.asarray(phases, dtype=np.float64)
    design = np.column_stack(
        [np.ones_like(p), np.cos(p), np.sin(p), np.cos(2 * p), np.sin(2 * p)]
    )
    coef, *_ = np.linalg.lstsq(design, np.asarray(values, dtype=np.float64), rcond=None)
    a0, a1, b1, a2, b2 = (float(c) for c in coef)
    return {
        "a0": a0,
        "first_harmonic": math.hypot(a1, b1),
        "first_harmonic_phase": math.atan2(-b1, a1),
        "second_harmonic": math.hypot(a2, b2),
    }


def intensity_centroid(image: np.ndarray) -> tuple[float, float]:
    img = np.asarray(image, dtype=np.float64)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    total = img.sum()
    return float((yy * img).sum() / total), float((xx * img).sum() / total)

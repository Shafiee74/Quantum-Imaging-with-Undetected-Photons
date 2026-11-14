"""Two-crystal near-field independent physics checks (historical module name).

Module path ``verification.config_d_checks`` is retained for freeze stability.
After the 2026-10-19 manuscript letter migration, the physical profile under
test is **manuscript B** = registry ``two_crystal_near_field``
(legacy evidence letter was ``D``). Manuscript letter ``D`` is now
single-crystal near-field and is **not** the target of these checks.

Each check evaluates the production simulator and compares it with an
expected value computed only from ``verification.independent_reference``.
Configuration inputs (wavelengths, lengths, temperatures, magnifications,
filter widths) are read from the resolved setup; no production physics
function is used to form an expected value.

Calibration values labelled ``SOFTWARE_TEST_ONLY`` are synthetic software
test inputs. They are not laboratory calibrations.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from latents import LatentState
from optics import (
    _interfere_from_moments,
    apply_signal_path_losses,
    architecture_object_transfer,
    signal_filter_spectral_response,
    spdc_double_pass_moments,
    spdc_nbar_from_r0_L_dk,
)
from setup_params import (
    delta_k_qpm_per_m_from_setup,
    delta_k_qpm_per_um,
    ktp_nz,
    near_field_joint_gaussian_coefficients,
    near_field_visibility_1e_um,
    phase_match_poling_period_um,
    pump_powers_from_setup,
    qpm_poling_period_um,
    resolve_forward,
    resolve_setup_selection,
    resolved_spatial_mapping,
)
from simulate import simulate_forward
from verification import independent_reference as ref

SEED = 20260926
# Stable physical identity (manuscript B). Do not use letter "D" here:
# manuscript D is single_crystal_near_field after the B↔D letter migration.
PROFILE = "two_crystal_near_field"
MANUSCRIPT_LETTER = "B"
LEGACY_EVIDENCE_LETTER = "D"
SOFTWARE_TEST_K_PER_SQRT_MW = 0.05
SOFTWARE_TEST_CALIBRATION_ID = "SOFTWARE_TEST_ONLY_NOT_A_LAB_CALIBRATION"

AREA_PUMP = "7.1 pump-power distribution"
AREA_MOMENTS = "7.2 two-source SPDC moments"
AREA_OBJECT = "7.3 single-pass transmissive object"
AREA_MAPPING = "7.4 near-field relay mapping"
AREA_QPM = "7.5 QPM / temperature"
AREA_INTERFERENCE = "7.6 interference output"
AREA_INVARIANT = "8 physical invariants"
AREAS = (
    AREA_PUMP, AREA_MOMENTS, AREA_OBJECT, AREA_MAPPING,
    AREA_QPM, AREA_INTERFERENCE, AREA_INVARIANT,
)


@dataclass
class CheckRecord:
    area: str
    test: str
    reference: str
    parameters: dict[str, Any]
    expected: Any
    observed: Any
    max_abs_error: float
    tolerance: str
    status: str
    seed: int | None = SEED
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration": PROFILE,
            "area": self.area,
            "test": self.test,
            "reference": self.reference,
            "parameters": self.parameters,
            "expected": self.expected,
            "observed": self.observed,
            "max_abs_error": self.max_abs_error,
            "tolerance": self.tolerance,
            "status": self.status,
            "seed": self.seed,
            "notes": self.notes,
        }


@dataclass
class Registry:
    records: list[CheckRecord] = field(default_factory=list)

    def compare(
        self,
        area: str,
        test: str,
        reference: str,
        expected: Any,
        observed: Any,
        *,
        atol: float = 0.0,
        rtol: float = 0.0,
        parameters: dict[str, Any] | None = None,
        notes: str = "",
    ) -> bool:
        exp = np.asarray(expected, dtype=np.complex128)
        obs = np.asarray(observed, dtype=np.complex128)
        if exp.ndim == 0 and obs.ndim > 0:
            exp = np.broadcast_to(exp, obs.shape)
        if exp.shape != obs.shape:
            ok = False
            err = float("inf")
        else:
            diff = np.abs(obs - exp)
            err = float(np.max(diff)) if diff.size else 0.0
            ok = bool(np.all(np.isfinite(obs)) and np.all(diff <= atol + rtol * np.abs(exp)))
        self.records.append(CheckRecord(
            area=area, test=test, reference=reference,
            parameters=_jsonable(parameters or {}),
            expected=_summary(expected), observed=_summary(observed),
            max_abs_error=err, tolerance=f"atol={atol:g}, rtol={rtol:g}",
            status="PASS" if ok else "FAIL", notes=notes,
        ))
        return ok

    def condition(
        self,
        area: str,
        test: str,
        reference: str,
        passed: bool,
        *,
        expected: Any,
        observed: Any,
        parameters: dict[str, Any] | None = None,
        notes: str = "",
    ) -> bool:
        self.records.append(CheckRecord(
            area=area, test=test, reference=reference,
            parameters=_jsonable(parameters or {}),
            expected=_summary(expected), observed=_summary(observed),
            max_abs_error=0.0 if passed else float("nan"),
            tolerance="exact condition", status="PASS" if passed else "FAIL",
            notes=notes,
        ))
        return bool(passed)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, complex):
        return {"re": value.real, "im": value.imag}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _summary(value: Any) -> Any:
    if isinstance(value, (str, bool)) or value is None:
        return value
    arr = np.asarray(value)
    if arr.ndim == 0:
        item = arr.item()
        if isinstance(item, complex):
            return {"re": item.real, "im": item.imag}
        return item
    if arr.size <= 8:
        return _jsonable([complex(v) if np.iscomplexobj(arr) else float(v) for v in arr.ravel()])
    mag = np.abs(arr)
    return {"shape": list(arr.shape), "min_abs": float(mag.min()), "max_abs": float(mag.max()),
            "mean_abs": float(mag.mean())}


# --------------------------------------------------------------------------
# Scenario helpers
# --------------------------------------------------------------------------
def d_setup(**overrides):
    base = resolve_setup_selection(PROFILE)
    return base.overridden(**overrides) if overrides else base


def calibrated_setup(**overrides):
    kw = dict(
        spdc_source_mode="calibrated_source",
        spdc_gain_per_sqrt_mW=SOFTWARE_TEST_K_PER_SQRT_MW,
        spdc_gain_calibration_id=SOFTWARE_TEST_CALIBRATION_ID,
        spdc_gain_calibration_id_NL2=SOFTWARE_TEST_CALIBRATION_ID,
    )
    kw.update(overrides)
    return d_setup(**kw)


def quiet_state(setup, **kw) -> LatentState:
    base = dict(
        setup=setup, num_frames=1, exposure_s=0.02, em_gain=100.0,
        s_pump=0.0, s_gain=0.0, s_align=0.0, s_thermal=0.0,
        s_det=0.0, s_cool=0.0, s_focus=0.0, s_slm=0.0,
    )
    base.update(kw)
    return LatentState(**base)


def object_ref(T: np.ndarray, phi: np.ndarray) -> dict[str, Any]:
    T = np.asarray(T, dtype=np.float32)
    one = np.ones_like(T)
    return {"M": one, "T": T, "phi": np.asarray(phi, dtype=np.float32),
            "O_i": one, "D": np.zeros_like(T), "target_size": T.shape[0]}


def uniform_ref(size: int = 16, T: float = 1.0, phi: float = 0.0) -> dict[str, Any]:
    return object_ref(np.full((size, size), T), np.full((size, size), phi))


def run(setup, reference: dict[str, Any], state: LatentState):
    if state.get_setup() != setup:
        raise ValueError("LatentState was built from a different setup than the scenario")
    return simulate_forward(reference, state, seed=SEED, detector_noise=False)


def frame_observables(result) -> dict[str, np.ndarray]:
    P = np.asarray(result.frames_plus[0], dtype=np.float64)
    M = np.asarray(result.frames_minus[0], dtype=np.float64)
    return {"P": P, "M": M, "sum": P + M, "contrast": (P - M) / (P + M)}


def spectral_inputs(setup, opd_mm: float) -> dict[str, float]:
    return ref.spectral_coherence_quadrature(
        float(setup.lambda_s_nm), float(setup.signal_spectrum_fwhm_nm),
        float(setup.camera_filter_center_nm), float(setup.camera_filter_fwhm_nm),
        opd_nm=float(opd_mm) * 1.0e6,
    )


def reference_uniform_output(
    setup,
    n1: float,
    n2: float,
    T: float = 1.0,
    phi: float = 0.0,
) -> dict[str, float]:
    """Independent end-to-end expectation for a uniform D object.

    Inputs are configuration values. Idler relay overlap is 1 for the aligned
    default relay (checked separately as a precondition).
    """
    opd_mm = float(setup.delta_L_mm)
    spec = spectral_inputs(setup, opd_mm)
    theta = ref.carrier_phase_rad(opd_mm, float(setup.lambda_s_nm)) + spec["coherence_phase_rad"]
    eta_seed = float(setup.idler_mode_matching) * float(setup.idler_path_transmission)
    t_seed = math.sqrt(eta_seed) * math.sqrt(T) * complex(
        math.cos(phi + float(setup.relative_interferometric_phase)),
        math.sin(phi + float(setup.relative_interferometric_phase)),
    )
    m = ref.reduced_order_detected_moments(
        n1, n2, t_seed,
        eta_s1=float(setup.path1_intensity), eta_s2=float(setup.path2_intensity),
        coherence=spec["coherence_magnitude"], throughput=spec["throughput"],
    )
    plus, minus = ref.combiner_intensities(m["N1"], m["N2"], m["C12"], theta)
    return {
        "N1": m["N1"], "N2": m["N2"], "C12": m["C12"], "theta": theta,
        "plus": plus, "minus": minus, "sum": plus + minus,
        "contrast": (plus - minus) / (plus + minus),
        "visibility": ref.fixed_combiner_visibility(m["N1"], m["N2"], m["C12"]),
        "T_seed": abs(t_seed) ** 2, "coherence": spec["coherence_magnitude"],
        "throughput": spec["throughput"],
    }


def reference_nbar(setup, interaction: int, pump_mW: float) -> float:
    """Reference nbar from k√P, κ=r0/L and the reference QPM mismatch."""
    two = interaction == 2
    L_m = float(setup.crystal_length_NL2_mm if two and setup.crystal_length_NL2_mm else setup.crystal_length_mm) * 1e-3
    T = float(setup.T_crystal_NL2_C if two else setup.T_crystal_NL1_C)
    T0 = float(setup.T_pm0_NL2_C if two else setup.T_pm0_NL1_C)
    period0 = float(setup.poling_period_NL2_um if two and setup.poling_period_NL2_um else setup.poling_period_um)
    order = int(setup.qpm_order_NL2 if two and setup.qpm_order_NL2 else setup.qpm_order)
    dk = ref.delta_k_per_m(
        float(setup.lambda_p_nm), float(setup.lambda_s_nm), float(setup.lambda_i_nm),
        T, ref.poling_period_at_temperature(period0, T, T0), order,
    )
    r0 = ref.gain_from_pump(SOFTWARE_TEST_K_PER_SQRT_MW, pump_mW)
    return ref.closed_form_nbar(r0 / L_m, dk, L_m)


# --------------------------------------------------------------------------
# 7.1 Pump-power distribution
# --------------------------------------------------------------------------
def check_pump(reg: Registry) -> None:
    eq_b = "P1=R*eta_pre*P*(1+dP)_+, P2=(1-R)*eta_NL2*P*(1+dP)_+"
    cases = [
        ("equal split R=0.5", dict(pump_bs_reflectance=0.5), 100.0, 0.0, (0.5, 1.0, 1.0)),
        ("unequal split with losses R=0.3", dict(pump_bs_reflectance=0.3, pump_pre_crystal_transmission=0.9,
                                                 pump_to_NL2_transmission=0.8), 100.0, 0.0, (0.3, 0.9, 0.8)),
        ("boundary R=0 (all pump to NL2)", dict(pump_bs_reflectance=0.0), 100.0, 0.0, (0.0, 1.0, 1.0)),
        ("boundary R=1 (all pump to NL1)", dict(pump_bs_reflectance=1.0), 100.0, 0.0, (1.0, 1.0, 1.0)),
        ("laser fluctuation dP=+0.2", dict(pump_bs_reflectance=0.4), 50.0, 0.2, (0.4, 1.0, 1.0)),
        ("negative power clipped dP=-1.5", dict(pump_bs_reflectance=0.4), 50.0, -1.5, (0.4, 1.0, 1.0)),
    ]
    for name, ov, P, dP, (R, e_pre, e_nl2) in cases:
        s = d_setup(**ov)
        out = pump_powers_from_setup(s, pump_laser_mW=P, relative_power_fluctuation=dP)
        exp = ref.branched_pump_split(P, R, e_pre, e_nl2, dP)
        reg.compare(AREA_PUMP, f"branched: {name}", eq_b, exp,
                    (out["pump_power_NL1_mW"], out["pump_power_NL2_mW"]), atol=1e-12, rtol=1e-12,
                    parameters={"P_laser_mW": P, "R_BS": R, "eta_pre": e_pre, "eta_NL2": e_nl2, "dP": dP})
        if e_pre == 1.0 and e_nl2 == 1.0:
            reg.compare(AREA_PUMP, f"conservation without loss: {name}", "P1+P2=P*(1+dP)_+",
                        P * ref.fluctuation_factor(dP),
                        out["pump_power_NL1_mW"] + out["pump_power_NL2_mW"], atol=1e-12, rtol=1e-12,
                        parameters={"P_laser_mW": P, "R_BS": R, "dP": dP})
        else:
            reg.compare(AREA_PUMP, f"loss accounting: {name}", "(P1+P2)/P = R*eta_pre+(1-R)*eta_NL2",
                        R * e_pre + (1 - R) * e_nl2, out["pump_branch_sum_over_laser"],
                        atol=1e-12, rtol=1e-12, parameters={"R_BS": R, "eta_pre": e_pre, "eta_NL2": e_nl2})
    s = d_setup(two_crystal_pump_topology="sequential", pump_between_crystals_transmission=0.6,
                pump_pre_crystal_transmission=0.9)
    out = pump_powers_from_setup(s, pump_laser_mW=100.0)
    reg.compare(AREA_PUMP, "sequential topology eta_between=0.6", "P1=eta_pre*P, P2=eta_between*P1",
                ref.sequential_pump_split(100.0, 0.9, 0.6),
                (out["pump_power_NL1_mW"], out["pump_power_NL2_mW"]), atol=1e-12, rtol=1e-12,
                parameters={"P_laser_mW": 100.0, "eta_pre": 0.9, "eta_between": 0.6})
    s = d_setup(use_independent_pump_power_NL2=True, pump_power_NL2_override_mW=12.5,
                pump_power_NL2_override_calibration_id=SOFTWARE_TEST_CALIBRATION_ID)
    out = pump_powers_from_setup(s, pump_laser_mW=100.0, relative_power_fluctuation=0.1)
    reg.compare(AREA_PUMP, "measured NL2 override keeps laser fluctuation",
                "P1=R*P*(1+dP), P2=P2_override*(1+dP)",
                (0.5 * 100.0 * 1.1, 12.5 * 1.1),
                (out["pump_power_NL1_mW"], out["pump_power_NL2_mW"]), atol=1e-12, rtol=1e-12,
                parameters={"P2_override_mW": 12.5, "dP": 0.1})

    # Pump split propagated to generation (calibrated source, software-test k).
    s = calibrated_setup(pump_bs_reflectance=0.3)
    op = quiet_state(s, pump_power_mW=40.0).spdc_operating_point()
    p1, p2 = ref.branched_pump_split(40.0, 0.3)
    reg.compare(AREA_PUMP, "split reaches each crystal (R=0.3, 40 mW)", eq_b, (p1, p2),
                (op["pump_power_NL1_mW"], op["pump_power_NL2_mW"]), atol=1e-12, rtol=1e-12,
                parameters={"P_laser_mW": 40.0, "R_BS": 0.3, "k_per_sqrt_mW": SOFTWARE_TEST_K_PER_SQRT_MW})
    reg.compare(AREA_PUMP, "per-crystal gain r_i = k*sqrt(P_i)", "r_i=k*sqrt(P_i)",
                (ref.gain_from_pump(SOFTWARE_TEST_K_PER_SQRT_MW, p1), ref.gain_from_pump(SOFTWARE_TEST_K_PER_SQRT_MW, p2)),
                (op["spdc_gain_r_NL1"], op["spdc_gain_r_NL2"]), atol=1e-12, rtol=1e-12,
                parameters={"P1_mW": p1, "P2_mW": p2})


# --------------------------------------------------------------------------
# 7.2 Two-source SPDC moments
# --------------------------------------------------------------------------
def check_moments(reg: Registry) -> None:
    ref_text = "vacuum Bogoliubov coefficients: N1=|b_s1|^2, N2=|b_s2|^2, C12=<b_s2,b_s1>"
    t_values = (1.0, 0.5, 0.5 * complex(math.cos(0.7), math.sin(0.7)), 0.0)
    pairs = ((1e-6, 1e-6), (0.1, 0.1), (5.0, 5.0), (50.0, 50.0), (0.2, 0.05), (5.0, 0.1))
    for n1, n2 in pairs:
        for t in t_values:
            m = ref.two_crystal_bogoliubov_moments(n1, n2, t)
            src = spdc_double_pass_moments(n1, n2, t)
            obs = (float(src["N1_source"]), float(src["N2_source"]), complex(src["C12_source"]))
            reg.compare(AREA_MOMENTS, f"source moments n1={n1:g} n2={n2:g} t={complex(t):.3g}",
                        ref_text, (m["N1"], m["N2"], m["C12"]), obs, atol=1e-15, rtol=1e-12,
                        parameters={"n1": n1, "n2": n2, "t_seed": complex(t)})
    m = ref.two_crystal_bogoliubov_moments(5.0, 0.1, 0.5j)
    reg.compare(AREA_MOMENTS, "reference unitarity (commutators)", "[d,d^dagger]=1, [d1,d2^dagger]=0",
                (1.0, 1.0, 0.0), (m["commutator_d1"], m["commutator_d2"], m["commutator_d1_d2"]),
                atol=1e-12, notes="self-consistency of the independent reference")
    for e1, e2, g in ((0.7, 0.4, 1.0), (0.7, 0.4, 0.8), (1.0, 1.0, 0.0)):
        m = ref.two_crystal_bogoliubov_moments(0.2, 0.05, 0.6, eta_s1=e1, eta_s2=e2)
        src = spdc_double_pass_moments(0.2, 0.05, 0.6)
        out = apply_signal_path_losses(src["N1_source"], src["N2_source"], src["C12_source"],
                                       eta_s1=e1, eta_s2=e2, gamma_path=g)
        reg.compare(AREA_MOMENTS, f"signal losses eta1={e1} eta2={e2} gamma={g}",
                    "beam-splitter losses on each signal (gamma scales C12 once)",
                    (m["N1"], m["N2"], g * m["C12"]),
                    (float(out["N1"]), float(out["N2"]), complex(out["C12"])), atol=1e-15, rtol=1e-12,
                    parameters={"eta_s1": e1, "eta_s2": e2, "gamma": g})

    # Normalization: phase-matched nbar = sinh^2(r) from coupled-mode integration.
    L = 2.0e-3
    for r0 in (1e-3, 0.1, 1.0, 3.0):
        cm = ref.coupled_mode_bogoliubov(r0 / L, 0.0, L)
        out = spdc_nbar_from_r0_L_dk(r0=r0, crystal_length_m=L, delta_k_qpm_per_m=0.0)
        reg.compare(AREA_MOMENTS, f"normalization nbar(r0={r0:g}) phase matched",
                    "RK4 coupled-mode |V(L)|^2 (analytic sinh^2 r)", cm["nbar"],
                    out["mean_photons_per_mode"], rtol=1e-9, parameters={"r0": r0, "L_m": L})
        reg.compare(AREA_MOMENTS, f"unitarity |U|^2-|V|^2 (r0={r0:g})", "|U|^2-|V|^2=1",
                    1.0, cm["unitarity"], atol=1e-9, notes="reference self-consistency")

    # Pump dependence through the calibrated generation chain (software-test k).
    for P in (0.4, 4.0, 40.0, 400.0):
        s = calibrated_setup()
        op = quiet_state(s, pump_power_mW=P).spdc_operating_point()
        exp = (reference_nbar(s, 1, 0.5 * P), reference_nbar(s, 2, 0.5 * P))
        reg.compare(AREA_MOMENTS, f"pump dependence nbar(P_laser={P:g} mW)",
                    "nbar=kappa^2|sinh(gL)/g|^2, kappa=k*sqrt(P_i)/L, reference dk",
                    exp, (op["mean_photons_per_mode_NL1"], op["mean_photons_per_mode_NL2"]), rtol=1e-9,
                    parameters={"P_laser_mW": P, "k_per_sqrt_mW": SOFTWARE_TEST_K_PER_SQRT_MW})

    # End-to-end moment regimes (fixed_nbar) and unequal sources.
    for n1, n2, label in ((1e-4, 1e-4, "low gain"), (0.1, 0.1, "transition"), (5.0, 5.0, "high gain"),
                          (0.2, 0.05, "unequal NL1>NL2"), (0.05, 0.2, "unequal NL2>NL1")):
        s = d_setup()
        res = run(s, uniform_ref(), quiet_state(s, mean_photons_per_mode_NL1=n1, mean_photons_per_mode_NL2=n2))
        vis = res.meta["vis"]
        exp = reference_uniform_output(s, n1, n2)
        reg.compare(AREA_MOMENTS, f"end-to-end detected moments ({label})",
                    "Bogoliubov moments x filter throughput; C12 x spectral coherence",
                    (exp["N1"], exp["N2"], abs(exp["C12"])),
                    (vis["N1_mean"], vis["N2_mean"], vis["C12_abs_mean"]), rtol=1e-6,
                    parameters={"n1": n1, "n2": n2})
        reg.compare(AREA_MOMENTS, f"end-to-end fixed-combiner visibility ({label})", "V=2|C12|/(N1+N2)",
                    exp["visibility"], vis["visibility_fixed_combiner_mean"], rtol=1e-6,
                    parameters={"n1": n1, "n2": n2})


# --------------------------------------------------------------------------
# 7.3 Single-pass transmissive object
# --------------------------------------------------------------------------
def check_object(reg: Registry) -> None:
    s = d_setup()
    n = 0.1
    base = run(s, uniform_ref(), quiet_state(s, mean_photons_per_mode=n))
    base_obs = frame_observables(base)
    base_exp = reference_uniform_output(s, n, n)
    reg.compare(AREA_OBJECT, "idler relay overlap precondition (aligned default)", "eta_mode=1 for matched waists",
                1.0, base.meta["vis"]["eta_mode_relay"], atol=1e-12)
    reg.compare(AREA_OBJECT, "transparent object contrast", "I+-I- / I++I- from reference moments",
                base_exp["contrast"], base_obs["contrast"], atol=5e-6, parameters={"T": 1.0, "phi": 0.0, "nbar": n})
    for T in (0.64, 0.25, 0.01):
        res = run(s, uniform_ref(T=T), quiet_state(s, mean_photons_per_mode=n))
        obs = frame_observables(res)
        exp = reference_uniform_output(s, n, n, T=T)
        reg.compare(AREA_OBJECT, f"attenuating object T={T}: T_seed", "T_seed=T (single pass)",
                    T, res.meta["vis"]["T_seed_mean"], rtol=1e-6, parameters={"T": T})
        reg.compare(AREA_OBJECT, f"attenuating object T={T}: |C12| ratio", "|C12(T)|/|C12(1)|=sqrt(T)",
                    math.sqrt(T), res.meta["vis"]["C12_abs_mean"] / base.meta["vis"]["C12_abs_mean"],
                    rtol=1e-6, parameters={"T": T})
        reg.compare(AREA_OBJECT, f"attenuating object T={T}: contrast", "reference induced-coherence chain",
                    exp["contrast"], obs["contrast"], atol=5e-6, parameters={"T": T})
        reg.compare(AREA_OBJECT, f"attenuating object T={T}: mean intensity ratio",
                    "(N1+N2(T))/(N1+N2(1)), N2=n2(1+T n1)", exp["sum"] / base_exp["sum"],
                    obs["sum"] / base_obs["sum"], rtol=2e-6, parameters={"T": T})
    # Uniform phase-only objects on the SLM grid.
    levels = list(range(0, 256, 32))
    phases = np.array([ref.slm_grid_phase(k) for k in levels])
    contrasts = []
    for phi in phases:
        res = run(s, uniform_ref(phi=phi), quiet_state(s, mean_photons_per_mode=n))
        obs = frame_observables(res)
        exp = reference_uniform_output(s, n, n, phi=float(phi))
        contrasts.append(float(np.mean(obs["contrast"])))
        reg.compare(AREA_OBJECT, f"uniform phase-only object phi={phi:.4f} rad", "V*cos(phi+theta)",
                    exp["contrast"], obs["contrast"], atol=5e-6, parameters={"phi_rad": float(phi)})
    fit = ref.harmonic_fit(phases, np.array(contrasts))
    exp0 = reference_uniform_output(s, n, n)
    reg.compare(AREA_OBJECT, "phase applied once: second harmonic of contrast(phi)",
                "single pass => contrast has only the cos(phi+theta) harmonic", 0.0,
                fit["second_harmonic"], atol=1e-5)
    reg.compare(AREA_OBJECT, "phase applied once: first-harmonic amplitude", "V=2|C12|/(N1+N2)",
                exp0["visibility"], fit["first_harmonic"], rtol=1e-4,
                notes="amplitude of the phase response equals the fixed-combiner visibility")
    wrapped = math.remainder(fit["first_harmonic_phase"] - exp0["theta"], 2 * math.pi)
    reg.compare(AREA_OBJECT, "phase applied once: fringe phase offset", "offset = theta (carrier + spectral)",
                0.0, wrapped, atol=1e-4)
    # Object not omitted / not doubled.
    res = run(s, uniform_ref(T=0.25, phi=ref.slm_grid_phase(64)), quiet_state(s, mean_photons_per_mode=n))
    reg.condition(AREA_OBJECT, "object applied once (not squared)", "T_seed=0.25 (single pass), not 0.0625",
                  abs(res.meta["vis"]["T_seed_mean"] - 0.25) < 1e-6 and abs(res.meta["vis"]["T_seed_mean"] - 0.0625) > 0.1,
                  expected=0.25, observed=res.meta["vis"]["T_seed_mean"])
    res = run(s, uniform_ref(T=0.0), quiet_state(s, mean_photons_per_mode=n))
    obs = frame_observables(res)
    exp = reference_uniform_output(s, n, n, T=0.0)
    reg.compare(AREA_OBJECT, "opaque object removes coherence (object not omitted)", "T=0 => C12=0, contrast=0",
                0.0, obs["contrast"], atol=1e-7, parameters={"T": 0.0})
    reg.compare(AREA_OBJECT, "opaque object: NL2 unseeded", "N2=n2 (no seeding)", exp["N2"],
                res.meta["vis"]["N2_mean"], rtol=1e-6)
    check_spatial_phase_object(reg)


def block_object(size: int = 32) -> tuple[np.ndarray, np.ndarray, list[tuple[slice, slice, float, float]]]:
    """Four quadrants with distinct transmission and SLM-grid phases."""
    h = size // 2
    spec = [
        (slice(0, h), slice(0, h), 1.0, ref.slm_grid_phase(0)),
        (slice(0, h), slice(h, size), 1.0, ref.slm_grid_phase(64)),
        (slice(h, size), slice(0, h), 0.49, ref.slm_grid_phase(128)),
        (slice(h, size), slice(h, size), 0.81, ref.slm_grid_phase(192)),
    ]
    T = np.zeros((size, size))
    phi = np.zeros((size, size))
    for sy, sx, t, p in spec:
        T[sy, sx] = t
        phi[sy, sx] = p
    return T, phi, spec


def check_spatial_phase_object(reg: Registry) -> None:
    s = d_setup(blur_model="single_blur")
    n = 0.1
    T, phi, spec = block_object(32)
    res = run(s, object_ref(T, phi), quiet_state(s, mean_photons_per_mode=n))
    obs = frame_observables(res)
    margin = 5  # > optical-PSF support so block interiors are unaffected
    for sy, sx, t, p in spec:
        interior = (slice(sy.start + margin, sy.stop - margin), slice(sx.start + margin, sx.stop - margin))
        exp = reference_uniform_output(s, n, n, T=t, phi=p)
        reg.compare(AREA_OBJECT, f"spatially varying object block T={t} phi={p:.4f}",
                    "per-pixel single-pass t(rho_o) mapped with scale 1", exp["contrast"],
                    obs["contrast"][interior], atol=5e-6, parameters={"T": t, "phi_rad": p, "blur_model": "single_blur"})


# --------------------------------------------------------------------------
# 7.4 Near-field relay mapping
# --------------------------------------------------------------------------
def _patterns(size: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    c = (size - 1) / 2.0
    zero = np.zeros((size, size))
    impulse_c = zero.copy()
    impulse_c[size // 2, size // 2] = 1.0
    impulse_o = zero.copy()
    impulse_o[size // 2 - 5, size // 2 + 3] = 1.0
    gauss = np.exp(-((yy - c) ** 2 + (xx - c) ** 2) / (2 * 3.0 ** 2))
    asym = np.exp(-((yy - c + 4) ** 2 / (2 * 2.0 ** 2) + (xx - c - 6) ** 2 / (2 * 1.5 ** 2)))
    asym[size // 2 + 4: size // 2 + 7, size // 2 - 6: size // 2 - 2] = 0.6
    return {
        "centered impulse": (impulse_c, zero),
        "off-axis impulse": (impulse_o, zero),
        "horizontal gradient": (xx / (size - 1), 0.5 * xx / (size - 1)),
        "vertical gradient": (yy / (size - 1), 0.5 * yy / (size - 1)),
        "symmetric Gaussian": (gauss, zero),
        "asymmetric pattern": (np.clip(asym, 0, 1), 1.2 * asym),
    }


def _object_field(T: np.ndarray, phi: np.ndarray, transform: str = "identity") -> np.ndarray:
    f = np.sqrt(np.asarray(T, dtype=np.float64)) * np.exp(1j * np.asarray(phi, dtype=np.float64))
    if transform == "flip_x":
        f = f[:, ::-1]
    elif transform == "flip_y":
        f = f[::-1, :]
    elif transform == "rotate_180":
        f = f[::-1, ::-1]
    return f


def check_mapping(reg: Registry) -> None:
    size = 32
    cases = [
        (1.0, 2.0, 8.0), (1.0, 4.0, 8.0), (2.0, 1.0, 16.0), (1.0, 2.5, 8.0), (1.5, 1.5, 16.0),
    ]
    patterns = _patterns(size)
    for m_u, m_s, p_obj in cases:
        s = d_setup(blur_model="single_blur", near_field_idler_magnification=m_u,
                    near_field_signal_magnification=m_s, object_map_pixel_um=p_obj)
        p_cam = float(s.pixel_um)
        scale = ref.near_field_index_scale(m_u, m_s, p_obj, p_cam)
        mapping = resolved_spatial_mapping(s)
        params = {"M_u": m_u, "M_s": m_s, "p_object_um": p_obj, "p_camera_um": p_cam, "index_scale": scale}
        reg.compare(AREA_MAPPING, f"index scale M_u={m_u} M_s={m_s} p_obj={p_obj}",
                    "(M_s/M_u)*p_obj/p_cam", scale, mapping["object_to_camera_index_scale"], rtol=1e-14,
                    parameters=params)
        reg.condition(AREA_MAPPING, f"orientation (no inversion) M_u={m_u} M_s={m_s}",
                      "position correlation rho_c=+(M_s/M_u) rho_o", mapping["orientation_sign"] == 1
                      and mapping["signed_object_to_camera_magnification"] > 0,
                      expected=1, observed=mapping["orientation_sign"], parameters=params)
        fwd = resolve_forward(s, size)
        for name, (T, phi) in patterns.items():
            out = architecture_object_transfer(T.astype(np.float32), phi.astype(np.float32), s, fwd)
            exp = ref.bilinear_edge_clamped(_object_field(T.astype(np.float32), phi.astype(np.float32)), 1.0 / scale)
            reg.compare(AREA_MAPPING, f"{name} (scale {scale:g})", "bilinear edge-clamped rho_c=s*rho_o",
                        exp, out["correlated_field_before_spdc_psf"], atol=1e-6, parameters=params)
        # Centroid of a smooth off-axis spot moves by the index scale.
        big = 64
        yy, xx = np.mgrid[0:big, 0:big].astype(np.float64)
        c = (big - 1) / 2.0
        dy, dx = -2.0, 3.0
        spot = np.exp(-((yy - c - dy) ** 2 + (xx - c - dx) ** 2) / (2 * 1.5 ** 2))
        out = architecture_object_transfer(spot.astype(np.float32), np.zeros_like(spot, np.float32), s,
                                           resolve_forward(s, big))
        cy, cx = ref.intensity_centroid(np.abs(out["correlated_field_before_spdc_psf"]) ** 2)
        reg.compare(AREA_MAPPING, f"spot centroid displacement (scale {scale:g})", "Delta rho_c = s*Delta rho_o",
                    (scale * dy, scale * dx), (cy - c, cx - c), atol=0.05, parameters=params,
                    notes="smooth spot > 4 sigma from boundaries; linear-interpolation centroid bias < 0.05 px")
    # Spatial identity.
    s = d_setup(blur_model="single_blur")
    fwd = resolve_forward(s, size)
    T, phi = patterns["asymmetric pattern"]
    out = architecture_object_transfer(T.astype(np.float32), phi.astype(np.float32), s, fwd)
    reg.compare(AREA_MAPPING, "unit index scale is the identity map (default D: M_s/M_u=2, p_obj/p_cam=1/2)",
                "s=1 => rho_c=rho_o", _object_field(T.astype(np.float32), phi.astype(np.float32)),
                out["correlated_field_before_spdc_psf"], atol=1e-7)
    # Flips.
    for transform in ("flip_x", "flip_y", "rotate_180"):
        s = d_setup(blur_model="single_blur", object_forward_transform=transform,
                    near_field_signal_magnification=2.5)
        fwd = resolve_forward(s, size)
        scale = ref.near_field_index_scale(1.0, 2.5, float(s.object_map_pixel_um), float(s.pixel_um))
        out = architecture_object_transfer(T.astype(np.float32), phi.astype(np.float32), s, fwd)
        exp = ref.bilinear_edge_clamped(_object_field(T.astype(np.float32), phi.astype(np.float32), transform), 1.0 / scale)
        reg.compare(AREA_MAPPING, f"declared object transform {transform} then relay (scale {scale:g})",
                    "flip applied once before rho_c=s*rho_o", exp, out["correlated_field_before_spdc_psf"], atol=1e-6)
    # Boundary handling for demagnification (outer samples replicate edges).
    s = d_setup(blur_model="single_blur", near_field_idler_magnification=2.0, near_field_signal_magnification=1.0,
                object_map_pixel_um=16.0)
    fwd = resolve_forward(s, size)
    T, phi = patterns["horizontal gradient"]
    out = architecture_object_transfer(T.astype(np.float32), phi.astype(np.float32), s, fwd)
    field_out = out["correlated_field_before_spdc_psf"]
    obj = _object_field(T.astype(np.float32), phi.astype(np.float32))
    reg.compare(AREA_MAPPING, "boundary: magnification 0.5 replicates edge samples",
                "input coordinate clamped to [0, N-1]", (obj[:, 0].mean(), obj[:, -1].mean()),
                (field_out[:, 0].mean(), field_out[:, -1].mean()), atol=1e-6)
    # Relay-derived widths and P10 coefficients.
    for m_s, w in ((2.0, None), (4.0, 100.0)):
        ov = dict(near_field_signal_magnification=m_s)
        if w is not None:
            ov.update(pump_waist_um=w, pump_waist_NL1_um=w)
        s = d_setup(**ov)
        L_um = float(s.crystal_length_mm) * 1e3
        ld, lu = float(s.lambda_s_nm) * 1e-3, float(s.lambda_i_nm) * 1e-3
        wp = float(s.pump_waist_um)
        dv = ref.p07_visibility_half_width_um(L_um, ld, lu, wp)
        reg.compare(AREA_MAPPING, f"P07 visibility width (w_p={wp} um)", "P07 Eq. 4.9", dv,
                    near_field_visibility_1e_um(s), rtol=1e-12, parameters={"w_p_um": wp, "L_um": L_um})
        mapping = resolved_spatial_mapping(s)
        reg.compare(AREA_MAPPING, f"camera correlation width (M_s={m_s}, w_p={wp})", "sigma_c=|M_s|*Delta_V/sqrt(2)",
                    abs(m_s) * dv / math.sqrt(2.0), mapping["correlation_sigma_camera_um"], rtol=1e-12)
        jg = ref.p10_joint_gaussian(L_um, ld, lu, wp)
        co = near_field_joint_gaussian_coefficients(s)
        reg.compare(AREA_MAPPING, f"P10 joint Gaussian coefficients (w_p={wp})", "A=2/(w^2 S^2), B=4pi/(L S)",
                    (jg["A"], jg["B"], jg["alpha"], jg["beta"], jg["gamma"], jg["conditional_scale"]),
                    (co["A_per_um2"], co["B_per_um2"], co["alpha_uu_per_um2"], co["beta_dd_per_um2"],
                     co["gamma_du_per_um2"], co["conditional_center_scale"]), rtol=1e-12)
    reg.compare(AREA_MAPPING, "large-waist limit of Delta_V", "Delta_V -> sqrt(L(ls+li)/4pi)",
                math.sqrt(2000.0 * (0.81 + 1.55) / (4 * math.pi)),
                ref.p07_visibility_half_width_um(2000.0, 0.81, 1.55, 1e6), rtol=1e-6,
                notes="reference self-consistency")
    # Full P10 transfer (double_blur, default D) including off-axis pump.
    for ov, label in ((dict(), "default D"), (dict(near_field_signal_magnification=2.5), "scale 1.25"),
                      (dict(pump_center_x_um=40.0, pump_center_y_um=-24.0, pump_waist_um=120.0,
                            pump_waist_NL1_um=120.0), "off-axis pump, w_p=120 um")):
        s = d_setup(**ov)
        fwd = resolve_forward(s, size)
        T, phi = patterns["asymmetric pattern"]
        out = architecture_object_transfer(T.astype(np.float32), phi.astype(np.float32), s, fwd)
        exp = ref.p10_near_field_transfer(
            _object_field(T.astype(np.float32), phi.astype(np.float32)),
            m_u=float(s.near_field_idler_magnification), m_s=float(s.near_field_signal_magnification),
            p_object_um=float(s.object_map_pixel_um), p_camera_um=float(s.pixel_um),
            length_um=float(s.crystal_length_mm) * 1e3, lambda_d_um=float(s.lambda_s_nm) * 1e-3,
            lambda_u_um=float(s.lambda_i_nm) * 1e-3, waist_um=float(s.pump_waist_um),
            pump_x_um=float(s.pump_center_x_um), pump_y_um=float(s.pump_center_y_um),
        )
        reg.compare(AREA_MAPPING, f"P10 conditional object response t_V ({label})",
                    "t_V = G/intP, G = int P10(rho_d,rho_u) t(rho_o)", exp["t_V"], out["idler_return_field"],
                    atol=1e-9)
        reg.compare(AREA_MAPPING, f"P10 marginal intensity weight ({label})", "intP / max(intP)",
                    exp["marginal_weight"], out["joint_marginal_weight"], atol=1e-9)
        unit = architecture_object_transfer(np.ones((size, size), np.float32), np.zeros((size, size), np.float32), s, fwd)
        reg.compare(AREA_MAPPING, f"unit object stays unit through P10 ({label})", "t=1 => t_V=1",
                    np.ones((size, size)), unit["idler_return_field"], atol=1e-12)
    check_mapping_end_to_end(reg)


def check_mapping_end_to_end(reg: Registry) -> None:
    size = 32
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    c = (size - 1) / 2.0
    dy, dx = 3.0, -4.0
    spot = np.exp(-((yy - c - dy) ** 2 + (xx - c - dx) ** 2) / (2 * 1.5 ** 2))
    for transform, sign_y, sign_x in (("identity", 1, 1), ("flip_x", 1, -1)):
        s = d_setup(blur_model="single_blur", near_field_signal_magnification=2.5, object_forward_transform=transform,
                    delta_L_mm=0.0)
        scale = ref.near_field_index_scale(1.0, 2.5, float(s.object_map_pixel_um), float(s.pixel_um))
        res = run(s, object_ref(spot, np.zeros_like(spot)), quiet_state(s, mean_photons_per_mode=1e-3))
        contrast = frame_observables(res)["contrast"]
        cy, cx = ref.intensity_centroid(np.abs(contrast))
        reg.compare(AREA_MAPPING, f"end-to-end image position and orientation ({transform}, scale {scale:g})",
                    "camera centroid = s * (declared transform of object offset)",
                    (scale * sign_y * dy, scale * sign_x * dx), (cy - c, cx - c), atol=0.15,
                    parameters={"object_offset_px": [dy, dx], "transform": transform, "scale": scale},
                    notes="centroid of |contrast| after optical PSF; tolerance covers sqrt(T) nonlinearity")


# --------------------------------------------------------------------------
# 7.5 QPM / temperature
# --------------------------------------------------------------------------
def check_qpm(reg: Registry) -> None:
    s = d_setup()
    lp, ls, li = float(s.lambda_p_nm), float(s.lambda_s_nm), float(s.lambda_i_nm)
    for lam in (lp, ls, li):
        for T in (20.0, 83.7):
            reg.compare(AREA_QPM, f"KTP n_z(lambda={lam:.2f} nm, T={T} C)", "Kato & Takaoka 2002 + dn/dT",
                        ref.ktp_nz(lam * 1e-3, T), ktp_nz(lam * 1e-3, T), rtol=1e-13)
    reg.compare(AREA_QPM, "energy conservation of the D wavelength triple", "1/li = 1/lp - 1/ls",
                ref.idler_wavelength_nm(lp, ls), li, rtol=1e-9)
    for interaction, T0, period0 in ((1, float(s.T_pm0_NL1_C), float(s.poling_period_um)),
                                     (2, float(s.T_pm0_NL2_C), float(s.poling_period_NL2_um))):
        for dT in (0.0, -1.0, 1.0, 5.0):
            T = T0 + dT
            period = ref.poling_period_at_temperature(period0, T, T0)
            reg.compare(AREA_QPM, f"NL{interaction} poling period Lambda(T={T:.1f} C)",
                        "Lambda0(1+alpha(T-Tpm0))", period,
                        qpm_poling_period_um(s, T_C=T, interaction=interaction), rtol=1e-14)
            reg.compare(AREA_QPM, f"NL{interaction} delta k at T={T:.1f} C", "2pi(np/lp-ns/ls-ni/li-m/Lambda(T))",
                        ref.delta_k_per_m(lp, ls, li, T, period),
                        delta_k_qpm_per_m_from_setup(s, T_C=T, interaction=interaction), rtol=1e-9, atol=1e-6,
                        parameters={"T_C": T, "Lambda0_um": period0, "T_pm0_C": T0})
        reg.compare(AREA_QPM, f"NL{interaction} phase-matched period", "m/(np/lp-ns/ls-ni/li)",
                    ref.phase_matched_period_um(lp, ls, li, T0),
                    phase_match_poling_period_um(s, interaction=interaction), rtol=1e-12)
        Tpm = ref.phase_matching_temperature_C(lp, ls, li, period0, T0)
        dk_at_tpm = delta_k_qpm_per_m_from_setup(s, T_C=Tpm, interaction=interaction)
        reg.compare(AREA_QPM, f"NL{interaction} production dk vanishes at reference T_pm={Tpm:.3f} C",
                    "bisection of reference dk(T)=0", 0.0, dk_at_tpm, atol=1e-6,
                    notes="nominal lab temperatures are not at the Sellmeier-predicted phase match; "
                          "calibration-dependent")
        h = 0.01
        slope_ref = (ref.delta_k_per_m(lp, ls, li, T0 + h, ref.poling_period_at_temperature(period0, T0 + h, T0))
                     - ref.delta_k_per_m(lp, ls, li, T0 - h, ref.poling_period_at_temperature(period0, T0 - h, T0))) / (2 * h)
        slope = (delta_k_qpm_per_m_from_setup(s, T_C=T0 + h, interaction=interaction)
                 - delta_k_qpm_per_m_from_setup(s, T_C=T0 - h, interaction=interaction)) / (2 * h)
        reg.compare(AREA_QPM, f"NL{interaction} temperature tuning slope d(dk)/dT", "finite difference of reference",
                    slope_ref, slope, rtol=1e-6)
    # NL1 and NL2 are independent crystals.
    m = d_setup(crystal_matching_mode="mismatch_study", crystal_length_NL2_mm=5.0, poling_period_NL2_um=9.70,
                T_crystal_NL2_C=86.0)
    T2, T20 = float(m.T_crystal_NL2_C), float(m.T_pm0_NL2_C)
    reg.compare(AREA_QPM, "NL2 uses its own period/temperature (mismatch study)",
                "dk_NL2(T2, Lambda2(T2))",
                ref.delta_k_per_m(lp, ls, li, T2, ref.poling_period_at_temperature(9.70, T2, T20)),
                delta_k_qpm_per_m_from_setup(m, interaction=2), rtol=1e-9,
                parameters={"L2_mm": 5.0, "Lambda2_um": 9.70, "T2_C": T2, "T_pm0_NL2_C": T20})
    reg.compare(AREA_QPM, "NL1 unchanged by NL2 parameters", "dk_NL1 independent of NL2 fields",
                delta_k_qpm_per_m_from_setup(s, interaction=1), delta_k_qpm_per_m_from_setup(m, interaction=1),
                rtol=1e-15)
    # Alternative wavelength triple through the production dk kernel.
    ls2 = 800.0
    li2 = ref.idler_wavelength_nm(lp, ls2)
    reg.compare(AREA_QPM, "alternative wavelengths ls=800 nm, li from energy conservation",
                "reference dk [rad/um]", ref.delta_k_per_m(lp, ls2, li2, 83.7, 9.675) * 1e-6,
                delta_k_qpm_per_um(lambda_p_nm=lp, lambda_s_nm=ls2, lambda_i_nm=li2, T_C=83.7, Lambda_um=9.675),
                rtol=1e-9)
    # Mismatched gain law against coupled-mode integration.
    L = 2.0e-3
    for r0 in (0.05, 0.5, 2.0):
        for dkL in (1.0, 2.8, 8.0):
            dk = dkL / L
            cm = ref.coupled_mode_bogoliubov(r0 / L, dk, L)
            out = spdc_nbar_from_r0_L_dk(r0=r0, crystal_length_m=L, delta_k_qpm_per_m=dk)
            reg.compare(AREA_QPM, f"nbar with mismatch r0={r0:g}, dk*L={dkL:g}", "RK4 coupled-mode |V(L)|^2",
                        cm["nbar"], out["mean_photons_per_mode"], rtol=1e-8,
                        parameters={"r0": r0, "delta_k_L": dkL, "branch": out["branch"]})
    r0 = 1e-3
    for dkL in (0.5, 2.8, 6.0):
        out = spdc_nbar_from_r0_L_dk(r0=r0, crystal_length_m=L, delta_k_qpm_per_m=dkL / L)
        x = 0.5 * dkL
        reg.compare(AREA_QPM, f"low-gain sinc^2 law dk*L={dkL}", "nbar/r0^2 -> sinc^2(dkL/2)",
                    (math.sin(x) / x) ** 2, out["mean_photons_per_mode"] / r0 ** 2, rtol=1e-5)
    # Temperature detuning of NL2 propagates to generation (software-test k).
    for dT in (0.0, 2.0):
        c = calibrated_setup(T_crystal_NL2_C=float(s.T_crystal_NL2_C) + dT)
        op = quiet_state(c, pump_power_mW=40.0).spdc_operating_point()
        reg.compare(AREA_QPM, f"NL2 temperature detuning {dT:+.1f} C reaches nbar_NL2",
                    "nbar(k sqrt(P2), L2, dk_NL2(T2))", (reference_nbar(c, 1, 20.0), reference_nbar(c, 2, 20.0)),
                    (op["mean_photons_per_mode_NL1"], op["mean_photons_per_mode_NL2"]), rtol=1e-9)


# --------------------------------------------------------------------------
# 7.6 Interference output
# --------------------------------------------------------------------------
def check_interference(reg: Registry) -> None:
    for n1, n2, t, theta in ((0.1, 0.1, 1.0, 0.0), (0.2, 0.05, 0.5j, 1.1), (5.0, 5.0, 0.3, 2.5)):
        m = ref.two_crystal_bogoliubov_moments(n1, n2, t)
        exp = ref.combiner_intensities_operator(m, theta)
        plus, minus = _interfere_from_moments(m["N1"], m["N2"], m["C12"], theta=theta)
        reg.compare(AREA_INTERFERENCE, f"combiner from moments n1={n1} n2={n2} t={t} theta={theta}",
                    "operator form out=(d1 +/- e^{-i theta} d2)/sqrt2", exp, (float(plus), float(minus)),
                    rtol=2e-7, atol=1e-12, notes="float32 output of the production combiner")
    n = 0.1
    cases = [
        ("transparent, carrier phase (default dL)", dict(), dict()),
        ("zero phase (dL=0, high coherence)", dict(delta_L_mm=0.0), dict()),
        ("known uniform phase pi/2", dict(delta_L_mm=0.0), dict(phi=ref.slm_grid_phase(64))),
        ("known uniform phase pi (destructive)", dict(delta_L_mm=0.0), dict(phi=ref.slm_grid_phase(128))),
        ("reduced transmission T=0.25", dict(delta_L_mm=0.0), dict(T=0.25)),
        ("unequal signal paths eta1=0.7 eta2=0.4", dict(delta_L_mm=0.0, path1_intensity=0.7, path2_intensity=0.4), dict()),
        ("partial coherence dL=0.1 mm", dict(delta_L_mm=0.1), dict()),
        # 0.3 mm = 1.5 L_coh is the largest OPD accepted by the F4 request-validity gate.
        ("near-zero coherence dL=0.3 mm", dict(delta_L_mm=0.3), dict()),
        ("relative interferometric phase 0.7 rad", dict(delta_L_mm=0.0, relative_interferometric_phase=0.7), dict()),
        ("idler relay loss eta_i=0.5, mode matching 0.8", dict(delta_L_mm=0.0, idler_path_transmission=0.5,
                                                                idler_mode_matching=0.8), dict()),
    ]
    zero_run = None
    for name, ov, obj in cases:
        s = d_setup(**ov)
        res = run(s, uniform_ref(**obj), quiet_state(s, mean_photons_per_mode=n))
        obs = frame_observables(res)
        exp = reference_uniform_output(s, n, n, **obj)
        if name.startswith("zero phase"):
            zero_run = (obs, exp)
        reg.compare(AREA_INTERFERENCE, f"{name}: contrast", "(I+ - I-)/(I+ + I-) from reference chain",
                    exp["contrast"], obs["contrast"], atol=5e-6, parameters={**ov, **obj, "nbar": n})
        reg.compare(AREA_INTERFERENCE, f"{name}: spectral coherence", "|int S T exp(i k OPD)|/int S T",
                    exp["coherence"], res.meta["vis"]["gamma_path"], atol=1e-7)
    obs0, exp0 = zero_run
    reg.compare(AREA_INTERFERENCE, "constructive port at theta=0", "contrast=+V", exp0["visibility"],
                obs0["contrast"], atol=5e-6)
    # Balanced vs unequal sources, mean intensity and modulation depth over phase steps.
    for n1, n2, label in ((0.1, 0.1, "balanced sources"), (0.3, 0.02, "unequal sources")):
        s = d_setup(delta_L_mm=0.0)
        steps = [ref.slm_grid_phase(k) for k in (0, 64, 128, 192)]
        P, M = [], []
        for phi in steps:
            res = run(s, uniform_ref(phi=phi), quiet_state(s, mean_photons_per_mode_NL1=n1, mean_photons_per_mode_NL2=n2))
            o = frame_observables(res)
            P.append(float(o["P"][8, 8]))
            M.append(float(o["M"][8, 8]))
        exp = reference_uniform_output(s, n1, n2)
        P = np.array(P)
        vis = (P.max() - P.min()) / (P.max() + P.min())
        reg.compare(AREA_INTERFERENCE, f"{label}: visibility from four phase steps", "V=2|C12|/(N1+N2)",
                    exp["visibility"], vis, rtol=1e-5, parameters={"n1": n1, "n2": n2})
        reg.compare(AREA_INTERFERENCE, f"{label}: mean intensity is phase independent",
                    "I+ + I- = N1+N2 for every phase", np.full(4, 1.0), (P + np.array(M)) / (P[0] + M[0]), rtol=2e-6)
        mod = (P - np.array(M)) / (P + np.array(M))
        reg.compare(AREA_INTERFERENCE, f"{label}: modulation follows cos(phi)", "V cos(phi)",
                    exp["visibility"] * np.cos(np.array(steps)), mod, atol=5e-6)
    # Low-gain transparent limit: visibility -> spectral coherence.
    s = d_setup(delta_L_mm=0.0)
    res = run(s, uniform_ref(), quiet_state(s, mean_photons_per_mode=1e-6))
    reg.compare(AREA_INTERFERENCE, "high-coherence low-gain transparent limit", "V -> gamma = 1",
                1.0, res.meta["vis"]["visibility_fixed_combiner_mean"], atol=2e-6)
    # Spatial transfer with default (double_blur) D is verified through t_V in 7.4; here single_blur blocks.
    s = d_setup(blur_model="single_blur", delta_L_mm=0.0)
    T, phi, spec = block_object(32)
    res = run(s, object_ref(T, phi), quiet_state(s, mean_photons_per_mode=n))
    obs = frame_observables(res)
    for sy, sx, t, p in spec:
        interior = (slice(sy.start + 5, sy.stop - 5), slice(sx.start + 5, sx.stop - 5))
        exp = reference_uniform_output(s, n, n, T=t, phi=p)
        reg.compare(AREA_INTERFERENCE, f"spatial phase/amplitude transfer block T={t} phi={p:.4f}",
                    "local contrast = local V cos(phi + theta)", exp["contrast"], obs["contrast"][interior],
                    atol=5e-6)


# --------------------------------------------------------------------------
# 8 Physical invariants
# --------------------------------------------------------------------------
def check_invariants(reg: Registry) -> None:
    for R in (0.0, 0.25, 0.5, 0.75, 1.0):
        out = pump_powers_from_setup(d_setup(pump_bs_reflectance=R), pump_laser_mW=10.0)
        reg.compare(AREA_INVARIANT, f"pump conservation R={R}", "P1+P2=P_laser (lossless BS)", 10.0,
                    out["pump_power_NL1_mW"] + out["pump_power_NL2_mW"], rtol=1e-14)
    s = d_setup(delta_L_mm=0.0)
    res = run(s, uniform_ref(), quiet_state(s, mean_photons_per_mode=0.1))
    exp = reference_uniform_output(s, 0.1, 0.1)
    reg.compare(AREA_INVARIANT, "transparent limit T=1, phi=0", "T_seed=eta_mode*eta_i=1; V from moments",
                (1.0, exp["visibility"]), (res.meta["vis"]["T_seed_mean"], res.meta["vis"]["visibility_fixed_combiner_mean"]),
                rtol=1e-6)
    res = run(s, uniform_ref(T=0.0), quiet_state(s, mean_photons_per_mode=0.1))
    reg.compare(AREA_INVARIANT, "zero transmission", "C12=0, contrast=0", (0.0, 0.0),
                (res.meta["vis"]["C12_abs_mean"], float(np.max(np.abs(frame_observables(res)["contrast"])))), atol=1e-7)
    src = spdc_double_pass_moments(0.0, 0.0, 1.0)
    reg.compare(AREA_INVARIANT, "zero gain: no photons, no coherence", "nbar=0 => N1=N2=C12=0",
                (0.0, 0.0, 0.0), (float(src["N1_source"]), float(src["N2_source"]), complex(src["C12_source"])), atol=0.0)
    reg.compare(AREA_INVARIANT, "zero gain: nbar(r0=0)=0", "sinh^2(0)=0", 0.0,
                spdc_nbar_from_r0_L_dk(r0=0.0, crystal_length_m=2e-3, delta_k_qpm_per_m=1406.0)["mean_photons_per_mode"],
                atol=0.0)
    c = calibrated_setup()
    op = quiet_state(c, pump_power_mW=0.0).spdc_operating_point()
    reg.compare(AREA_INVARIANT, "zero gain: zero pump power gives nbar=0 in both crystals", "P=0 => r=0 => nbar=0",
                (0.0, 0.0), (op["mean_photons_per_mode_NL1"], op["mean_photons_per_mode_NL2"]), atol=0.0)
    # Symmetric sources: a pi relative phase exchanges the output ports.
    a = frame_observables(run(s, uniform_ref(), quiet_state(s, mean_photons_per_mode=0.1)))
    shifted = s.overridden(relative_interferometric_phase=math.pi)
    b = frame_observables(run(shifted, uniform_ref(), quiet_state(shifted, mean_photons_per_mode=0.1)))
    reg.compare(AREA_INVARIANT, "symmetric sources: pi phase swaps output ports", "I+(theta+pi)=I-(theta)",
                a["M"], b["P"], rtol=2e-6)
    m_sym = ref.two_crystal_bogoliubov_moments(1e-6, 1e-6, 1.0)
    reg.compare(AREA_INVARIANT, "symmetric low-gain sources: N1=N2 and V=1", "n1=n2->0, T=1",
                (1.0, 1.0), (m_sym["N2"] / m_sym["N1"], ref.fixed_combiner_visibility(m_sym["N1"], m_sym["N2"], m_sym["C12"])),
                atol=2e-6, notes="reference self-consistency")
    # Single pass t (not t^2) for amplitude and phase.
    r1 = run(s, uniform_ref(T=0.36), quiet_state(s, mean_photons_per_mode=0.1))
    r0 = run(s, uniform_ref(), quiet_state(s, mean_photons_per_mode=0.1))
    ratio = r1.meta["vis"]["C12_abs_mean"] / r0.meta["vis"]["C12_abs_mean"]
    reg.condition(AREA_INVARIANT, "object enters as t (single pass), not t^2", "|C12| ratio = 0.6, not 0.36",
                  abs(ratio - 0.6) < 1e-6, expected=0.6, observed=ratio)
    # Spatial identity.
    size = 16
    fwd = resolve_forward(d_setup(blur_model="single_blur"), size)
    rng = np.random.default_rng(SEED)
    T = rng.uniform(0.0, 1.0, (size, size)).astype(np.float32)
    phi = rng.uniform(0.0, 2 * math.pi, (size, size)).astype(np.float32)
    out = architecture_object_transfer(T, phi, d_setup(blur_model="single_blur"), fwd)
    reg.compare(AREA_INVARIANT, "spatial identity for unit index scale", "rho_c=rho_o",
                _object_field(T, phi), out["correlated_field_before_spdc_psf"], atol=1e-6,
                parameters={"seed": SEED})
    # Spectral filter throughput applied once.
    spec = signal_filter_spectral_response(s, opd_mm=0.0)
    reg.compare(AREA_INVARIANT, "filter throughput (Gaussian x Gaussian) applied once",
                "sigma_f/sqrt(sigma_s^2+sigma_f^2)", ref.gaussian_filter_throughput_analytic(
                    float(s.lambda_s_nm), float(s.signal_spectrum_fwhm_nm),
                    float(s.camera_filter_center_nm), float(s.camera_filter_fwhm_nm)),
                spec["throughput"], rtol=1e-6)
    reg.compare(AREA_INVARIANT, "detected N1 carries throughput exactly once", "N1 = throughput * nbar1",
                spec["throughput"] * 0.1, r0.meta["vis"]["N1_mean"], rtol=1e-6)


ALL_CHECKS: tuple[Callable[[Registry], None], ...] = (
    check_pump, check_moments, check_object, check_mapping, check_qpm, check_interference, check_invariants,
)


def run_all_checks() -> list[dict[str, Any]]:
    reg = Registry()
    for fn in ALL_CHECKS:
        fn(reg)
    return [r.to_dict() for r in reg.records]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="write records as JSON")
    args = parser.parse_args(argv)
    records = run_all_checks()
    failed = [r for r in records if r["status"] != "PASS"]
    for r in records:
        print(f"{r['status']:4s} | {r['area']:36s} | {r['test']} | err={r['max_abs_error']:.3g} | {r['tolerance']}")
    print(f"{len(records) - len(failed)}/{len(records)} checks passed")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""C_lab shared visibility: arbitrary mirror phases, RAW TIFF + simulated stacks.

Integrated metrics: ``visibility_raw_unweighted`` (canonical), ``visibility_weighted_sem``.
Pixelwise: ``pixelwise_mean_raw`` (distinct from integrated). Figure 5 exact quadrature:
``figure5_four_step_map`` — for general phases use ``fit_pixelwise``.

Does not modify SPDC forward physics; pairs with ``tools.z1.estimator`` for legacy Fig.5.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import tifffile

from tools.visibility.intensity import BackgroundMode, prepare_intensity_frame


DEFAULT_POSITION_UM = {"0": 0.0, "pi2": 25.0, "pi": 50.0,
                       "3pi2": 75.0, "2pi": 100.0}


def phase_from_mirror_um(positions_um, *, wavelength_um=1.550,
                         double_pass=True):
    """Return relative optical phase (radians); positions must be measured stage positions.

    For double pass, phi=4*pi*z/lambda. Offset in stage origin does not matter
    for fitted visibility (absorbed into sinusoid phase), unlike position errors.
    """
    z = np.asarray(positions_um, dtype=np.float64)
    if not np.isfinite(z).all() or wavelength_um <= 0:
        raise ValueError("Invalid mirror positions or idler wavelength")
    factor = 4.0 if double_pass else 2.0
    return factor * np.pi * (z - z[0]) / float(wavelength_um)


def _design(phases):
    ph = np.asarray(phases, dtype=float).reshape(-1)
    if ph.size < 4:
        raise ValueError("Need >=4 phase positions for residual-based validation")
    if not np.isfinite(ph).all():
        raise ValueError("Nonfinite phase")
    x = np.column_stack((np.ones(ph.size), np.cos(ph), np.sin(ph)))
    if np.linalg.matrix_rank(x) != 3 or np.linalg.cond(x) > 1.0e6:
        raise ValueError("Phase sampling is rank-deficient/ill-conditioned")
    return x


def _lstsq_sinusoid(phases_rad, y, weights=None):
    x = _design(phases_rad)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(y) != len(x) or not np.isfinite(y).all():
        raise ValueError("Invalid phase mean intensities")
    if weights is None:
        coeff = np.linalg.lstsq(x, y, rcond=None)[0]
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape != y.shape or not np.all(w > 0):
            raise ValueError("Invalid LS weights")
        coeff = np.linalg.lstsq(x * w[:, None], y * w, rcond=None)[0]
    return x, coeff


def visibility_from_abc(a: float, b: float, c: float, *, denominator_abs: bool = False) -> float:
    amp = float(np.hypot(b, c))
    if denominator_abs:
        return float(amp / abs(a)) if abs(a) > 0 else float("nan")
    return float(amp / a) if a > 0 else float("nan")


def fit_integrated(phases_rad, phase_means, phase_sems=None,
                   *, max_chi2_reduced=5.0, max_relative_residual=0.25):
    """Fit A + B cos(phi) + C sin(phi) with no bounds, clamps or offsets.

    Canonical integrated visibility uses **unweighted** LS (deterministic). When
    phase SEMs are supplied, a **separate** inverse-variance weighted fit is also
    reported; older C_lab pilots used that weighted V (e.g. ~0.5465 no-object).
    """
    y = np.asarray(phase_means, dtype=np.float64).reshape(-1)
    x, coeff = _lstsq_sinusoid(phases_rad, y, weights=None)
    a, b, c = map(float, coeff)
    fitted = x @ coeff
    residual = y - fitted
    amplitude = float(np.hypot(b, c))
    v_unw = visibility_from_abc(a, b, c, denominator_abs=False)
    norm_resid = float(np.max(np.abs(residual)) / max(amplitude, 1e-12))
    chi2_reduced = float("nan")
    sigma = None
    if phase_sems is not None:
        sigma = np.asarray(phase_sems, dtype=float).reshape(-1)
        if sigma.shape == y.shape and np.isfinite(sigma).all() and np.all(sigma > 0):
            chi2_reduced = float(np.sum((residual / sigma) ** 2) / (len(y) - 3))
    v_w = float("nan")
    coeff_w = None
    if sigma is not None and np.all(sigma > 0):
        _, coeff_w = _lstsq_sinusoid(phases_rad, y, weights=1.0 / (sigma ** 2))
        v_w = visibility_from_abc(float(coeff_w[0]), float(coeff_w[1]), float(coeff_w[2]), denominator_abs=True)
    reasons = []
    if a <= 0 or not np.isfinite(v_unw) or v_unw > 1.0 + 1.0e-8:
        reasons.append("NONPHYSICAL_DC_OR_VISIBILITY")
    if norm_resid > max_relative_residual:
        reasons.append("LARGE_RESIDUAL")
    if not np.isfinite(chi2_reduced):
        reasons.append("UNCERTAINTY_NOT_ESTIMABLE")
    elif chi2_reduced > max_chi2_reduced:
        reasons.append("EXCESS_CHI_SQUARED")
    out = {
        "dc_adu": a,
        "cos_coeff_adu": b,
        "sin_coeff_adu": c,
        "amplitude_adu": amplitude,
        "visibility_raw": v_unw,
        "visibility_raw_unweighted": v_unw,
        "visibility_weighted_sem": v_w,
        "weighted_dc_adu": float(coeff_w[0]) if coeff_w is not None else float("nan"),
        "weighted_cos_coeff_adu": float(coeff_w[1]) if coeff_w is not None else float("nan"),
        "weighted_sin_coeff_adu": float(coeff_w[2]) if coeff_w is not None else float("nan"),
        "phase_offset_rad": float(np.arctan2(-c, b)),
        "fitted_adu": fitted.tolist(),
        "residual_adu": residual.tolist(),
        "max_relative_residual_to_amplitude": norm_resid,
        "chi2_reduced": chi2_reduced,
        "statistically_valid": not reasons,
        "flags": reasons,
        "ls_method_primary": "unweighted_lstsq",
        "note": "RAW ADU visibility: includes uncalibrated camera bias; not necessarily optical fringe visibility.",
    }
    return out


def fit_pixelwise(phases_rad, mean_images, roi):
    """Return pixel-wise RAW V map from the SAME sinusoidal model as integrated.

    Mask is fixed and not altered; invalid pixel fits remain NaN. Do not confuse
    mean(Vpixel) with V(sum(Ipixel)).
    """
    stack = np.asarray(mean_images, dtype=np.float64)
    mask = np.asarray(roi, dtype=bool)
    if stack.ndim != 3 or stack.shape[1:] != mask.shape or mask.sum() < 1:
        raise ValueError("Expected [phase, row, col] mean images and 2D ROI")
    x = _design(phases_rad)
    if stack.shape[0] != len(x):
        raise ValueError("Phase count mismatch")
    val = stack[:, mask]
    if not np.isfinite(val).all():
        raise ValueError("Invalid pixel values")
    betas = np.linalg.lstsq(x, val, rcond=None)[0]
    dc = betas[0]
    amp = np.hypot(betas[1], betas[2])
    valid = (dc > 0) & (amp <= dc * (1 + 1e-8))
    v = np.full(dc.shape, np.nan)
    v[valid] = amp[valid] / dc[valid]
    out = np.full(mask.shape, np.nan)
    out[mask] = v
    sums = {
        "dc_adu_sum": float(np.sum(dc)),
        "cos_coeff_adu_sum": float(np.sum(betas[1])),
        "sin_coeff_adu_sum": float(np.sum(betas[2])),
    }
    sums["visibility_from_summed_coefficients"] = visibility_from_abc(
        sums["dc_adu_sum"], sums["cos_coeff_adu_sum"], sums["sin_coeff_adu_sum"]
    )
    return out, {
        "roi_pixels": int(mask.sum()),
        "valid_pixels": int(valid.sum()),
        "pixelwise_mean_raw": float(np.nanmean(v)) if valid.any() else float("nan"),
        "pixelwise_median_raw": float(np.nanmedian(v)) if valid.any() else float("nan"),
        **sums,
        "betas": betas,
    }


def verify_integrated_matches_pixelwise_sums(phases_rad, mean_images, roi, fit_integrated_result):
    """Integrated unweighted LS on ROI sums must equal summed pixelwise A,B,C."""
    _, vstats = fit_pixelwise(phases_rad, mean_images, roi)
    v_int = float(fit_integrated_result["visibility_raw_unweighted"])
    v_sum = float(vstats["visibility_from_summed_coefficients"])
    return {
        "integrated_unweighted_V": v_int,
        "visibility_from_summed_pixel_coefficients": v_sum,
        "abs_difference": abs(v_int - v_sum),
        "consistent_within_1e-9": abs(v_int - v_sum) < 1.0e-9,
        "dc_adu_integrated": float(fit_integrated_result["dc_adu"]),
        "dc_adu_sum_pixels": float(vstats["dc_adu_sum"]),
    }



def figure5_four_step_map(phase_frames, roi=None):
    """Exact historical Figure-5 estimator for 0,pi/2,pi,3pi/2 ONLY.

    Do not apply blindly to approximate stage increments; use fit_pixelwise
    with displacement-derived phases for arbitrary/approximate positions.
    """
    stack = np.asarray(phase_frames, dtype=np.float64)
    if stack.ndim != 3 or stack.shape[0] != 4:
        raise ValueError("Four exact quadrature frames required")
    mask = (np.ones(stack.shape[1:], dtype=bool) if roi is None
            else np.asarray(roi, dtype=bool))
    if mask.shape != stack.shape[1:]:
        raise ValueError("ROI shape does not match frames")
    a = stack[0] - stack[2]
    b = stack[1] - stack[3]
    denom = stack.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = 2.0 * np.hypot(a, b) / denom
    good = mask & np.isfinite(v) & (denom > 0) & (v >= 0) & (v <= 1.0+1e-8)
    out = np.full(mask.shape, np.nan, dtype=float)
    out[good] = v[good]
    return out, {"mean_v":float(np.nanmean(out)) if good.any() else float("nan"),
                 "valid_pixels":int(good.sum()), "roi_pixels":int(mask.sum())}


def moments_to_phase_stack(n1, n2, c12, phases_rad):
    """The ORIGINAL optics._interfere_from_moments I_plus convention:

        I_plus = (N1 + N2)/2 + Re(C12 * exp(i*phi)).

    Gives V_theory = 2*abs(C12)/(N1+N2). Supports spatially complex C12.
    Camera PSF/crop/noise/background must be modeled separately and consistently.
    """
    n1 = np.asarray(n1, dtype=float)
    n2 = np.asarray(n2, dtype=float)
    c12 = np.asarray(c12, dtype=np.complex128)
    if n1.shape != n2.shape or n1.shape != c12.shape or n1.ndim != 2:
        raise ValueError("N1, N2 and C12 must have same 2D shape")
    if np.any(n1 < 0) or np.any(n2 < 0):
        raise ValueError("Negative theoretical intensity moments")
    base = (n1+n2)/2
    stack = np.stack([base + np.real(c12*np.exp(1j*phi))
                      for phi in np.asarray(phases_rad)], axis=0)
    if (stack < -1e-8).any():
        raise ValueError("Unphysical moments (negative camera intensity)")
    return stack


def theory_integrated_moments(n1, n2, c12, roi):
    m = np.asarray(roi, bool)
    s = float(np.sum(np.asarray(n1)[m]+np.asarray(n2)[m]))
    return float(2*np.abs(np.sum(np.asarray(c12)[m]))/s) if s > 0 else float("nan")


def unique_tiff_files(directory: Path) -> list[Path]:
    """Return each TIFF path exactly once, including on case-insensitive systems.

    Earlier code concatenated four case-specific glob results.  On Windows the
    glob patterns are case-insensitive, so every physical file could be counted
    up to four times.  Enumerating once and filtering by a lower-case suffix
    preserves the files and their order while fixing frame counts and SEM.
    """
    directory = Path(directory)
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}],
        key=lambda p: (p.name.casefold(), str(p.resolve()).casefold()),
    )


def load_raw_tiff_phase_series(
    condition_dir: Path,
    position_by_folder: Mapping[str, float],
    roi,
    *,
    background_mode: BackgroundMode = BackgroundMode.RAW_NO_DARK,
    bias_adu: np.ndarray | None = None,
):
    """Read only the explicit phase folders; never open 'dark' or 'ref' subdirs.

    Each phase contains 1+ original TIFF frames. All frames share original sensor
    coordinate system (no resize, flip, crop or per-frame normalization).
    """
    condition_dir = Path(condition_dir)
    roi = np.asarray(roi, dtype=bool)
    phase_mean_imgs, mean_integrals, sems, phase_rows = [], [], [], []
    raw_per_phase = []
    for label, z_um in position_by_folder.items():
        p = condition_dir / label
        if not p.is_dir():
            raise FileNotFoundError(f"Required phase folder missing: {p}")
        files = unique_tiff_files(p)
        if not files:
            raise ValueError(f"No TIFF files: {p}")
        sum_img = np.zeros(roi.shape, dtype=np.float64)
        integrals = []
        num_sat = 0
        for f in files:
            frame = tifffile.imread(f)
            if frame.ndim != 2 or frame.shape != roi.shape or frame.dtype != np.uint16:
                raise ValueError(f"Unexpected TIFF shape/type: {f}, {frame.shape}, {frame.dtype}")
            num_sat += int(np.count_nonzero((frame == 65535) & roi))
            arr = prepare_intensity_frame(frame, mode=background_mode, bias_adu=bias_adu)
            sum_img += arr
            integrals.append(float(arr[roi].sum()))
        integ = np.asarray(integrals, dtype=np.float64)
        mean_integrals.append(float(integ.mean()))
        sems.append(float(integ.std(ddof=1)/np.sqrt(integ.size)) if integ.size > 1 else float("nan"))
        phase_mean_imgs.append(sum_img / len(files))
        raw_per_phase.append(integ)
        phase_rows.append({"phase_folder":label, "mirror_um":float(z_um),
                           "n_frames": len(files),"roi_sum_mean_adu":float(integ.mean()),
                           "roi_sum_sem_adu": sems[-1],
                           "saturated_roi_pixel_events":num_sat})
    return np.asarray(phase_mean_imgs), np.asarray(mean_integrals), np.asarray(sems), phase_rows, raw_per_phase


def bootstrap_integrated(phases, raw_per_phase, *, trials=400, seed=2026):
    """Conditional frame-resampling uncertainty; excludes phase/drift/bias error.

    Never use CI to override a failed sinusoidal goodness-of-fit test.
    """
    rng = np.random.default_rng(seed)
    vals=[]
    for _ in range(trials):
        means = [float(np.mean(rng.choice(v, size=len(v), replace=True))) for v in raw_per_phase]
        v = fit_integrated(phases, means)["visibility_raw"]
        if np.isfinite(v): vals.append(v)
    return ([float(a) for a in np.percentile(vals, [2.5,97.5])]
            if len(vals) >= max(20, trials//2) else [float("nan")]*2)


def _jsonable(v):
    if isinstance(v, dict): return {k:_jsonable(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)): return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):return v.item()
    return v


def _plot_results(out, image, roi, phase, fits, observations, sems):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    ax.imshow(image, cmap="gray", origin="upper")
    ax.contour(roi,levels=[.5],colors=["cyan"],linewidths=1)
    ax.set_title(f"Frozen ROI: {int(roi.sum()):,} pixels on RAW no-object reference")
    ax.set_xlabel("Detector column (pixel)");ax.set_ylabel("Detector row (pixel)")
    fig.tight_layout();fig.savefig(out/"ROI_OVERLAY.png",dpi=170);plt.close(fig)
    fig, axs =plt.subplots(2,1,figsize=(8,8),sharex=True)
    smooth = np.linspace(0,2*np.pi,300)
    for condition in ("no_object","rabbit"):
        f=fits[condition]; y=observations[condition]; se=sems[condition]
        wrapped=np.mod(phase,2*np.pi)
        yfit=f["dc_adu"]+f["cos_coeff_adu"]*np.cos(smooth)+f["sin_coeff_adu"]*np.sin(smooth)
        err=np.where(np.isfinite(se),se,0.0)
        axs[0].errorbar(wrapped,y,yerr=err,fmt="o",capsize=3,label=f"{condition}: {f['visibility_raw']:.3f} [RAW; {('PASS' if f['statistically_valid'] else 'FLAGGED')}]" )
        axs[0].plot(smooth,yfit,alpha=.8)
        axs[1].scatter(wrapped,np.asarray(f['residual_adu']),label=condition)
    axs[0].set_ylabel("Integrated RAW counts (ADU)")
    axs[1].set_ylabel("Residual (ADU)");axs[1].set_xlabel("Mirror-derived wrapped phase (rad)")
    axs[0].legend();axs[1].legend();axs[1].axhline(0,linewidth=.7,color="gray")
    fig.tight_layout();fig.savefig(out/"RAW_PHASE_FITS.png",dpi=170);plt.close(fig)


def run_simulated_stack(
    phase_stack: np.ndarray,
    roi: np.ndarray,
    phases_rad: np.ndarray,
    *,
    condition_label: str = "simulation",
) -> dict[str, Any]:
    """Analyze a simulated [phase, row, col] stack with the same estimators as experiment."""
    stack = np.asarray(phase_stack, dtype=np.float64)
    mask = np.asarray(roi, dtype=bool)
    ph = np.asarray(phases_rad, dtype=np.float64)
    ints = stack[:, mask].sum(axis=1)
    sem = np.full(ints.shape, np.nan)
    fit = fit_integrated(ph, ints, sem)
    _, vstats = fit_pixelwise(ph, stack, mask)
    vstats.pop("betas", None)
    fit.update(vstats)
    fit["coefficient_consistency"] = verify_integrated_matches_pixelwise_sums(ph, stack, mask, fit)
    fit["data_condition"] = condition_label
    fit["background_mode"] = "SIMULATED_STACK"
    return fit


def run_experiment(root, roi_npy, position_by_folder, *, output,
                   without_name="witout object", with_name="with object", bootstrap=400):
    """A read-only analysis of both object conditions on ONE frozen ROI."""
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    roi=np.load(roi_npy, allow_pickle=False).astype(bool)
    if roi.ndim != 2 or roi.sum() < 500:
        raise ValueError("Invalid or degenerate ROI (<500 px)")
    ph=phase_from_mirror_um(list(position_by_folder.values()))
    fits={};obs={};sems={};stack_ref=None;rows=[]
    for name,dirname in (("no_object",without_name),("rabbit",with_name)):
        stack,ints,sem,phase_rows,repeats=load_raw_tiff_phase_series(root/dirname, position_by_folder, roi)
        if stack_ref is None: stack_ref=stack.mean(axis=0)
        fit=fit_integrated(ph,ints,sem)
        vmap, vstats=fit_pixelwise(ph,stack,roi)
        vstats.pop("betas", None)
        fit.update(vstats)
        fit["coefficient_consistency"] = verify_integrated_matches_pixelwise_sums(ph, stack, roi, fit)
        fit["bootstrap_ci95_conditional"] = bootstrap_integrated(ph,repeats,trials=bootstrap)
        fit["bootstrap_caveat"] = "Frame bootstrap is conditional on phase positions and stationarity; NOT calibrated physical uncertainty."
        fit["data_condition"] = name
        fit["background_mode"] = "RAW_NO_DARK"
        fit["mirror_positions_um"] = list(map(float,position_by_folder.values()))
        fit["phases_wrapped_rad"] = np.mod(ph,2*np.pi).tolist()
        fit["roi_pixels"] = int(roi.sum())
        fit["saturated_roi_pixel_events"] = sum(r["saturated_roi_pixel_events"] for r in phase_rows)
        if fit["saturated_roi_pixel_events"]:
            fit["statistically_valid"] = False
            fit["flags"].append("SATURATION_INSIDE_FIXED_ROI")
        fits[name]=fit; obs[name]=ints;sems[name]=sem
        np.save(out/f"RAW_VMAP_{name.upper()}.npy",vmap)
        for row in phase_rows:
            row.update({"condition":name,"phase_rad_wrapped":float(np.mod(ph[list(position_by_folder).index(row['phase_folder'])],2*np.pi))})
            rows.append(row)
    with (out/"PHASE_RAW_SUMS.csv").open("w",encoding="utf-8",newline="") as fh:
        writer=csv.DictWriter(fh,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    _plot_results(out,stack_ref,roi,ph,fits,obs,sems)
    payload={"instrument":"EMCCD RAW TIFF", "background_mode":"NO_DARK_SUBTRACTION",
             "source_root":str(root),"roi_source":str(roi_npy),"roi_pixels":int(roi.sum()),
             "note":"RAW visibility is NOT corrected optical visibility. Fitted data may be flagged despite 0<=V<=1.",
             "analysis":fits}
    (out/"VISIBILITY_RESULT.json").write_text(json.dumps(_jsonable(payload),indent=2,allow_nan=True),encoding="utf-8")
    return payload


def reconciliation_report(root, roi_npy, position_by_folder, *, without_name="witout object"):
    """Document first cause of pilot vs shared integrated-V differences (read-only)."""
    root = Path(root)
    roi = np.load(roi_npy).astype(bool)
    ph = phase_from_mirror_um(list(position_by_folder.values()))
    stack, ints, sem, phase_rows, _ = load_raw_tiff_phase_series(
        root / without_name, position_by_folder, roi
    )
    fit_u = fit_integrated(ph, ints, None)
    fit_w = fit_integrated(ph, ints, sem)
    return {
        "first_difference": "inverse_variance_weighted_lstsq_vs_unweighted_lstsq",
        "cause": (
            "Earlier five-phase pilots (tools.c_lab_a6_11 fit_integrated with phase SEMs) "
            "used weighted LS and V=amplitude/|A|. This module's canonical visibility_raw "
            "uses unweighted LS and V=amplitude/A (A>0). Same TIFF integrals and phases; "
            "different estimator — not different ROI or TIFF selection."
        ),
        "shared_inputs": {
            "roi_pixels": int(roi.sum()),
            "phase_vector_rad": ph.tolist(),
            "integrated_means_ADU": ints.tolist(),
            "integrated_sem_ADU": sem.tolist(),
            "phase_rows": phase_rows,
        },
        "unweighted": {
            "A": fit_u["dc_adu"],
            "B": fit_u["cos_coeff_adu"],
            "C": fit_u["sin_coeff_adu"],
            "V": fit_u["visibility_raw_unweighted"],
        },
        "weighted_sem_legacy_pilot_style": {
            "A": fit_w["weighted_dc_adu"],
            "B": fit_w["weighted_cos_coeff_adu"],
            "C": fit_w["weighted_sin_coeff_adu"],
            "V": fit_w["visibility_weighted_sem"],
        },
        "coefficient_consistency": verify_integrated_matches_pixelwise_sums(ph, stack, roi, fit_u),
        "reference_pilot_values_not_targets": {"no_object_weighted_V": 0.5465, "rabbit_weighted_V": 0.1265},
    }


def main(argv=None):
    p=argparse.ArgumentParser(description="RAW TIFF visibility from arbitrary mirror phase folders")
    p.add_argument("--experimental-root",type=Path,required=True,
                   help="parent of 'witout object' and 'with object'")
    p.add_argument("--roi",type=Path,required=True,help="frozen A6.12E 20,309px .npy mask")
    p.add_argument("--positions",type=Path,required=True,help="JSON map {folder_name: mirror_um}")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--without-name",default="witout object")
    p.add_argument("--with-name",default="with object")
    p.add_argument("--bootstrap",type=int,default=400)
    args=p.parse_args(argv)
    pos=json.loads(args.positions.read_text(encoding="utf-8"))
    if len(pos)<4 or not isinstance(pos,dict) or not all(isinstance(v,(int,float)) for v in pos.values()):
        raise ValueError("Positions JSON must have >=4 folder-to-displacement entries")
    res=run_experiment(args.experimental_root,args.roi,pos,output=args.output,
                       without_name=args.without_name,with_name=args.with_name,
                       bootstrap=args.bootstrap)
    for label, fit in res["analysis"].items():
        print(f"{label}: RAW V={fit['visibility_raw']:.4f}; "+
              f"quality={'PASS' if fit['statistically_valid'] else 'FLAGGED'}; "+
              f"chi2_red={fit['chi2_reduced']:.2f}; flags={','.join(fit['flags'])}")
    print("PRODUCTION_FILES_MODIFIED = NO")


if __name__ == "__main__":
    main()

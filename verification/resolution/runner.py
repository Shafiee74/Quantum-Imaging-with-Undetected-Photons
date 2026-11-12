"""A/B/C/D resolution benchmark runner (Task 08)."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.targets import build_target
from latents import LatentState
from setup_params import resolve_setup_selection, resolved_spatial_mapping
from simulate import simulate_forward
from verification.resolution.definitions import (
    DEFAULT_GRID_N,
    DEFAULT_NBAR,
    DEFAULT_SEED,
    DEFINITIONS_MANIFEST,
    PROFILES,
    QUANTITIES,
    QUANTITY_INTENSITY,
    QUANTITY_VISIBILITY,
    TARGETS,
)
from verification.resolution.profiles import (
    ProfileFit,
    average_cols,
    average_rows,
    build_esf_profile,
    build_lsf_from_esf,
    build_psf_profile,
    extract_central_row,
)
from verification.resolution.quantities import quantity_maps


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def quiet_state(setup: Any, *, nbar: float = DEFAULT_NBAR) -> LatentState:
    return LatentState(
        setup=setup,
        mean_photons_per_mode=float(nbar),
        num_frames=1,
        exposure_s=0.02,
        em_gain=100.0,
        s_pump=0.0,
        s_gain=0.0,
        s_align=0.0,
        s_thermal=0.0,
        s_det=0.0,
        s_cool=0.0,
        s_focus=0.0,
        s_slm=0.0,
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_setup(setup: Any) -> str:
    payload = json.dumps(setup.to_dict(), sort_keys=True, default=str).encode("utf-8")
    return _sha256_bytes(payload)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def save_profile_csv(path: Path, fit: ProfileFit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "axis_um",
                "original_profile",
                "processed_profile",
                "in_fitting_region",
            ]
        )
        for i in range(fit.axis_um.size):
            w.writerow(
                [
                    float(fit.axis_um[i]),
                    float(fit.original_profile[i]),
                    float(fit.processed_profile[i]),
                    int(bool(fit.fitting_region_mask[i])),
                ]
            )


def plot_profile(path: Path, fit: ProfileFit, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True)
    axes[0].plot(fit.axis_um, fit.original_profile, label="original")
    axes[0].set_ylabel("original [arb]")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(fit.axis_um, fit.processed_profile, label="processed")
    region = fit.fitting_region_mask
    if np.any(region):
        axes[1].fill_between(
            fit.axis_um,
            0.0,
            np.max(fit.processed_profile) if np.max(fit.processed_profile) > 0 else 1.0,
            where=region,
            color="C1",
            alpha=0.15,
            label="fitting region",
        )
    axes[1].set_xlabel("axis [µm]")
    axes[1].set_ylabel("processed [1]")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    metric = fit.metric_value
    metric_s = "n/a" if metric is None else f"{metric:.4g} {fit.metric_unit}"
    fig.suptitle(f"{title}\n{fit.metric_name} = {metric_s}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@dataclass
class CaseResult:
    profile: str
    target_id: str
    quantity: str
    metrics: dict[str, Any]
    fits: dict[str, ProfileFit]


def analyze_maps(
    *,
    profile: str,
    target_id: str,
    quantity: str,
    image: np.ndarray,
    x_um: np.ndarray,
    y_um: np.ndarray,
    expected_point_xy_um: tuple[float, float],
    expected_edge_um: float,
    edge_orientation: str | None,
) -> CaseResult:
    fits: dict[str, ProfileFit] = {}
    metrics: dict[str, Any] = {
        "profile": profile,
        "target_id": target_id,
        "quantity": quantity,
    }

    if target_id in ("point", "narrow_slit"):
        # Horizontal PSF/LSF cut through mid-row (predeclared PSF_AXIS).
        row = int(np.argmin(np.abs(y_um - expected_point_xy_um[1])))
        raw = extract_central_row(image, row=row)
        psf = build_psf_profile(x_um, raw, quantity=quantity)
        fits["PSF"] = psf
        metrics["psf_fwhm_um"] = psf.metric_value
        metrics["centroid_um"] = psf.extras.get("centroid_um")
        metrics["expected_centroid_um"] = float(expected_point_xy_um[0])
        cen = psf.extras.get("centroid_um")
        metrics["centroid_error_um"] = (
            None if cen is None else float(cen) - float(expected_point_xy_um[0])
        )
        # For slit also report ESF-like? No — slit → treat central cut as line response
        # proxy via |gradient| of the cut after normalizing like a soft edge pair.
        if target_id == "narrow_slit":
            # Use absolute derivative of the raw cut as an LSF-like line response.
            deriv = np.abs(np.gradient(raw, x_um))
            lsf_proxy = build_psf_profile(x_um, deriv, quantity=quantity)
            lsf_proxy.profile_kind = "LSF"
            lsf_proxy.metric_name = "lsf_fwhm"
            fits["LSF"] = lsf_proxy
            metrics["lsf_fwhm_um"] = lsf_proxy.metric_value

    if target_id.startswith("knife_edge"):
        if edge_orientation == "vertical":
            raw = average_rows(image)
            axis = x_um
        else:
            raw = average_cols(image)
            axis = y_um
        esf = build_esf_profile(
            axis, raw, quantity=quantity, expected_edge_um=expected_edge_um
        )
        lsf = build_lsf_from_esf(esf)
        fits["ESF"] = esf
        fits["LSF"] = lsf
        metrics["esf_10_90_width_um"] = esf.metric_value
        metrics["lsf_fwhm_um"] = lsf.metric_value
        metrics["spatial_mapping_error_um"] = esf.extras.get("spatial_mapping_error_um")
        metrics["esf_x50_um"] = esf.extras.get("x50_um")
        metrics["expected_edge_um"] = expected_edge_um

    return CaseResult(
        profile=profile,
        target_id=target_id,
        quantity=quantity,
        metrics=metrics,
        fits=fits,
    )


def run_profile_suite(
    profile: str,
    *,
    n: int = DEFAULT_GRID_N,
    seed: int = DEFAULT_SEED,
    nbar: float = DEFAULT_NBAR,
    evidence_root: Path | None = None,
) -> dict[str, Any]:
    setup = resolve_setup_selection(profile)
    sm = resolved_spatial_mapping(setup)
    pixel_um = float(sm["object_plane_pixel_um"])
    signed_m = float(sm["signed_object_to_camera_magnification"])
    state = quiet_state(setup, nbar=nbar)

    data_dir = None if evidence_root is None else evidence_root / "data" / profile
    fig_dir = None if evidence_root is None else evidence_root / "figures" / profile
    if data_dir is not None:
        data_dir.mkdir(parents=True, exist_ok=True)
    if fig_dir is not None:
        fig_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    for target_id in TARGETS:
        tgt = build_target(target_id, n, pixel_um=pixel_um)
        ref = tgt.to_simulation_reference()
        result = simulate_forward(ref, state, seed=int(seed), detector_noise=False)
        maps = quantity_maps(result)

        if data_dir is not None:
            np.save(data_dir / f"{target_id}_intensity.npy", maps[QUANTITY_INTENSITY])
            np.save(data_dir / f"{target_id}_visibility.npy", maps[QUANTITY_VISIBILITY])
            np.save(
                data_dir / f"{target_id}_amplitude_sqrtI_diagnostic.npy",
                maps["amplitude_sqrt_intensity_diagnostic"],
            )
            np.save(data_dir / f"{target_id}_object_T.npy", tgt.T)
            np.save(data_dir / f"{target_id}_x_um.npy", tgt.x_um)
            np.save(data_dir / f"{target_id}_y_um.npy", tgt.y_um)

        expected_point = (
            float(tgt.metadata.get("point_x_um", 0.0)),
            float(tgt.metadata.get("point_y_um", 0.0)),
        )
        if target_id == "point":
            expected_point = (
                float(tgt.metadata["point_x_um"]),
                float(tgt.metadata["point_y_um"]),
            )
        else:
            expected_point = (0.0, 0.0)

        edge_orientation = None
        expected_edge = 0.0
        if target_id == "knife_edge_vertical":
            edge_orientation = "vertical"
            expected_edge = float(tgt.metadata.get("edge_x_um", 0.0))
        elif target_id == "knife_edge_horizontal":
            edge_orientation = "horizontal"
            expected_edge = float(tgt.metadata.get("edge_y_um", 0.0))

        for quantity in QUANTITIES:
            case = analyze_maps(
                profile=profile,
                target_id=target_id,
                quantity=quantity,
                image=maps[quantity],
                x_um=tgt.x_um,
                y_um=tgt.y_um,
                expected_point_xy_um=expected_point,
                expected_edge_um=expected_edge,
                edge_orientation=edge_orientation,
            )
            case.metrics["object_pixel_um"] = pixel_um
            case.metrics["signed_magnification"] = signed_m
            case.metrics["imaging_mode"] = sm.get("imaging_mode")
            case.metrics["seed"] = int(seed)
            case.metrics["nbar"] = float(nbar)
            case.metrics["n"] = int(n)
            case.metrics["setup_hash"] = hash_setup(setup)

            if data_dir is not None:
                for kind, fit in case.fits.items():
                    stem = f"{target_id}__{quantity}__{kind}"
                    save_profile_csv(data_dir / f"{stem}_profile.csv", fit)
                    np.save(data_dir / f"{stem}_axis_um.npy", fit.axis_um)
                    np.save(data_dir / f"{stem}_original.npy", fit.original_profile)
                    np.save(data_dir / f"{stem}_processed.npy", fit.processed_profile)
                    np.save(
                        data_dir / f"{stem}_fitting_region.npy",
                        fit.fitting_region_mask.astype(np.uint8),
                    )
                    save_json(
                        data_dir / f"{stem}_metric.json",
                        {
                            "metric_name": fit.metric_name,
                            "metric_value": fit.metric_value,
                            "metric_unit": fit.metric_unit,
                            "extras": fit.extras,
                            "quantity": quantity,
                            "profile_kind": kind,
                        },
                    )
                    if fig_dir is not None:
                        plot_profile(
                            fig_dir / f"{stem}_profile.png",
                            fit,
                            f"{profile} {target_id} {quantity} {kind}",
                        )

            cases.append({"metrics": case.metrics, "fit_kinds": list(case.fits.keys())})

        # Preview montage intensity vs visibility
        if fig_dir is not None:
            fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
            im0 = axes[0].imshow(maps[QUANTITY_INTENSITY], origin="lower", cmap="magma")
            axes[0].set_title(f"{profile} {target_id} intensity")
            plt.colorbar(im0, ax=axes[0], fraction=0.046)
            im1 = axes[1].imshow(maps[QUANTITY_VISIBILITY], origin="lower", cmap="viridis")
            axes[1].set_title(f"{profile} {target_id} visibility")
            plt.colorbar(im1, ax=axes[1], fraction=0.046)
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])
            fig.tight_layout()
            fig.savefig(fig_dir / f"{target_id}_intensity_vs_visibility.png", dpi=150)
            plt.close(fig)

    return {
        "profile": profile,
        "pixel_um": pixel_um,
        "signed_magnification": signed_m,
        "spatial_mapping": {k: sm[k] for k in sm if not isinstance(sm[k], (np.ndarray,))},
        "setup_hash": hash_setup(setup),
        "cases": cases,
    }


def run_all_profiles(
    *,
    n: int = DEFAULT_GRID_N,
    seed: int = DEFAULT_SEED,
    nbar: float = DEFAULT_NBAR,
    evidence_root: Path | None = None,
    profiles: tuple[str, ...] = PROFILES,
) -> dict[str, Any]:
    out = {
        "utc": utc_now(),
        "definitions": DEFINITIONS_MANIFEST,
        "n": int(n),
        "seed": int(seed),
        "nbar": float(nbar),
        "profiles": {},
    }
    for p in profiles:
        out["profiles"][p] = run_profile_suite(
            p, n=n, seed=seed, nbar=nbar, evidence_root=evidence_root
        )
    return out

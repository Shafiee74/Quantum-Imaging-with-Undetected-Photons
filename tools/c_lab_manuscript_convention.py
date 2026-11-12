"""Active C_lab manuscript analysis convention (Figures 5–6).

Joint-SPDC theory: physical I_minus, delta_L_mm=0 (OPD0), theory ROI 234498 px.
Experiment: mirror-drift TIFFs, fixed circular ROI r=130 px at (227, 209), no phase offsets.
"""
from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from tools.publication_graphics_style import (
    CONFIGURATION_COLORS,
    CONFIGURATION_LINESTYLES,
    CONFIGURATION_MARKERS,
    apply_publication_graphics_style,
)
from tools.visibility.c_lab_shared import unique_tiff_files

ROOT = Path(__file__).resolve().parents[1]

# --- Active manuscript defaults (Fig 5a / 5b / shared mirror-drift experiment) ---
def _resolve_manuscript_exp_root() -> Path:
    for candidate in (
        ROOT / "exp data" / "exp data" / "mirror drift",
        ROOT / "exp data" / "mirror drift",
    ):
        if candidate.is_dir():
            return candidate
    return ROOT / "exp data" / "exp data" / "mirror drift"


MANUSCRIPT_EXP_ROOT = _resolve_manuscript_exp_root()
MANUSCRIPT_THEORY_DATA_CSV = ROOT / "article figures" / "Figure5a_Final_Data.csv"
LAMBDA_I_UM = 1.55
DELTA_L_MM = 0.0
THEORY_OUTPUT_PORT = "I_minus"
THEORY_ROI_PIXELS = 234_498
ROI_CENTER_X = 227
ROI_CENTER_Y = 209
ROI_RADIUS_PX = 130
ROI_PIXEL_COUNT = 53_077
TIFF_SHAPE = (419, 455)
SKIP_FOLDERS = frozenset({"dark", "ref"})
FOLDER_DZ_UM: dict[str, float] = {
    "0": 0.0,
    "pi2": 25.0,
    "pi": 50.0,
    "3pi2": 75.0,
    "2pi": 100.0,
}
COND_SUBDIR = {"no_object": "witout object", "rabbit": "with object"}
TWOPI = 2.0 * math.pi


@dataclass(frozen=True)
class CLabManuscriptSettings:
    exp_root: Path = MANUSCRIPT_EXP_ROOT
    theory_csv: Path = MANUSCRIPT_THEORY_DATA_CSV
    lambda_i_um: float = LAMBDA_I_UM
    delta_l_mm: float = DELTA_L_MM
    theory_port: str = THEORY_OUTPUT_PORT
    theory_roi_pixels: int = THEORY_ROI_PIXELS
    roi_center: tuple[int, int] = (ROI_CENTER_X, ROI_CENTER_Y)
    roi_radius_px: int = ROI_RADIUS_PX


DEFAULT_MANUSCRIPT_SETTINGS = CLabManuscriptSettings()


def wrapped_phase_rad(dz_um: float, *, lambda_um: float = LAMBDA_I_UM) -> float:
    return float(4.0 * math.pi * dz_um / lambda_um) % TWOPI


def dz_eq_um_from_phase(phi_rad: np.ndarray | float) -> np.ndarray | float:
    p = np.asarray(phi_rad, dtype=np.float64)
    out = p * LAMBDA_I_UM / (4.0 * math.pi)
    return float(out) if np.ndim(phi_rad) == 0 else out


def experimental_roi_mask(settings: CLabManuscriptSettings = DEFAULT_MANUSCRIPT_SETTINGS) -> np.ndarray:
    h, w = TIFF_SHAPE
    yy, xx = np.ogrid[:h, :w]
    cx, cy = settings.roi_center
    r = settings.roi_radius_px
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    assert int(mask.sum()) == ROI_PIXEL_COUNT
    return mask


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_phase_folders(settings: CLabManuscriptSettings = DEFAULT_MANUSCRIPT_SETTINGS) -> list[str]:
    no_root = settings.exp_root / COND_SUBDIR["no_object"]
    rb_root = settings.exp_root / COND_SUBDIR["rabbit"]
    out: list[str] = []
    for name in ("0", "pi2", "pi", "3pi2", "2pi"):
        if name in SKIP_FOLDERS:
            continue
        if (no_root / name).is_dir() and (rb_root / name).is_dir():
            out.append(name)
    return out


def position_map(folders: list[str] | None = None) -> dict[str, float]:
    folders = folders or discover_phase_folders()
    return {f: FOLDER_DZ_UM[f] for f in folders}


def load_opd0_iminus_theory(
    settings: CLabManuscriptSettings = DEFAULT_MANUSCRIPT_SETTINGS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense OPD0 I_minus theory samples (same as validated Fig. 5a export)."""
    path = settings.theory_csv
    phi, clear_y, rb_y = [], [], []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("row") != "theory_sample":
                continue
            p = float(row["phi_z_rad"])
            v = float(row["theory_I_minus_au"])
            if row["series"] == "I_minus_no_object":
                phi.append(p)
                clear_y.append(v)
            elif row["series"] == "I_minus_rabbit70":
                rb_y.append(v)
    order = np.argsort(phi)
    return (
        np.asarray(phi, dtype=np.float64)[order],
        np.asarray(clear_y, dtype=np.float64)[order],
        np.asarray(rb_y, dtype=np.float64)[order],
    )


def theory_csv_dict(settings: CLabManuscriptSettings = DEFAULT_MANUSCRIPT_SETTINGS) -> dict[str, np.ndarray]:
    """Legacy-compatible dict for Figure 5 assembly (panel a / b loaders)."""
    phi, m_no, m_rb = load_opd0_iminus_theory(settings)
    dz = dz_eq_um_from_phase(phi)
    return {
        "phase_rad": phi,
        "dz_um_one_period": dz,
        "roi_mean_no_object_au": m_no,
        "roi_mean_rabbit_au": m_rb,
    }


def _folder_stats_deduped(
    folder: Path, roi: np.ndarray
) -> tuple[list[float], list[dict[str, Any]]]:
    files = unique_tiff_files(folder)
    seen: set[str] = set()
    means: list[float] = []
    audit: list[dict[str, Any]] = []
    for f in files:
        digest = sha256_file(f)
        dup = digest in seen
        if not dup:
            seen.add(digest)
        frame = tifffile.imread(f)
        if frame.dtype != np.uint16 or frame.shape != TIFF_SHAPE:
            raise ValueError(f"Unexpected TIFF {f}: {frame.shape} {frame.dtype}")
        m = float(frame[roi].mean())
        audit.append(
            {
                "source_path": str(f.relative_to(ROOT)).replace("\\", "/"),
                "sha256": digest,
                "roi_mean_adu_per_pixel": m,
                "duplicate_in_folder": int(dup),
                "included_in_statistics": 0 if dup else 1,
            }
        )
        if not dup:
            means.append(m)
    return means, audit


def compute_experimental_statistics(
    settings: CLabManuscriptSettings = DEFAULT_MANUSCRIPT_SETTINGS,
) -> tuple[dict[str, Any], dict[str, float], list[dict[str, Any]]]:
    """Recompute mirror-drift ROI means from on-disk TIFFs (no cached CSV)."""
    roi = experimental_roi_mask(settings)
    folders = discover_phase_folders(settings)
    pos = position_map(folders)
    folder_order = list(pos.keys())
    out: dict[str, Any] = {}
    all_audit: list[dict[str, Any]] = []

    for cond_key, subdir in COND_SUBDIR.items():
        means_list, sems, sds = [], [], []
        for folder in folder_order:
            fdir = settings.exp_root / subdir / folder
            vals, audit = _folder_stats_deduped(fdir, roi)
            for a in audit:
                all_audit.append(
                    {
                        **a,
                        "condition": cond_key,
                        "phase_folder": folder,
                        "delta_z_um": pos[folder],
                        "phi_z_rad": wrapped_phase_rad(pos[folder]),
                    }
                )
            arr = np.asarray(vals, dtype=np.float64)
            n = int(arr.size)
            mean = float(arr.mean()) if n else float("nan")
            sd = float(arr.std(ddof=1)) if n > 1 else 0.0
            sem = sd / math.sqrt(n) if n > 1 else 0.0
            means_list.append(mean)
            sems.append(sem)
            sds.append(sd)
        out[cond_key] = {
            "roi_mean_adu": np.asarray(means_list, dtype=np.float64),
            "roi_mean_sem_adu": np.asarray(sems, dtype=np.float64),
            "roi_mean_sd_adu": np.asarray(sds, dtype=np.float64),
            "folder_order": folder_order,
        }
    return out, pos, all_audit


def load_experiment_from_figure5a_csv(
    csv_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Load verified experimental means/SEM from Figure5a_Final_Data.csv (no TIFF reread)."""
    by_cond: dict[str, dict[str, list[float]]] = {
        "no_object": {"folders": [], "mean": [], "sem": [], "sd": []},
        "rabbit": {"folders": [], "mean": [], "sem": [], "sd": []},
    }
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("row") != "experiment_point":
                continue
            cond = row["condition"]
            if cond not in by_cond:
                continue
            by_cond[cond]["folders"].append(row["folder"])
            by_cond[cond]["mean"].append(float(row["mean"]))
            by_cond[cond]["sem"].append(float(row["sem"]))
            by_cond[cond]["sd"].append(float(row["sd"]))
    folders = list(by_cond["no_object"]["folders"])
    if folders != list(by_cond["rabbit"]["folders"]):
        raise ValueError("no_object and rabbit folder order differ in CSV")
    out: dict[str, Any] = {}
    for cond in ("no_object", "rabbit"):
        out[cond] = {
            "roi_mean_adu": np.asarray(by_cond[cond]["mean"], dtype=np.float64),
            "roi_mean_sem_adu": np.asarray(by_cond[cond]["sem"], dtype=np.float64),
            "roi_mean_sd_adu": np.asarray(by_cond[cond]["sd"], dtype=np.float64),
            "folder_order": folders,
        }
    return out, folders


def setup_phase_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0.0, TWOPI)
    ax.margins(x=0)
    ticks = [0, math.pi / 2, math.pi, 1.5 * math.pi, TWOPI]
    ax.set_xticks(ticks)
    ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_xlabel(r"Interferometric phase $\varphi_z$ (rad)")


def draw_figure5a_manuscript(
    ax_left: plt.Axes,
    *,
    phi_th: np.ndarray,
    y_clear: np.ndarray,
    y_rb: np.ndarray,
    exp: dict[str, Any],
    folders: list[str],
    ylim_exp: tuple[float, float] | None = None,
) -> plt.Axes:
    """Panel (a) curves + points; figure face must not mention ROI radius."""
    apply_publication_graphics_style()
    phi_th = np.asarray(phi_th, dtype=np.float64)
    y_clear = np.asarray(y_clear, dtype=np.float64)
    y_rb = np.asarray(y_rb, dtype=np.float64)
    phi_dense = np.linspace(0.0, TWOPI, 721)
    y_clear_d = np.interp(phi_dense, phi_th, y_clear)
    y_rb_d = np.interp(phi_dense, phi_th, y_rb)
    ax_left.plot(
        phi_dense,
        y_clear_d,
        color=CONFIGURATION_COLORS["A"],
        ls=CONFIGURATION_LINESTYLES["A"],
        lw=1.8,
        clip_on=False,
        label="Simulation, without object",
    )
    ax_left.plot(
        phi_dense,
        y_rb_d,
        color=CONFIGURATION_COLORS["B"],
        ls=CONFIGURATION_LINESTYLES["B"],
        lw=1.8,
        clip_on=False,
        label="Simulation, with object",
    )
    ax_left.set_ylabel("Theoretical ROI mean signal intensity (a.u.)")
    setup_phase_axis(ax_left)
    ax_left.set_xmargin(0)
    ax_left.set_ylim(-0.01, max(float(y_clear.max()), float(y_rb.max())) * 1.08)

    order = exp["no_object"]["folder_order"]
    idxs = [order.index(f) for f in folders]
    phi_exp = np.array([wrapped_phase_rad(FOLDER_DZ_UM[f]) for f in folders])
    ax_right = ax_left.twinx()
    ax_right.errorbar(
        phi_exp,
        exp["no_object"]["roi_mean_adu"][idxs],
        yerr=exp["no_object"]["roi_mean_sem_adu"][idxs],
        fmt=CONFIGURATION_MARKERS["C"],
        ms=7,
        color=CONFIGURATION_COLORS["C"],
        markerfacecolor="white",
        markeredgewidth=1.2,
        capsize=3,
        clip_on=False,
        label="Experimental, without object",
    )
    ax_right.errorbar(
        phi_exp,
        exp["rabbit"]["roi_mean_adu"][idxs],
        yerr=exp["rabbit"]["roi_mean_sem_adu"][idxs],
        fmt=CONFIGURATION_MARKERS["D"],
        ms=7,
        color=CONFIGURATION_COLORS["D"],
        markeredgecolor="0.25",
        markeredgewidth=0.8,
        capsize=3,
        clip_on=False,
        label="Experimental, with object",
    )
    ax_right.set_xlim(ax_left.get_xlim())
    ax_right.set_ylabel("Experimental ROI mean signal intensity (ADU/pixel)")
    y_all = np.concatenate(
        [exp["no_object"]["roi_mean_adu"], exp["rabbit"]["roi_mean_adu"]]
    )
    if ylim_exp is None:
        ylim_exp = (float(y_all.min()) * 0.92, float(y_all.max()) * 1.05)
    ax_right.set_ylim(ylim_exp[0], ylim_exp[1])

    sec = ax_left.secondary_xaxis(
        "top",
        functions=(dz_eq_um_from_phase, lambda z: np.asarray(z, dtype=np.float64) * 4.0 * math.pi / LAMBDA_I_UM),
    )
    sec.set_xlabel(r"Equivalent mirror displacement $\Delta z$ ($\mu$m, one $2\pi$ period)")
    ax_left.set_xlim(0.0, TWOPI)
    ax_right.set_xlim(0.0, TWOPI)

    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_right.get_legend_handles_labels()
    ax_left.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.5, framealpha=0.95)
    return ax_right


def theory_high_low_phases(
    settings: CLabManuscriptSettings = DEFAULT_MANUSCRIPT_SETTINGS,
) -> tuple[float, float, float, float]:
    """High/low-signal phases from no-object OPD0 I_minus curve (Fig. 5a convention)."""
    phi, y_clear, _y_rb = load_opd0_iminus_theory(settings)
    i_hi = int(np.argmax(y_clear))
    i_lo = int(np.argmin(y_clear))
    return float(phi[i_hi]), float(phi[i_lo]), float(y_clear[i_hi]), float(y_clear[i_lo])


def select_experimental_extrema_folders(exp: dict[str, Any]) -> dict[str, str]:
    """Per-condition brightest/dimmest folder from recomputed ROI means."""
    out: dict[str, str] = {}
    for cond in ("no_object", "rabbit"):
        order = list(exp[cond]["folder_order"])
        means = np.asarray(exp[cond]["roi_mean_adu"], dtype=np.float64)
        out[f"high_{cond}"] = order[int(np.argmax(means))]
        out[f"low_{cond}"] = order[int(np.argmin(means))]
    return out


def mean_experimental_folder_image(
    condition: str,
    folder: str,
    settings: CLabManuscriptSettings = DEFAULT_MANUSCRIPT_SETTINGS,
) -> tuple[np.ndarray, int, float]:
    """Deduped temporal mean image and ROI130 mean (ADU/pixel)."""
    roi = experimental_roi_mask(settings)
    sub = COND_SUBDIR[condition]
    fdir = settings.exp_root / sub / folder
    means, _audit = _folder_stats_deduped(fdir, roi)
    if not means:
        raise ValueError(f"No unique frames in {fdir}")
    stack = []
    seen: set[str] = set()
    for f in unique_tiff_files(fdir):
        digest = sha256_file(f)
        if digest in seen:
            continue
        seen.add(digest)
        stack.append(tifffile.imread(f).astype(np.float64))
    img = np.mean(np.stack(stack, 0), axis=0)
    return img, len(stack), float(np.mean(means))

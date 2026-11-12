"""1-D profile metrics: FWHM, ESF 10–90%, centroid (predeclared methods)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from verification.resolution.definitions import (
    ESF_HIGH_LEVEL,
    ESF_LOW_LEVEL,
    ESF_PLATEAU_FRACTION,
    FWHM_METHOD,
)


@dataclass
class ProfileFit:
    """Bundle for one fitted/extracted 1-D resolution profile."""

    quantity: str
    profile_kind: str  # PSF | ESF | LSF
    axis_um: np.ndarray
    original_profile: np.ndarray
    processed_profile: np.ndarray
    fitting_region_mask: np.ndarray
    metric_name: str
    metric_value: float | None
    metric_unit: str
    extras: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # arrays handled by caller when saving
        return d


def _interp_crossing(
    x: np.ndarray, y: np.ndarray, level: float, *, rising: bool | None = None
) -> float | None:
    """Linear-interpolated abscissa where ``y`` crosses ``level``."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dy = y[1:] - y[:-1]
    crossings: list[float] = []
    for i in range(dy.size):
        y0, y1 = y[i], y[i + 1]
        if (y0 - level) == 0.0:
            crossings.append(float(x[i]))
            continue
        if (y0 - level) * (y1 - level) > 0.0:
            continue
        if y1 == y0:
            continue
        is_rising = y1 > y0
        if rising is True and not is_rising:
            continue
        if rising is False and is_rising:
            continue
        t = (level - y0) / (y1 - y0)
        crossings.append(float(x[i] + t * (x[i + 1] - x[i])))
    if not crossings:
        return None
    # Prefer the crossing nearest the peak / mid for stability.
    return crossings[len(crossings) // 2] if len(crossings) > 1 else crossings[0]


def fwhm_linear(
    x_um: np.ndarray,
    y: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Half-maximum FWHM with linear interpolation (predeclared method)."""
    x = np.asarray(x_um, dtype=np.float64)
    y0 = np.asarray(y, dtype=np.float64)
    if mask is None:
        mask = np.ones(y0.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return {
            "fwhm_um": None,
            "half_level": None,
            "x_left_um": None,
            "x_right_um": None,
            "method": FWHM_METHOD,
        }
    yy = y0.copy()
    yy[~mask] = 0.0
    peak_i = int(np.argmax(yy))
    peak = float(yy[peak_i])
    if peak <= 0.0 or not np.isfinite(peak):
        return {
            "fwhm_um": None,
            "half_level": None,
            "x_left_um": None,
            "x_right_um": None,
            "method": FWHM_METHOD,
        }
    half = 0.5 * peak
    # Search left / right from peak within mask.
    left = None
    for i in range(peak_i, 0, -1):
        if not mask[i] or not mask[i - 1]:
            break
        if yy[i] >= half and yy[i - 1] < half:
            t = (half - yy[i - 1]) / (yy[i] - yy[i - 1])
            left = float(x[i - 1] + t * (x[i] - x[i - 1]))
            break
        if yy[i - 1] == half:
            left = float(x[i - 1])
            break
    right = None
    for i in range(peak_i, yy.size - 1):
        if not mask[i] or not mask[i + 1]:
            break
        if yy[i] >= half and yy[i + 1] < half:
            t = (half - yy[i]) / (yy[i + 1] - yy[i])
            right = float(x[i] + t * (x[i + 1] - x[i]))
            break
        if yy[i + 1] == half:
            right = float(x[i + 1])
            break
    fwhm = None if left is None or right is None else float(abs(right - left))
    return {
        "fwhm_um": fwhm,
        "half_level": half,
        "x_left_um": left,
        "x_right_um": right,
        "peak_index": peak_i,
        "peak_value": peak,
        "method": FWHM_METHOD,
    }


def weighted_centroid_1d(x_um: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x_um, dtype=np.float64)
    w = np.clip(np.asarray(y, dtype=np.float64), 0.0, None)
    s = float(np.sum(w))
    if s <= 0.0:
        return {"centroid_um": None, "sum_weight": s}
    return {"centroid_um": float(np.sum(x * w) / s), "sum_weight": s}


def extract_central_row(image: np.ndarray, *, row: int | None = None) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    r = img.shape[0] // 2 if row is None else int(row)
    return img[r, :].copy()


def extract_central_col(image: np.ndarray, *, col: int | None = None) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    c = img.shape[1] // 2 if col is None else int(col)
    return img[:, c].copy()


def average_rows(image: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(image, dtype=np.float64), axis=0)


def average_cols(image: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(image, dtype=np.float64), axis=1)


def build_psf_profile(
    axis_um: np.ndarray,
    raw: np.ndarray,
    *,
    quantity: str,
) -> ProfileFit:
    """Background-subtracted, peak-normalized PSF cut."""
    x = np.asarray(axis_um, dtype=np.float64)
    original = np.asarray(raw, dtype=np.float64)
    # Background: median of outer 15% on each side.
    n = original.size
    k = max(1, int(round(ESF_PLATEAU_FRACTION * n)))
    bg = float(np.median(np.concatenate([original[:k], original[-k:]])))
    subtracted = original - bg
    peak = float(np.max(subtracted))
    processed = subtracted / peak if peak > 0.0 else subtracted.copy()
    mask = np.ones(n, dtype=bool)
    fw = fwhm_linear(x, processed, mask=mask)
    cen = weighted_centroid_1d(x, np.clip(subtracted, 0.0, None))
    return ProfileFit(
        quantity=quantity,
        profile_kind="PSF",
        axis_um=x,
        original_profile=original,
        processed_profile=processed,
        fitting_region_mask=mask,
        metric_name="psf_fwhm",
        metric_value=fw["fwhm_um"],
        metric_unit="um",
        extras={
            "background": bg,
            "peak_raw_minus_bg": peak,
            "fwhm": fw,
            "centroid_um": cen["centroid_um"],
            "normalization": "peak_of_background_subtracted",
        },
    )


def build_esf_profile(
    axis_um: np.ndarray,
    raw: np.ndarray,
    *,
    quantity: str,
    expected_edge_um: float = 0.0,
) -> ProfileFit:
    """Plateau-normalized ESF with predeclared 10–90% width."""
    x = np.asarray(axis_um, dtype=np.float64)
    original = np.asarray(raw, dtype=np.float64)
    n = original.size
    k = max(1, int(round(ESF_PLATEAU_FRACTION * n)))
    # Orient so the high side is on the right for a consistent rising ESF.
    left_mean = float(np.mean(original[:k]))
    right_mean = float(np.mean(original[-k:]))
    oriented = original.copy()
    flipped = False
    if left_mean > right_mean:
        oriented = original[::-1].copy()
        x = x[::-1].copy()
        flipped = True
        left_mean, right_mean = right_mean, left_mean
    span = right_mean - left_mean
    if span <= 0.0:
        processed = np.zeros_like(oriented)
    else:
        processed = (oriented - left_mean) / span
    # Fitting region: exclude plateau averages already used; full curve used
    # for crossing search but plateaus marked False in mask for transparency.
    mask = np.ones(n, dtype=bool)
    mask[:k] = False
    mask[-k:] = False
    x10 = _interp_crossing(x, processed, ESF_LOW_LEVEL, rising=True)
    x90 = _interp_crossing(x, processed, ESF_HIGH_LEVEL, rising=True)
    x50 = _interp_crossing(x, processed, 0.5, rising=True)
    width = None if x10 is None or x90 is None else float(abs(x90 - x10))
    mapping_err = None if x50 is None else float(x50 - float(expected_edge_um))
    return ProfileFit(
        quantity=quantity,
        profile_kind="ESF",
        axis_um=x,
        original_profile=oriented,
        processed_profile=processed,
        fitting_region_mask=mask,
        metric_name="esf_10_90_width",
        metric_value=width,
        metric_unit="um",
        extras={
            "I_low": left_mean,
            "I_high": right_mean,
            "x10_um": x10,
            "x90_um": x90,
            "x50_um": x50,
            "levels": [ESF_LOW_LEVEL, ESF_HIGH_LEVEL],
            "flipped_for_rising": flipped,
            "plateau_fraction": ESF_PLATEAU_FRACTION,
            "expected_edge_um": float(expected_edge_um),
            "spatial_mapping_error_um": mapping_err,
        },
    )


def build_lsf_from_esf(esf: ProfileFit) -> ProfileFit:
    """LSF = d(ESF_processed)/d(x); FWHM on the primary rising-edge lobe.

    Predeclared (not chosen after seeing results):
    ESF is oriented rising, so the physical line-spread peak is
    ``max(dESF/dx)``. FWHM is evaluated on the non-negative lobe
    ``max(LSF, 0)`` peak-normalized — not on ``|LSF|``, which can merge
    secondary opposite-signed lobes from non-monotonic intensity ESFs.
    """
    x = np.asarray(esf.axis_um, dtype=np.float64)
    esf_y = np.asarray(esf.processed_profile, dtype=np.float64)
    if x.size < 2:
        lsf = np.zeros_like(esf_y)
    else:
        lsf = np.gradient(esf_y, x)
    original = lsf.copy()
    positive_lobe = np.clip(lsf, 0.0, None)
    peak = float(np.max(positive_lobe))
    processed = positive_lobe / peak if peak > 0.0 else positive_lobe.copy()
    mask = np.ones(processed.shape, dtype=bool)
    fw = fwhm_linear(x, processed, mask=mask)
    return ProfileFit(
        quantity=esf.quantity,
        profile_kind="LSF",
        axis_um=x,
        original_profile=original,
        processed_profile=processed,
        fitting_region_mask=mask,
        metric_name="lsf_fwhm",
        metric_value=fw["fwhm_um"],
        metric_unit="um",
        extras={
            "from": "gradient_of_processed_ESF",
            "lobe": "nonnegative_rising_edge",
            "fwhm": fw,
            "parent_esf_metric": esf.metric_value,
            "parent_esf_fitting_region_mask": np.asarray(
                esf.fitting_region_mask, dtype=bool
            ),
        },
    )

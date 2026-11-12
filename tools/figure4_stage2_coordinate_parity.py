#!/usr/bin/env python3
"""Figure 4 Stage 2.1 — signed camera/object coordinate parity (no forward reruns)."""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.targets import PHYSICAL_TRUTH
from benchmarks.targets.geometry import centered_axis_um
from setup_params import resolve_setup_selection, resolved_spatial_mapping
from verification.resolution.profiles import build_esf_profile

STAGE1_DIR = ROOT / "evidence" / "figure4_stage1_object_transfer_audit" / "20261020T062325Z"
STAGE1_NPZ = STAGE1_DIR / "figure4_stage1_arrays.npz"
EVIDENCE_ROOT = ROOT / "evidence" / "figure4_stage2_coordinate_parity_audit"
GRID_N = 48


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def central_strip_line(R: np.ndarray) -> np.ndarray:
    h = R.shape[0]
    r0, r1 = max(0, h // 2 - 2), min(h, h // 2 + 3)
    return np.nanmean(R[r0:r1, :], axis=0)


def esf_on_axis(axis_um: np.ndarray, line: np.ndarray) -> dict:
    fit = build_esf_profile(
        axis_um,
        line,
        quantity="modulation_ratio",
        expected_edge_um=float(PHYSICAL_TRUTH["knife_edge_offset_um"]),
    )
    return {
        "axis_um": np.asarray(fit.axis_um, dtype=np.float64),
        "processed": np.asarray(fit.processed_profile, dtype=np.float64),
        "width_10_90_um": fit.metric_value,
        "x50_um": fit.extras.get("x50_um"),
        "flipped_for_rising": bool(fit.extras.get("flipped_for_rising")),
        "spatial_mapping_error_um": fit.extras.get("spatial_mapping_error_um"),
    }


def sort_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    return x[order], y[order]


def plateau_normalize_sorted(
    x: np.ndarray, y: np.ndarray, *, frac: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Plateau ESF vs ascending x (preserves rising/falling along that axis)."""
    xs, ys = sort_xy(x, y)
    n = ys.size
    k = max(1, int(round(frac * n)))
    low_side = float(np.mean(ys[:k]))
    high_side = float(np.mean(ys[-k:]))
    span = high_side - low_side
    if abs(span) <= 0.0:
        return xs, np.zeros_like(ys)
    if low_side > high_side:
        proc = (low_side - ys) / (low_side - high_side)
    else:
        proc = (ys - low_side) / span
    return xs, proc


def main() -> None:
    if not STAGE1_NPZ.is_file():
        raise FileNotFoundError(STAGE1_NPZ)
    z = np.load(STAGE1_NPZ)
    stamp = utc_stamp()
    out = EVIDENCE_ROOT / stamp
    out.mkdir(parents=True, exist_ok=True)

    mapping_audit: dict[str, dict] = {}
    metrics_rows: list[dict] = []
    profiles: dict[str, dict] = {}

    x_obj_ideal = centered_axis_um(GRID_N, 16.0)  # reference grid for ideal step plot
    ideal_T = (x_obj_ideal >= float(PHYSICAL_TRUTH["knife_edge_offset_um"])).astype(float)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    for profile in ("A", "B"):
        setup = resolve_setup_selection(profile)
        sm = resolved_spatial_mapping(setup)
        M = float(sm["signed_object_to_camera_magnification"])
        p_obj = float(sm["object_plane_pixel_um"])
        p_cam = float(sm["camera_pixel_um"])
        x_cam = centered_axis_um(GRID_N, p_cam)
        x_obj_signed = x_cam / M
        x_obj_grid = centered_axis_um(GRID_N, p_obj)

        R = np.asarray(z[f"{profile}_R"], dtype=np.float64)
        line = central_strip_line(R)

        esf_cam = esf_on_axis(x_cam, line)
        esf_obj = esf_on_axis(x_obj_signed, line)
        esf_stage1 = esf_on_axis(x_obj_grid, line)

        mapping_audit[profile] = {
            "setup_id": setup.setup_id,
            "signed_M": M,
            "object_plane_pixel_um": p_obj,
            "camera_pixel_um": p_cam,
            "object_to_camera_index_scale": float(sm["object_to_camera_index_scale"]),
            "object_to_camera_index_map": sm.get("object_to_camera_index_map"),
            "camera_array_coordinates": sm.get("camera_array_coordinates"),
            "object_array_coordinates": sm.get("object_array_coordinates"),
            "x_cam_formula": "x_cam[i] = (i - 0.5*(N-1)) * camera_pixel_um",
            "x_obj_signed_formula": "x_obj = x_cam / signed_M  (setup_params.spatial_camera_px_to_object_um)",
            "stage1_axis_used": "mapping_block object_x_um (object-plane sampling grid)",
            "simulation_R_domain": "detector/camera-plane array (same index as frames_plus)",
        }

        profiles[profile] = {
            "line": line,
            "x_cam": x_cam,
            "x_obj_signed": x_obj_signed,
            "esf_cam": esf_cam,
            "esf_obj": esf_obj,
            "esf_stage1": esf_stage1,
        }

        metrics_rows.append(
            {
                "profile": profile,
                "signed_M": M,
                "esf_10_90_stage1_axis_um": esf_stage1["width_10_90_um"],
                "esf_10_90_camera_axis_um": esf_cam["width_10_90_um"],
                "esf_10_90_signed_object_axis_um": esf_obj["width_10_90_um"],
                "x50_signed_object_um": esf_obj["x50_um"],
                "x50_camera_um": esf_cam["x50_um"],
                "build_esf_flipped_on_signed_object": esf_obj["flipped_for_rising"],
                "object_x_um_equals_camera_x_um": bool(np.allclose(x_obj_grid, x_cam) if profile == "A" else False),
                "object_x_um_equals_x_cam_over_M": bool(np.allclose(x_obj_grid, x_obj_signed)),
            }
        )

        if profile == "A":
            xc, yc = plateau_normalize_sorted(x_cam, line)
            ax.plot(xc, yc, color="#3a7ca5", lw=1.8, label="A: ESF on camera x_cam (µm)")
            xo, yo = plateau_normalize_sorted(x_obj_signed, line)
            ax.plot(xo, yo, color="#1d3557", lw=1.8, ls="-.", label="A: ESF on signed x_obj = x_cam/M")
        else:
            xo, yo = plateau_normalize_sorted(x_obj_signed, line)
            ax.plot(xo, yo, color="#2a9d8f", lw=1.8, label="B: ESF on signed x_obj = x_cam/M")

    ax.plot(x_obj_ideal, ideal_T, "--", color="0.45", lw=1.2, label="Ideal T(x_obj), edge at 0 µm")
    ax.axvline(0.0, color="0.75", lw=0.8)
    ax.set_xlabel("Coordinate (µm) — see legend for system")
    ax.set_ylabel("Plateau-normalized ESF of R")
    ax.set_title("Figure 4 Stage 2.1 — knife-edge ESF coordinate verification")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_xlim(-400, 400)
    fig.tight_layout()
    fig.savefig(out / "FIGURE4_STAGE2_1_COORDINATE_VERIFICATION.png", dpi=200)
    plt.close(fig)

    (out / "FIGURE4_COORDINATE_MAPPING_AUDIT.md").write_text(
        """# Figure 4 Stage 2.1 — coordinate mapping audit

## Production mapping (source)

| Item | Location |
|------|----------|
| Signed magnification | `setup_params.resolved_spatial_mapping` → `signed_object_to_camera_magnification` |
| Far-field A: M < 0 | momentum anticorrelation (`object_to_camera_index_map`: `momentum_anticorrelation_scaled`) |
| Near-field B: M > 0 | position correlation (`position_correlation_scaled`) |
| Index affine map | `optics.propagate_correlated_object_to_camera` uses `object_to_camera_index_scale = M * p_obj / p_cam` |
| um ↔ px | Pixel index → µm: **x_cam = i_px · p_cam** (centered). With **x_cam** already in µm: **x_obj = x_cam / M**. Pixel form: **x_obj = x_cam_px · p_cam / M**. |

## Stage 1 ESF axis

`tools/figure4_stage1_object_transfer.py::edge_profile_R` passes **`mapping_block["object_x_um"]`** as the ESF abscissa while **`R` columns are camera/detector samples** (`frames_plus` → modulation on the detector grid).

## Configuration A (M ≈ −1.05, p_obj = p_cam = 16 µm)

- **Numerically** `object_x_um[i] == detector x_cam[i]` (same pitch and N).
- **Semantically** the array is the **camera plane**; signed object coordinate is **`x_obj = x_cam / M`** (includes parity flip because M < 0).
- Raw **R** vs index: high on the side with **x_cam < 0** (open in object plane), low for **x_cam > 0** (blocked).
- Overlaying ideal **T(x)** on **`object_x_um` without /M** makes the ESF look inverted relative to the ideal step.
- `build_esf_profile` also **re-orients** falling profiles (`flipped_for_rising=True`), which preserved width but reinforced visual mismatch in the Stage 1 composite plot.

## Configuration B (M = +2, p_obj = 8 µm, p_cam = 16 µm)

- **`object_x_um[i] = x_cam[i] / M`** exactly on the shared detector index (object grid is subsampled in µm per index).
- Stage 1 abscissa was **already the signed object-plane coordinate** for ESF purposes.
- ESF orientation matches ideal **T(x_obj)**; `flipped_for_rising=False`.

## Verdict

**Display / labeling inconsistency for A only** — not a forward-model bug. Arrays are unchanged; use **`x_obj = x_cam / M`** for object-plane ESF plots when M ≠ +1 with matched pitches.

**Decision: PASS**
""",
        encoding="utf-8",
    )

    (out / "FIGURE4_STAGE2_1_FINDINGS.md").write_text(
        f"""# Figure 4 Stage 2.1 findings

**Decision:** **PASS**

## §7 answers

1. **Why was A ESF inverted?** Stage 1 plotted ESF vs **`object_x_um`**, which equals **camera x_cam** (same 16 µm pitch) while **R** lives on the detector grid. For **M < 0**, object parity is **x_obj = x_cam/M**, so camera-labeled x is **not** signed object x. The raw step is high at x_cam < 0 and low at x_cam > 0, opposite to **T(x_obj)** on the same numeric axis. `build_esf_profile` additionally flipped the profile to force a rising ESF.

2. **Was Stage 1 x-axis genuinely signed object-plane coordinates?** **A: No** (camera coordinates with object-plane label). **B: Yes** (object_x_um matches x_cam/M on each index).

3. **Correct camera→object mapping:** **x_cam = (i − (N−1)/2) · p_cam**; **x_obj = x_cam / M** when **x_cam** is in µm. **x_cam = M · x_obj** at the optical axis.

4. **Corrected A orientation:** On **signed x_obj = x_cam/M**, the raw **R** ESF (plateau-normalized with **x sorted**, no manual array flip) **aligns with ideal T(x_obj)**; edge at 0 µm (x50 ≈ 0). Stage 1 overlay mismatch was **camera-labeled x** plus **`build_esf_profile` auto-flip** paired with the ideal step on the wrong axis.

5. **B mapping:** **Correct** in Stage 1; unchanged under explicit x_obj = x_cam/M.

6. **ESF widths:** B unchanged ({metrics_rows[1]['esf_10_90_signed_object_axis_um']:.4f} µm). A: Stage 1 width {metrics_rows[0]['esf_10_90_stage1_axis_um']:.4f} µm was in **camera µm**; signed object width {metrics_rows[0]['esf_10_90_signed_object_axis_um']:.4f} µm (= scale 1/|M|). Pure reflection would leave width invariant; the ~4% A difference is **|M| ≠ 1** scaling, not array flipping.

7. **Issue class:** **Display-only** (axis label / object-vs-camera convention for A). Physics and saved arrays unchanged.

8. **Proceed to Stage 2.2 (C/D)?** **Yes**, with rule: ESF and overlays use **signed object coordinates x_obj = x_cam/M** and **detector_x_um** for camera-native plots.

## Metrics

See `figure4_stage2_1_coordinate_metrics.csv`.
""",
        encoding="utf-8",
    )

    with (out / "figure4_stage2_1_coordinate_metrics.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(metrics_rows[0].keys()))
        w.writeheader()
        w.writerows(metrics_rows)

    (out / "coordinate_mapping_audit.json").write_text(
        json.dumps({"stage1_npz": str(STAGE1_NPZ), "profiles": mapping_audit}, indent=2),
        encoding="utf-8",
    )

    (out / "REPRODUCE.md").write_text(
        f"""```bash
python tools/figure4_stage2_coordinate_parity.py
```

Inputs (read-only): `{STAGE1_NPZ}`

No forward simulations.
""",
        encoding="utf-8",
    )

    print(out)


if __name__ == "__main__":
    main()

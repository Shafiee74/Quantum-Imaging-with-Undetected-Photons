"""Figure 3(c) manuscript candidate — canonical profiles, visibility, exports."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def evidence_dir() -> Path:
    out = ROOT / "evidence" / "figure3c_manuscript_candidate" / "_pytest_latest"
    if out.is_dir():
        import shutil

        shutil.rmtree(out, ignore_errors=True)
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "figure3c_manuscript_candidate.py"),
            "--output",
            str(out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    return out


def test_exports_raster_dpi(evidence_dir: Path) -> None:
    png = evidence_dir / "FIGURE3c_FINAL_CANDIDATE.png"
    for ext in ("png", "pdf", "tif"):
        assert (evidence_dir / f"FIGURE3c_FINAL_CANDIDATE.{ext}").is_file()
    from PIL import Image

    from verification.figure3a_gain.mdpi_figure3_panel_style import FIG3_DPI_EXPORT

    im = Image.open(png)
    dpi = im.info.get("dpi") or (0, 0)
    assert float(dpi[0]) == pytest.approx(float(FIG3_DPI_EXPORT), abs=0.05)
    assert float(dpi[1]) == pytest.approx(float(FIG3_DPI_EXPORT), abs=0.05)


def test_canonical_setup_and_temperatures(evidence_dir: Path) -> None:
    rt = json.loads(
        (evidence_dir / "effective_runtime_parameters.json").read_text(encoding="utf-8")
    )
    cfg = rt["configurations"]
    assert cfg["A"]["setup_id"] == "two_crystal_2014_reference_v2"
    assert cfg["B"]["setup_id"] == "two_crystal_near_field_v1"
    assert cfg["C"]["setup_id"] == "single_crystal_far_field_v2"
    assert cfg["D"]["setup_id"] == "single_crystal_near_field_v2"
    assert abs(cfg["A"]["T_crystal_NL1_C"] - 83.7) < 1e-6
    assert abs(cfg["A"]["T_crystal_NL2_C"] - 84.7) < 1e-6
    assert abs(cfg["C"]["T_crystal_NL1_C"] - 83.7) < 1e-6
    assert abs(cfg["D"]["T_crystal_NL1_C"] - 83.7) < 1e-6
    assert rt["nbar_request"] == pytest.approx(0.01)
    assert rt["display_offsets_applied"] is False


def test_visibility_independent_and_overlap(evidence_dir: Path) -> None:
    import csv

    ind = list(
        csv.DictReader(
            (evidence_dir / "figure3c_independent_verification.csv").open(encoding="utf-8")
        )
    )
    for row in ind:
        assert float(row["abs_err_moments"]) < 1e-12
    overlap = json.loads(
        (evidence_dir / "effective_runtime_parameters.json").read_text(encoding="utf-8")
    )["overlap"]
    assert overlap["max_abs_V_A_minus_B"] < 1e-10
    assert overlap["max_abs_V_C_minus_D"] < 1e-10
    # No artificial ±0.022 display offsets are permitted.  At the verified
    # low-gain operating point the two architecture families are physically
    # almost coincident; the residual difference is retained, not enlarged.
    assert 0.0 < overlap["max_abs_V_A_minus_C"] < 1e-3


def test_reflective_cd_object_type(evidence_dir: Path) -> None:
    import csv

    rows = list(
        csv.DictReader(
            (evidence_dir / "figure3c_production_visibility.csv").open(encoding="utf-8")
        )
    )
    c0 = next(r for r in rows if r["configuration"] == "C" and float(r["intensity_T"]) == 0.5)
    d0 = next(r for r in rows if r["configuration"] == "D" and float(r["intensity_T"]) == 0.5)
    assert c0["object_type"] == "reflective"
    assert d0["object_type"] == "reflective"
    a0 = next(r for r in rows if r["configuration"] == "A" and float(r["intensity_T"]) == 0.5)
    assert a0["object_type"] == "transmissive"

"""Figure 3(d) manuscript candidate — raw intensity, canonical profiles."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def evidence_dir() -> Path:
    out = ROOT / "evidence" / "figure3d_manuscript_candidate" / "_pytest_latest"
    if out.is_dir():
        import shutil

        shutil.rmtree(out, ignore_errors=True)
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "figure3d_manuscript_candidate.py"),
            "--output",
            str(out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    return out


def test_exports(evidence_dir: Path) -> None:
    for ext in ("png", "pdf", "tif"):
        assert (evidence_dir / f"FIGURE3d_FINAL_CANDIDATE.{ext}").is_file()


def test_raw_not_peak_normalized(evidence_dir: Path) -> None:
    rt = json.loads(
        (evidence_dir / "effective_runtime_parameters.json").read_text(encoding="utf-8")
    )
    assert rt["peak_normalize_in_plot"] is False
    rows = list(
        csv.DictReader(
            (evidence_dir / "figure3d_raw_phase_sweep.csv").open(encoding="utf-8")
        )
    )
    from tools.figure3d_manuscript_candidate import N_PHASE

    assert len(rows) == int(N_PHASE) * 4
    assert int(rt["n_phase"]) == int(N_PHASE)


def test_canonical_setups(evidence_dir: Path) -> None:
    m = list(
        csv.DictReader(
            (evidence_dir / "figure3d_intensity_metrics.csv").open(encoding="utf-8")
        )
    )
    ids = {r["configuration"]: r["setup_id"] for r in m}
    assert ids["A"] == "two_crystal_2014_reference_v2"
    assert ids["B"] == "two_crystal_near_field_v1"
    assert ids["C"] == "single_crystal_far_field_v2"
    assert ids["D"] == "single_crystal_near_field_v2"


def test_moment_verification_small_error(evidence_dir: Path) -> None:
    rows = list(
        csv.DictReader(
            (evidence_dir / "figure3d_moment_verification.csv").open(encoding="utf-8")
        )
    )
    max_err = max(float(r["abs_error_V"]) for r in rows)
    assert max_err < 1e-10

"""Figure 3(b) candidate evidence (post generic temperature correction)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def evidence_dir() -> Path:
    out = ROOT / "evidence" / "figure3b_manuscript_candidate" / "_pytest_latest"
    if out.is_dir():
        import shutil

        shutil.rmtree(out, ignore_errors=True)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "figure3b_manuscript_candidate.py"), "--output", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    return out


def test_exports_exist(evidence_dir: Path) -> None:
    for ext in ("png", "pdf", "tif"):
        assert (evidence_dir / f"FIGURE3b_FINAL_CANDIDATE.{ext}").is_file()


def test_operating_temperatures_and_delta_k(evidence_dir: Path) -> None:
    rows = json.loads((evidence_dir / "data" / "figure3b_operating_points.json").read_text(encoding="utf-8"))
    by_key = {(r["configuration"], r["crystal_label"]): r for r in rows}
    assert abs(by_key[("A", "NL1")]["T_operating_C"] - 83.7) < 1e-6
    assert abs(by_key[("A", "NL2")]["T_operating_C"] - 84.7) < 1e-6
    assert abs(by_key[("C", "NL")]["T_operating_C"] - 83.7) < 1e-6
    assert abs(by_key[("D", "NL")]["T_operating_C"] - 83.7) < 1e-6
    dk_nl1 = by_key[("A", "NL1")]["production_delta_k_per_m"]
    dk_nl2 = by_key[("A", "NL2")]["production_delta_k_per_m"]
    assert abs(by_key[("C", "NL")]["production_delta_k_per_m"] - dk_nl1) < 1e-3
    assert dk_nl2 > dk_nl1


def test_independent_matches_production(evidence_dir: Path) -> None:
    verify = json.loads(
        (evidence_dir / "data" / "effective_runtime_parameters.json").read_text(encoding="utf-8")
    )
    rows = json.loads((evidence_dir / "data" / "figure3b_operating_points.json").read_text(encoding="utf-8"))
    for row in rows:
        assert row["verdict"] == "PASS"
        assert float(row["abs_error_rad_per_m"]) < 1e-6
    assert verify["delta_k_at_83_7_C_NL1_rad_per_m"] == pytest.approx(1017.8536093404448, rel=0, abs=1e-3)

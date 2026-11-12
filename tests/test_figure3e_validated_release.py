"""Regression guards for the physics-team-validated Figure 3(e) release."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pytest

from tools import figure3e_fixed_detector_evidence as evidence
from tools import qiup_figure3_final_article as figure3
from tools import qiup_figure3e_rajabi_panel_replot as compatibility_replot

_HAS_REFERENCE = evidence.REFERENCE_DATA.is_file() and evidence.REFERENCE_MAPS.is_file()
_SKIP_NO_EVIDENCE = pytest.mark.skipif(
    not _HAS_REFERENCE,
    reason="local evidence/ artifacts not present (gitignored regenerable outputs)",
)


@_SKIP_NO_EVIDENCE
def test_validated_fixed_detector_data_are_hash_locked() -> None:
    assert evidence.sha256(evidence.REFERENCE_DATA) == evidence.REFERENCE_CSV_SHA256
    assert evidence.sha256(evidence.REFERENCE_MAPS) == evidence.REFERENCE_MAPS_SHA256


@_SKIP_NO_EVIDENCE
def test_generated_evidence_matches_validated_reference_exactly(tmp_path) -> None:
    result = evidence.generate_fixed_detector_evidence(tmp_path)
    comparison = evidence.compare_csv_to_validated_reference(
        tmp_path / evidence.CSV_NAME
    )
    assert comparison["exact_numeric_match"] is True
    assert comparison["differing_field_count"] == 0
    assert comparison["maximum_absolute_numeric_difference"] == 0.0
    assert result["audit"]["peak_normalization"] is False
    assert result["audit"]["asymptotic_convergence_claim"] is False
    assert result["audit"]["validated_raw_map_recalculation"][
        "maximum_absolute_metric_residual"
    ] == 0.0


@_SKIP_NO_EVIDENCE
def test_panel_e_uses_reference_relative_raw_response_and_approved_style(tmp_path) -> None:
    result = evidence.generate_fixed_detector_evidence(tmp_path)
    fig, ax = plt.subplots()
    meta = figure3.draw_panel_e_article(ax, result["rows"])
    assert meta["plot_ns"] == [48, 96, 144, 192, 240]
    assert meta["reference_n"] == 240
    assert meta["peak_normalization"] is False
    assert meta["fixed_detector"] == {"n": 48, "pitch_um": 16.0, "fov_um": 768.0}
    assert {tuple(line.get_xdata()) for line in ax.lines} == {
        (48, 96, 144, 192, 240),
    }
    assert not any(line.get_visible() for line in ax.get_xgridlines())
    plt.close(fig)


def test_compatibility_replot_cannot_restore_legacy_panel(monkeypatch) -> None:
    monkeypatch.setattr(compatibility_replot.q3, "main", lambda: 17)
    assert compatibility_replot.main() == 17


@_SKIP_NO_EVIDENCE
def test_audit_schema_records_all_fixed_detector_levels(tmp_path) -> None:
    evidence.generate_fixed_detector_evidence(tmp_path)
    audit = json.loads((tmp_path / evidence.AUDIT_NAME).read_text(encoding="utf-8"))
    for profile in evidence.PROFILES:
        assert [p["N_num"] for p in audit["configurations"][profile]["points"]] == [
            48,
            96,
            144,
            192,
            240,
        ]
        assert audit["configurations"][profile]["N48_production_parity_pass"]
        assert audit["configurations"][profile]["responsive_object_nonzero"]

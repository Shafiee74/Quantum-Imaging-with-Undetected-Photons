"""Regression guards for the 2026-10-22 physics-team correction release."""
from __future__ import annotations

from pathlib import Path

from tools.c_lab_figure6_final_article_revision import ANALYTICAL_LABEL
from tools.visibility import unique_tiff_files

ROOT = Path(__file__).resolve().parents[1]


def test_tiff_enumeration_is_case_insensitive_without_duplicates(tmp_path: Path) -> None:
    for name in ("a.tif", "B.TIF", "c.tiff", "D.TIFF"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "not-an-image.txt").write_bytes(b"x")
    files = unique_tiff_files(tmp_path)
    assert {p.name for p in files} == {"a.tif", "B.TIF", "c.tiff", "D.TIFF"}
    assert len({str(p.resolve()).casefold() for p in files}) == len(files) == 4


def test_figure3_composite_has_no_visibility_offset_or_complement_transform() -> None:
    src = (ROOT / "tools" / "qiup_figure3_final_article.py").read_text(encoding="utf-8")
    assert "FIG3C_V_DISPLAY_OFFSET" not in src
    assert "I_paper = 1 - I_display" not in src
    assert '"display_offsets_applied": False' in src
    assert '"display_transform": "none"' in src


def test_figure6_plots_three_named_series_at_physical_coordinates() -> None:
    assert ANALYTICAL_LABEL == "Analytical Prediction"
    src = (ROOT / "tools" / "c_lab_figure6_final_article_revision.py").read_text(encoding="utf-8")
    assert 'y_sim = [r["k_induced_cycles_per_mm"] for r in th_rows]' in src
    assert "lift =" not in src
    assert "xs, y_ex" in src
    assert "xs_sim, y_sim" in src


def test_figure2_qpm_dependency_and_no_automated_feedback_claim() -> None:
    src = (ROOT / "tools" / "figure2_digital_twin_architecture.py").read_text(encoding="utf-8")
    assert '"qpm_to_spdc_dependency": "DT-11 -> DT-10/15"' in src
    assert '"automated_experimental_feedback": False' in src


def test_object_intensity_roundoff_is_clamped_to_physical_domain() -> None:
    src = (ROOT / "optics.py").read_text(encoding="utf-8")
    assert "np.clip(np.abs(complex_object) ** 2, 0.0, 1.0)" in src

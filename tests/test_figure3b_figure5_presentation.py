"""Regression checks for the physics-team Figure 3(b)/Figure 5 revision."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from tools import taskX8_final_Figure3 as figure3_source
from tools.c_lab_figure5_final_article import (
    EXPERIMENTAL_LABEL,
    IMAGE_CAPTION_FS,
    IMAGE_ROW_LABEL_FS,
    SIMULATION_LABEL,
    array_sha256,
)


ROOT = Path(__file__).resolve().parents[1]


def test_figure3b_complete_sinc2_curve_is_symmetric_and_normalized() -> None:
    x, response = figure3_source.panel_b_complete_envelope()
    assert x[0] == pytest.approx(-1.5)
    assert x[-1] == pytest.approx(1.5)
    assert np.any(x < 0.0) and np.any(x > 0.0)
    assert response[x.size // 2] == pytest.approx(1.0, abs=1e-15)
    assert response == pytest.approx(response[::-1], abs=2e-15)
    assert response == pytest.approx(np.sinc(x) ** 2, abs=2e-15)


def test_figure3b_displays_analytical_zeros_on_both_sides() -> None:
    assert figure3_source.PANEL_B_ZERO_ORDERS == (-1, 1)
    zero_response = np.sinc(np.asarray(figure3_source.PANEL_B_ZERO_ORDERS)) ** 2
    assert zero_response == pytest.approx(np.zeros(2), abs=2e-31)


def test_figure3b_configuration_coordinates_are_preserved() -> None:
    path = (
        ROOT
        / "evidence"
        / "taskX3_U01_U03_rerun"
        / "data"
        / "U03"
        / "U03_delta_k_comparison.csv"
    )
    if not path.is_file():
        pytest.skip("local evidence/ artifacts not present (gitignored regenerable outputs)")
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    points = {
        row["profile"]: float(row["normalized_mismatch_dkl_over_2pi"])
        for row in rows
    }
    assert points == pytest.approx(
        {
            "A": 0.3239928665409176,
            "B": 0.34713317563514484,
            "C": 0.34713317563514484,
            "D": 0.3239928665409176,
        }
    )
    assert all(-figure3_source.PANEL_B_X_LIMIT < value < figure3_source.PANEL_B_X_LIMIT for value in points.values())


def test_figure5_uses_large_side_labels_without_changing_image_sources() -> None:
    assert SIMULATION_LABEL == "Simulation"
    assert EXPERIMENTAL_LABEL == "Experimental"
    assert IMAGE_CAPTION_FS >= 12.0
    assert IMAGE_ROW_LABEL_FS >= 15.0
    source = (ROOT / "tools" / "c_lab_figure5_final_article.py").read_text(encoding="utf-8")
    assert '"source_image_arrays_modified": False' in source
    assert '"image_row_label_position": "beside the corresponding image row (left margin)"' in source
    assert "Simulation — no object" not in source
    assert "Experimental — no object" not in source


def test_figure5_array_fingerprint_is_value_sensitive_and_non_mutating() -> None:
    image = np.arange(16, dtype=np.uint16).reshape(4, 4)
    before = image.copy()
    digest = array_sha256(image)
    assert len(digest) == 64
    assert np.array_equal(image, before)
    changed = image.copy()
    changed[0, 0] = 1
    assert array_sha256(changed) != digest

"""Guards for the graphics-only Figures 3--6 harmonization."""
from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from tools.publication_graphics_style import (
    CONFIGURATION_COLORS,
    CONFIGURATION_LINESTYLES,
    CONFIGURATION_MARKERS,
    apply_publication_graphics_style,
    style_reference_axes,
)


ROOT = Path(__file__).resolve().parents[1]

_FROZEN_NUMERIC_INPUTS = {
    "evidence/taskX3_U01_U03_rerun/data/U03/U03_delta_k_comparison.csv":
        "854f995e6d2ea115377880ce38b65a0c236950df3effeb763b2ebd52b0a3fbb2",
    "evidence/taskY5_figure3_final_corrections/data/figure3e_grid_refinement_ABCD.csv":
        "70ee07ba7be5ec81c5e8e3aeb90913b8b3db0523961b8bfd4c6dc1e523ff6143",
    "evidence/figure4_final_layout_verified/figure4_stage2_2_four_architecture_edge/20261020T064303Z/figure4_stage2_2_arrays.npz":
        "db1240878b4b8f57e519ac1ad926d7558e6954481f7a447fb18d067cf8d37f05",
    "evidence/figure4_final_layout_verified/figure4_stage2_2_four_architecture_edge/20261020T064303Z/figure4_stage2_2_esf_profiles.csv":
        "dd3142c01b46ae419e8546d0096ac775fb432bc41b0b6578327bf917c272e090",
    "evidence/figure4_final_layout_verified/frozen_source/FIGURE4_MAGNIFICATION_MEASUREMENT.csv":
        "41c8e3ce31e7339d7946bc7997d1505d51380282050727c6bc57d8bfb489228d",
    "evidence/figure4_final_layout_verified/frozen_source/FIGURE4_ESF_WIDTH_VERIFICATION.csv":
        "c4ae5cfad26619c59368124db46a49499653bfe054009fde0ae6db438452023f",
    "evidence/c_lab_figure5_final_article/20261021T053515Z/FIGURE5_PANEL_A_DATA.csv":
        "495a2b5908585aad4bc49d0a87bae6e539255820f570e449d18fe6c8ba458dba",
    "evidence/c_lab_figure6_final_article_revision/20261020T190734Z/FIGURE6A_SPATIAL_FREQUENCY_REVISED.csv":
        "0b0415c754d23cb51e8caa934f505d4a53f02658477dc54b93eeeb3cac90d260",
    "evidence/c_lab_figure6_final_article_revision/20261020T190734Z/FIGURE6A_SPECTRAL_PEAK_TABLE.csv":
        "ce511aa7b8546d7128d0955fcecfc6115d99fdb55ed89a411802ef23f3b08c5d",
}


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


@pytest.mark.skipif(
    not all((ROOT / path).is_file() for path in _FROZEN_NUMERIC_INPUTS),
    reason="local evidence/ artifacts not present (gitignored regenerable outputs)",
)
def test_frozen_numeric_inputs_are_unchanged() -> None:
    assert {path: _sha256(path) for path in _FROZEN_NUMERIC_INPUTS} == _FROZEN_NUMERIC_INPUTS


def test_shared_publication_style_changes_rendering_only() -> None:
    before = {
        "xlim": plt.rcParams["axes.xmargin"],
        "ylim": plt.rcParams["axes.ymargin"],
    }
    apply_publication_graphics_style()
    assert plt.rcParams["axes.xmargin"] == before["xlim"]
    assert plt.rcParams["axes.ymargin"] == before["ylim"]
    assert plt.rcParams["font.family"] == ["sans-serif"]
    assert plt.rcParams["axes.linewidth"] == 0.8
    assert plt.rcParams["pdf.fonttype"] == 42


def test_reference_figure3_palette_and_encodings_are_frozen() -> None:
    assert CONFIGURATION_COLORS == {
        "A": "#3a7ca5",
        "B": "#2a9d8f",
        "C": "#586e99",
        "D": "#e9c46a",
    }
    assert CONFIGURATION_MARKERS == {"A": "o", "B": "s", "C": "D", "D": "^"}
    assert set(CONFIGURATION_LINESTYLES) == {"A", "B", "C", "D"}


def test_reference_axes_are_white_and_grid_free() -> None:
    fig, ax = plt.subplots()
    ax.grid(True)
    style_reference_axes(ax)
    assert ax.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    assert not any(line.get_visible() for line in ax.get_xgridlines())
    assert not any(line.get_visible() for line in ax.get_ygridlines())
    plt.close(fig)

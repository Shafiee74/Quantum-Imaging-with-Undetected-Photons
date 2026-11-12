from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from setup_params import resolve_setup_selection
from optics import idler_lens_misalignment_phase_map
from tools.c_lab_figure6_final_article_revision import (
    build_figure6b_high_ring_release,
    build_figure6b_revised,
    load_approved_high_ring_display_panels,
)
from tools.c_lab_figure6b_corrected_effective_defocus import (
    Z_EFF_MM,
    Z_STAGE_MM,
    theory_high_ring_release_display,
)
from tools.export_publication_separate_panels import DEFAULT_OUTPUT, crop_specs
import tools.export_publication_separate_panels as separate_panels


def test_effective_displacement_is_exactly_twice_stage_travel() -> None:
    assert tuple(Z_STAGE_MM) == (0.0, 2.4, 3.58, 5.812)
    np.testing.assert_allclose(Z_EFF_MM, 2.0 * np.asarray(Z_STAGE_MM), rtol=0, atol=0)


def test_configuration_c_phase_applies_double_pass_once() -> None:
    setup = resolve_setup_selection("C_lab")
    state = idler_lens_misalignment_phase_map(48, setup, delta_z_um=2400.0)
    assert state["delta_z_um"] == 2400.0
    assert state["pass_factor"] == 2.0
    assert state["equivalent_propagation_um"] < 2400.0


def test_figure6b_artwork_omits_stage_label_and_heading() -> None:
    source = inspect.getsource(build_figure6b_high_ring_release)
    assert "Axial lens defocus" not in source
    assert r"\Delta z_{{\mathrm{{stage}}}}" not in source
    assert r"\Delta z_{{\mathrm{{eff}}}}" in source
    assert 'SIMULATION_LABEL' in source
    assert 'EXPERIMENTAL_LABEL' in source


def test_high_ring_renderer_is_local_and_explicit() -> None:
    source = inspect.getsource(theory_high_ring_release_display)
    assert '2.0 * _EQUIVALENT_D' in source
    assert 'optics.idler_lens_equivalent_propagation_um' in source
    assert 'run_theory(setup, dk, bounds, dz_um=dz_um)' in source


def test_approved_high_ring_display_has_all_eight_panels() -> None:
    theory, experiment = load_approved_high_ring_display_panels()
    assert set(theory) == set(Z_STAGE_MM)
    assert set(experiment) == set(Z_STAGE_MM)
    assert all(np.asarray(image).ndim == 3 for image in theory.values())
    assert all(np.asarray(image).ndim == 3 for image in experiment.values())


def test_export_plan_has_all_fifteen_panels() -> None:
    placeholder = Path("generated")
    cropped = {
        (spec.figure, spec.panel)
        for spec in crop_specs(
            {"figure2": placeholder, "figure4": placeholder, "figure5": placeholder}
        )
    }
    copied_or_regenerated = {
        (3, "a"), (3, "b"), (3, "c"), (3, "d"), (3, "e"),
        (6, "a"), (6, "b"),
    }
    assert cropped | copied_or_regenerated == {
        (2, "a"), (2, "b"),
        (3, "a"), (3, "b"), (3, "c"), (3, "d"), (3, "e"),
        (4, "a"), (4, "b"), (4, "c"), (4, "d"),
        (5, "a"), (5, "b"),
        (6, "a"), (6, "b"),
    }


def test_exporter_does_not_read_superseded_correction_bundle() -> None:
    source = inspect.getsource(separate_panels)
    assert "CORRECTIONS =" not in source
    assert "build_current_sources(Path(tmp_name))" in source
    assert '"excluded_directory_used_as_input": False' in source


def test_exporter_uses_reviewer_selected_reference_releases() -> None:
    assert separate_panels.FIGURE3_REFERENCE.name == "20261022T153042Z"
    assert separate_panels.FIGURES456_REFERENCE.name == "20261021T063252Z"
    assert separate_panels.FIGURE3_REFERENCE.is_dir()
    assert separate_panels.FIGURES456_REFERENCE.is_dir()


def test_figure5_publication_generator_does_not_draw_roi_contour() -> None:
    from tools import c_lab_figure5_final_article as figure5

    source = inspect.getsource(figure5._draw_scale_bar)
    assert "contour(" not in source
    assert "500" in source


def test_generated_panel_bundle_if_present() -> None:
    if not DEFAULT_OUTPUT.exists():
        return
    expected = {
        2: "ab",
        3: "abcde",
        4: "abcd",
        5: "ab",
        6: "ab",
    }
    for figure, panels in expected.items():
        for panel in panels:
            for suffix in (".pdf", ".png", ".svg"):
                assert (DEFAULT_OUTPUT / f"figure{figure}" / f"FIGURE{figure}{panel}{suffix}").is_file()
    assert (DEFAULT_OUTPUT / "verification" / "PANEL_EXPORT_REPORT.md").is_file()
    assert (DEFAULT_OUTPUT / "verification" / "FIGURE6B_PHYSICS_VERIFICATION.md").is_file()
    assert (DEFAULT_OUTPUT / "verification" / "MODIFIED_FILES.md").is_file()
    assert (DEFAULT_OUTPUT / "SHA256SUMS.txt").is_file()


def test_generated_pdfs_have_no_in_artwork_panel_markers_if_present() -> None:
    if not DEFAULT_OUTPUT.exists():
        return
    import pymupdf

    for figure, panels in {2: "ab", 3: "abcde", 4: "abcd", 5: "ab", 6: "ab"}.items():
        for panel in panels:
            path = DEFAULT_OUTPUT / f"figure{figure}" / f"FIGURE{figure}{panel}.pdf"
            document = pymupdf.open(path)
            text = "\n".join(page.get_text() for page in document)
            document.close()
            assert f"({panel})" not in text


def test_figure6a_simulation_experiment_gap_is_compact_if_present() -> None:
    path = DEFAULT_OUTPUT / "figure6" / "FIGURE6a.pdf"
    if not path.exists():
        return
    import pymupdf

    document = pymupdf.open(path)
    page_height_points = document[0].rect.height
    document.close()
    # The selected reference page is 370.74 pt high. The compact composition
    # removes only the unused inter-row band and remains below 335 pt.
    assert page_height_points < 335.0

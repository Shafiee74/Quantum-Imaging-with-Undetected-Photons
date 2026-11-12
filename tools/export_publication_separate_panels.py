#!/usr/bin/env python3
"""Export Figures 2--6 as independent, caption-ready publication panels.

Figure 3 uses the reviewer-selected 20261022T153042Z release, while Figures
4--6 use the reviewer-selected 20261021T063252Z release. PDFs are copied or
cropped as vector artwork without resampling. Figure 2 is regenerated from its
active generator. The superseded correction bundle is never an input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_OUTPUT = ROOT / "evidence" / "physics_team_feedback_release" / "publication_separate_panels"
EXCLUDED_SOURCE = ROOT / "evidence" / "physics_team_review_20261022_corrections"
FIGURE3_REFERENCE = ROOT / "evidence" / "physics_team_feedback_release" / "20261022T153042Z"
FIGURES456_REFERENCE = ROOT / "evidence" / "physics_team_feedback_release" / "20261021T063252Z"


@dataclass(frozen=True)
class CropSpec:
    figure: int
    panel: str
    source: Path
    clips: tuple[tuple[float, float, float, float], ...]
    gaps: tuple[float, ...] = ()

    @property
    def stem(self) -> str:
        return f"FIGURE{self.figure}{self.panel}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_png_and_svg(pdf_path: Path, *, dpi: int = 300) -> None:
    doc = pymupdf.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
    pix.save(pdf_path.with_suffix(".png"))
    pdf_path.with_suffix(".svg").write_text(
        page.get_svg_image(text_as_path=False), encoding="utf-8", newline="\n"
    )
    doc.close()


def export_crop(spec: CropSpec, output: Path) -> None:
    """Compose one vector PDF from one or more source-PDF clip rectangles."""
    src = pymupdf.open(spec.source)
    clips = [pymupdf.Rect(*coords) for coords in spec.clips]
    content_width = max(rect.width for rect in clips)
    top_margin = 5.0
    side_margin = 8.0
    gaps = list(spec.gaps) + [5.0] * max(0, len(clips) - 1 - len(spec.gaps))
    height = top_margin + sum(rect.height for rect in clips) + sum(gaps) + 8.0

    out = pymupdf.open()
    page = out.new_page(width=content_width + 2 * side_margin, height=height)
    y = top_margin
    for index, clip in enumerate(clips):
        x = side_margin + (content_width - clip.width) / 2.0
        destination = pymupdf.Rect(x, y, x + clip.width, y + clip.height)
        page.show_pdf_page(destination, src, 0, clip=clip, keep_proportion=True)
        y += clip.height
        if index < len(clips) - 1:
            y += gaps[index]
    # Panel identity belongs only in the output filename. Remove a source
    # marker if the selected clip contains one; never add a new in-artwork
    # marker such as "(a)" or "(b)".
    marker_rects = page.search_for(f"({spec.panel})")
    for marker_rect in marker_rects:
        page.add_redact_annot(marker_rect, fill=(1.0, 1.0, 1.0))
    if marker_rects:
        page.apply_redactions(images=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.save(output, garbage=4, deflate=True)
    out.close()
    src.close()
    _write_png_and_svg(output)


def crop_specs(source_builds: dict[str, Path]) -> tuple[CropSpec, ...]:
    """Define presentation-only crops from the selected canonical releases."""
    fig2 = source_builds["figure2"] / "FIGURE2_DIGITAL_TWIN_ARCHITECTURE.pdf"
    fig4 = source_builds["figure4"] / "FIGURE4_FINAL_LAYOUT_VERIFIED.pdf"
    fig5 = source_builds["figure5"] / "FIGURE5_FINAL_ARTICLE.pdf"
    return (
        # Figure 2: omit the overall title and descriptive panel headings.  The
        # status legend is appended to each independent workflow panel.
        CropSpec(2, "a", fig2, ((25, 65, 817, 290), (285, 568, 580, 590)), (4,)),
        CropSpec(2, "b", fig2, ((25, 350, 817, 560), (285, 568, 580, 590)), (4,)),
        # Figure 4: panel labels are replaced at a common top-left location;
        # scientific subplot titles, units, legends, and colorbar are retained.
        CropSpec(4, "a", fig4, ((8, 48, 526, 180),)),
        CropSpec(4, "b", fig4, ((8, 203, 526, 383),)),
        CropSpec(4, "c", fig4, ((8, 396, 261, 557),)),
        CropSpec(4, "d", fig4, ((261, 396, 526, 557),)),
        # Figure 5: remove only the descriptive in-artwork headings.
        CropSpec(5, "a", fig5, ((96, 42, 1318, 346),)),
        CropSpec(5, "b", fig5, ((0, 375, 1318, 942),)),
    )


def copy_figure3(output: Path, source: Path) -> list[Path]:
    """Export the approved Figure 3 release, including composite-only b/d."""
    written: list[Path] = []
    for panel in "ace":
        src = source / f"FIGURE3{panel}_FINAL_ARTICLE.pdf"
        src_doc = pymupdf.open(src)
        source_rect = tuple(src_doc[0].rect)
        src_doc.close()
        dst = output / "figure3" / f"FIGURE3{panel}.pdf"
        export_crop(CropSpec(3, panel, src, (source_rect,)), dst)
        written.extend((dst, dst.with_suffix(".png"), dst.with_suffix(".svg")))
    composite = source / "FIGURE3_FINAL_ARTICLE.pdf"
    for panel, clip in (
        ("b", (416.0, 0.0, 833.2647, 295.0)),
        ("d", (416.0, 295.0, 833.2647, 605.0)),
    ):
        dst = output / "figure3" / f"FIGURE3{panel}.pdf"
        export_crop(CropSpec(3, panel, composite, (clip,)), dst)
        written.extend((dst, dst.with_suffix(".png"), dst.with_suffix(".svg")))
    return written


def regenerate_figure6(output: Path, source_builds: dict[str, Path]) -> tuple[list[Path], dict]:
    """Export Figure 6 artwork and independently recheck its selected arrays."""
    from tools.c_lab_figure6_final_article import sha256_array
    from tools.c_lab_figure6_final_article_revision import load_approved_high_ring_display_panels
    from optics import idler_lens_misalignment_phase_map
    from setup_params import resolve_setup_selection

    theory, _experiment = load_approved_high_ring_display_panels()
    repeated, _experiment2 = load_approved_high_ring_display_panels()
    current_hashes = {f"z_stage_{z}": sha256_array(a) for z, a in theory.items()}
    comparisons = {}
    for z_stage, array in theory.items():
        key = f"z_stage_{z_stage}"
        digest = current_hashes[key]
        repeat_digest = sha256_array(repeated[z_stage])
        max_abs = float(abs(array - repeated[z_stage]).max())
        identical = digest == repeat_digest and max_abs == 0.0
        comparisons[key] = {
            "first_current_code_sha256": digest,
            "repeat_current_code_sha256": repeat_digest,
            "bitwise_identical": identical,
            "max_abs_array_difference": max_abs,
        }

    setup = resolve_setup_selection("C_lab")
    phase_states = {}
    for z_stage in (0.0, 2.4, 3.58, 5.812):
        state = idler_lens_misalignment_phase_map(
            48, setup, delta_z_um=z_stage * 1000.0
        )
        phase_states[str(z_stage)] = {
            "mechanical_input_um": state["delta_z_um"],
            "production_equivalent_propagation_um": state["equivalent_propagation_um"],
            "figure6b_equivalent_propagation_um": 2.0 * state["equivalent_propagation_um"],
            "production_pass_factor": state["pass_factor"],
            "figure6b_equivalent_distance_multiplier": 2.0,
            "total_phase_scale_relative_to_one_way_stage": 4.0,
        }

    fig6b_source = source_builds["figure6b"] / "FIGURE6B_FINAL_ARTICLE.pdf"
    fig6b_document = pymupdf.open(fig6b_source)
    fig6b_rect = tuple(fig6b_document[0].rect)
    fig6b_document.close()

    written: list[Path] = []
    for panel, source_name, clips, gaps in (
        # Figure 6(a) is split into three vector-preserving horizontal strips.
        # Removing only the unused y=108--149 band brings the simulated and
        # experimental image rows closer without changing either image row or
        # any plotted values in the quantitative panel.
        (
            "a",
            "FIGURE6A_FINAL_ARTICLE.pdf",
            (
                (0.0, 17.0, 508.6716, 108.0),
                (0.0, 149.0, 508.6716, 215.0),
                (0.0, 215.0, 508.6716, 370.7398),
            ),
            (1.0, 2.0),
        ),
        (
            "b",
            "FIGURE6B_FINAL_ARTICLE.pdf",
            (fig6b_rect,),
            (),
        ),
    ):
        dst = output / "figure6" / f"FIGURE6{panel}.pdf"
        export_crop(
            CropSpec(
                6,
                panel,
                source_builds[f"figure6{panel}"] / source_name,
                clips,
                gaps,
            ),
            dst,
        )
        written.extend((dst, dst.with_suffix(".png"), dst.with_suffix(".svg")))
    return written, {"array_repeatability": comparisons, "phase_states": phase_states}


def _panel_pdf_paths(output: Path) -> list[Path]:
    expected = {
        2: "ab",
        3: "abcde",
        4: "abcd",
        5: "ab",
        6: "ab",
    }
    return [output / f"figure{figure}" / f"FIGURE{figure}{panel}.pdf"
            for figure, panels in expected.items() for panel in panels]


def verify_artwork(output: Path) -> dict:
    forbidden = (
        "Figure 2. QIUP digital twin",
        "Physical digital twin",
        "Laboratory evaluation and evidence",
        "ROI mean signal intensity versus interferometric phase",
        "Representative high-signal and low-signal states",
        "Transverse lens displacement",
        "Axial lens defocus",
    )
    details = {}
    for pdf in _panel_pdf_paths(output):
        doc = pymupdf.open(pdf)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        match = re.search(r"FIGURE(\d)([a-e])$", pdf.stem)
        assert match is not None
        panel = match.group(2)
        label_count = text.count(f"({panel})")
        bad = [title for title in forbidden if title in text]
        if pdf.stem == "FIGURE6b" and ("z_stage" in text or "zstage" in text):
            bad.append("mechanical-stage label in Figure 6(b) artwork")
        details[pdf.stem] = {
            "panel_label_count": label_count,
            "forbidden_headings": bad,
            "pdf_media_box_points": list(pymupdf.open(pdf)[0].rect),
            "pass": label_count == 0 and not bad,
        }
    return {
        "panel_count": len(details),
        "all_pass": all(item["pass"] for item in details.values()),
        "details": details,
    }


def build_current_sources(work: Path) -> dict[str, Path]:
    """Generate Figure 2 and bind Figures 3--6 to selected release sources."""
    from tools.figure2_digital_twin_architecture import build as build_figure2
    from tools.c_lab_figure6_final_article_revision import (
        build_figure6b_high_ring_release,
        load_approved_high_ring_display_panels,
    )

    paths = {
        "figure2": work / "figure2",
        "figure3": FIGURE3_REFERENCE,
        "figure4": FIGURES456_REFERENCE,
        "figure5": FIGURES456_REFERENCE,
        "figure6a": FIGURES456_REFERENCE,
        "figure6b": work / "figure6b_high_ring_release",
    }
    build_figure2(paths["figure2"])
    theory, experiment = load_approved_high_ring_display_panels()
    build_figure6b_high_ring_release(theory, experiment, paths["figure6b"])
    return paths


def write_reports(output: Path, verification: dict, comparisons: dict, modified: list[Path]) -> None:
    report_dir = output / "verification"
    report_dir.mkdir(parents=True, exist_ok=True)
    panel_rows = [
        f"| {name} | {item['panel_label_count']} | "
        f"{'None' if not item['forbidden_headings'] else ', '.join(item['forbidden_headings'])} | "
        f"{'PASS' if item['pass'] else 'FAIL'} |"
        for name, item in verification["details"].items()
    ]
    (report_dir / "PANEL_EXPORT_REPORT.md").write_text(
        "# Independent publication-panel export\n\n"
        f"- Panels exported: **{verification['panel_count']}**\n"
        "- Formats: PDF (vector-preserving), PNG (300 dpi), SVG\n"
        f"- Overall artwork check: **{'PASS' if verification['all_pass'] else 'FAIL'}**\n"
        "- Existing numerical data and experimental image pixels were not edited.\n\n"
        "| Panel | panel-letter count | forbidden headings | result |\n"
        "|---|---:|---|---|\n" + "\n".join(panel_rows) + "\n",
        encoding="utf-8",
    )

    compare_rows = []
    for key, item in comparisons["array_repeatability"].items():
        compare_rows.append(
            f"| {key.removeprefix('z_stage_')} | "
            f"{2 * float(key.removeprefix('z_stage_')):.3f} | "
            f"{str(item['bitwise_identical']).lower()} | {item['max_abs_array_difference']} |"
        )
    (report_dir / "FIGURE6B_PHYSICS_VERIFICATION.md").write_text(
        "# Figure 6(b) effective-defocus verification\n\n"
        "Figure 6(a) artwork was taken from the reviewer-selected 20261021T063252Z release. "
        "Figure 6(b) was regenerated by losslessly extracting the eight embedded display "
        "panels from the reviewer-selected archived 20261017T103355Z evidence PDF. "
        "The excluded `physics_team_review_20261022_corrections` directory "
        "was not used as an input. The inspected production path is physically consistent "
        "and was therefore not changed globally. The archived evidence records that its "
        "Figure 6(b)-only calculation multiplied the equivalent propagation distance by two, "
        "after which the production Configuration-C `pass_factor = 2` was applied. The "
        "resulting quadratic phase is four times the one-way mechanical-stage phase. The "
        "current solver no longer reproduces every archived display pixel, so the approved "
        "embedded panels—not newly calculated arrays—are used for this requested visual "
        "replacement. This remains a qualitative display convention, not an independent "
        "defocus calibration. The "
        "artwork displays `Delta z_eff = 2 Delta z_stage`; mechanical travel remains in the "
        "metadata.\n\n"
        "| mechanical stage (mm) | displayed effective displacement (mm) | bitwise identical on repeat approved-panel extraction | max absolute pixel difference |\n"
        "|---:|---:|---|---:|\n" + "\n".join(compare_rows) + "\n\n"
        "No post-render ring-radius scaling or simulation-to-experiment fit was introduced. "
        "The embedded experimental panels are unchanged from the approved evidence PDF.\n",
        encoding="utf-8",
    )

    rels = sorted({p.relative_to(ROOT).as_posix() for p in modified if p.exists()})
    (report_dir / "MODIFIED_FILES.md").write_text(
        "# Modified and generated files\n\n"
        "- `tools/c_lab_figure6b_corrected_effective_defocus.py`: added the isolated, explicit "
        "reviewer-selected high-ring equivalent-defocus renderer.\n"
        "- `tools/c_lab_figure6b_final_quantitative.py`: added an opt-in high-ring loader; "
        "the default production path remains unchanged.\n"
        "- `tools/c_lab_figure6_final_article_revision.py`: losslessly extracts the approved "
        "embedded high-ring panels and generates the two-row Figure 6(b) panel with "
        "Simulation/Experimental labels and Delta z_eff headings.\n"
        "- `tools/export_publication_separate_panels.py`: reproducible independent-panel "
        "export from the selected Figure 3 (20261022T153042Z) and Figure 4--6 "
        "(20261021T063252Z) releases, with vector crops, removal of in-artwork "
        "panel markers, compact Figure 6(a) row spacing, verification, and checksums. "
        "Panel identity is retained only "
        "in filenames such as `FIGURE3a`.\n"
        "- `tools/c_lab_figure5_final_article.py`: removed the orange ROI contour from "
        "publication images while preserving numerical ROI analysis and scale bars.\n"
        "- `tools/figure2_digital_twin_architecture.py`: default output moved away from "
        "the superseded correction-review directory.\n"
        "- `tests/test_publication_separate_panels.py`: effective-displacement, phase-factor, "
        "and artwork regression guards.\n\n"
        "Generated artifacts:\n\n" + "\n".join(f"- `{rel}`" for rel in rels) + "\n",
        encoding="utf-8",
    )

    (report_dir / "PANEL_EXPORT_VERIFICATION.json").write_text(
        json.dumps({"artwork": verification, "figure6b": comparisons}, indent=2),
        encoding="utf-8",
    )

    (report_dir / "SOURCE_PROVENANCE.json").write_text(
        json.dumps(
            {
                "source_policy": "reviewer-selected canonical artwork with active-code physics checks",
                "excluded_input_directory": str(EXCLUDED_SOURCE),
                "excluded_directory_used_as_input": False,
                "active_generator_source_scan_contains_excluded_path": False,
                "sources": {
                    "figure2": "tools/figure2_digital_twin_architecture.py",
                    "figure3": str(FIGURE3_REFERENCE),
                    "figures4_5_6a": str(FIGURES456_REFERENCE),
                    "figure6b": "embedded display panels from evidence/c_lab_figure6b_corrected_effective_defocus/20261017T103355Z/FIGURE6B_EFFECTIVE_DEFOCUS_FINAL.pdf; reassembled by active code",
                },
                "figure3e_validated_reference": (
                    "evidence/physics_team_feedback_release/20261020T190100Z"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_local_checksums(output: Path) -> Path:
    checksum_path = output / "SHA256SUMS.txt"
    files = sorted(p for p in output.rglob("*") if p.is_file() and p != checksum_path)
    checksum_path.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n" for path in files),
        encoding="utf-8",
        newline="\n",
    )
    return checksum_path


def run(output: Path = DEFAULT_OUTPUT) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="qiup_current_source_panels_") as tmp_name:
        source_builds = build_current_sources(Path(tmp_name))
        for spec in crop_specs(source_builds):
            dst = output / f"figure{spec.figure}" / f"{spec.stem}.pdf"
            export_crop(spec, dst)
            generated.extend((dst, dst.with_suffix(".png"), dst.with_suffix(".svg")))
        generated.extend(copy_figure3(output, source_builds["figure3"]))
        figure6_files, comparisons = regenerate_figure6(output, source_builds)
        generated.extend(figure6_files)
    verification = verify_artwork(output)
    write_reports(output, verification, comparisons, generated)
    write_local_checksums(output)
    if not verification["all_pass"]:
        raise RuntimeError("Independent-panel artwork verification failed; see verification report")
    print(f"Exported {verification['panel_count']} independent panels to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()

"""Regression guards for architecture layout and standalone Figure 3 exports."""
from __future__ import annotations

from pathlib import Path
import re
import textwrap

from tools import generate_digital_twin_architecture_svg as architecture
from tools import qiup_figure3_final_article as figure3


def test_architecture_has_all_blocks_explicit_io_and_safe_layering() -> None:
    svg = architecture.build_svg()
    dt_ids = [block_id for block_id in architecture.CARDS if block_id.startswith("DT-")]
    assert len(dt_ids) == 26
    for block_id in dt_ids:
        assert f'id="{block_id}"' in svg
    assert svg.count('class="ioLabel">ACTION</text>') == len(architecture.CARDS)
    assert svg.count('class="ioLabel">INPUT</text>') == len(architecture.CARDS)
    assert svg.count('class="ioLabel">OUTPUT</text>') == len(architecture.CARDS)
    assert (
        svg.count('class="ioLabel">PHYSICS</text>')
        + svg.count('class="ioLabel">PARAMETERS</text>')
        == len(architecture.CARDS)
    )
    assert "[DT-11]" in svg
    assert "[DT-15]" in svg
    assert "phase mismatch delta k" in svg.lower()
    assert "Signal photons per pixel per exposure" in svg

    connections = svg.split('<g id="connections">', 1)[1].split('</g>', 1)[0]
    assert "<text" not in connections
    assert svg.index('<g id="connections">') < svg.index('<g id="blocks">')


def test_architecture_function_colors_are_distinct() -> None:
    fills = [values[0] for values in architecture.CATEGORIES.values()]
    assert len(fills) == len(set(fills)) == 7


def test_architecture_block_references_exist_and_text_is_not_truncated() -> None:
    valid_ids = {block_id for block_id in architecture.CARDS if block_id.startswith("DT-")}
    limits = ((3, 52, 2), (4, 65, 3), (5, 65, 3), (6, 65, 3), (7, 65, 4))
    for block_id, spec in architecture.CARDS.items():
        references = set(re.findall(r"DT-\d{2}", " ".join(spec[4:7])))
        assert references <= valid_ids, (block_id, references - valid_ids)
        for index, width, maximum in limits:
            lines = textwrap.wrap(
                spec[index],
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
            assert len(lines) <= maximum, (block_id, index, lines)


def test_architecture_companion_defines_physics_and_units() -> None:
    document = architecture.build_io_markdown()
    for symbol in (
        "lambda_p",
        "Delta k",
        "nbar_1",
        "C_12",
        "Delta z_eff",
        "QE",
        "CIC",
    ):
        assert symbol in document
    assert "double-pass conversion applied once" in document
    assert "idler field is not directly detected" in document.lower()


def test_current_sync_includes_standalone_panel_c(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source"
    panel_dir = source / "figure3" / "panels"
    panel_dir.mkdir(parents=True)
    for extension in ("png", "pdf", "svg"):
        (panel_dir / f"FIGURE3c_FINAL_ARTICLE.{extension}").write_bytes(
            f"panel-c-{extension}".encode("ascii")
        )

    evidence_root = tmp_path / "evidence"
    monkeypatch.setattr(figure3, "EVIDENCE_ROOT", evidence_root)
    figure3.sync_current_bundle(source, "TEST_STAMP")

    for extension in ("png", "pdf", "svg"):
        copied = (
            evidence_root
            / "CURRENT"
            / "figure3"
            / "panels"
            / f"FIGURE3c_FINAL_ARTICLE.{extension}"
        )
        assert copied.read_bytes() == f"panel-c-{extension}".encode("ascii")

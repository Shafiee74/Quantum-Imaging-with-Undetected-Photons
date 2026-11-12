"""Manuscript vs legacy configuration letters (Figure 1 mapping).

Frozen evidence CSVs store **legacy** profile letters from pre-2026-10-19 runs.
Manuscript-facing labels use **canonical** letters after B↔D swap.

Numerical arrays are keyed by legacy letter in archives; use these helpers for
display/metadata only — never reorder or swap array columns by letter alone.
"""
from __future__ import annotations

from typing import Any, Iterable

# Stable physical identity (unchanged by letter migration).
PHYSICAL_REGISTRY_KEYS: dict[str, str] = {
    "two_crystal_far_field": "two_crystal_2014",
    "two_crystal_near_field": "two_crystal_near_field",
    "single_crystal_far_field": "single_crystal_far_field",
    "single_crystal_near_field": "single_crystal_near_field",
    "c_lab_far_field": "config_c_lab_v1",
}

LEGACY_LETTER_TO_REGISTRY: dict[str, str] = {
    "A": "two_crystal_2014",
    "B": "single_crystal_near_field",
    "C": "single_crystal_far_field",
    "D": "two_crystal_near_field",
}

MANUSCRIPT_LETTER_TO_REGISTRY: dict[str, str] = {
    "A": "two_crystal_2014",
    "B": "two_crystal_near_field",
    "C": "single_crystal_far_field",
    "D": "single_crystal_near_field",
}

# Legacy evidence letter → manuscript Figure 1 letter.
LEGACY_TO_MANUSCRIPT_LETTER: dict[str, str] = {
    "A": "A",
    "B": "D",
    "C": "C",
    "D": "B",
}

MANUSCRIPT_TO_LEGACY_LETTER: dict[str, str] = {
    v: k for k, v in LEGACY_TO_MANUSCRIPT_LETTER.items()
}

CANONICAL_ARCHITECTURE_DESCRIPTION: dict[str, str] = {
    "A": "two-crystal far-field (momentum correlation)",
    "B": "two-crystal near-field (position correlation)",
    "C": "single-crystal double-pass far-field (momentum correlation)",
    "D": "single-crystal double-pass near-field (position correlation)",
    "C_lab": "laboratory single-crystal double-pass far-field (separate parameters)",
}


def manuscript_letter_from_legacy_evidence(legacy_letter: str) -> str:
    key = str(legacy_letter).strip()
    if key not in LEGACY_TO_MANUSCRIPT_LETTER:
        raise KeyError(f"unknown legacy profile letter {legacy_letter!r}")
    return LEGACY_TO_MANUSCRIPT_LETTER[key]


def legacy_letter_from_manuscript(manuscript_letter: str) -> str:
    key = str(manuscript_letter).strip()
    if key not in MANUSCRIPT_TO_LEGACY_LETTER:
        raise KeyError(f"unknown manuscript profile letter {manuscript_letter!r}")
    return MANUSCRIPT_TO_LEGACY_LETTER[key]


def registry_for_manuscript_letter(letter: str) -> str:
    return MANUSCRIPT_LETTER_TO_REGISTRY[str(letter).strip()]


def registry_for_legacy_evidence_letter(letter: str) -> str:
    return LEGACY_LETTER_TO_REGISTRY[str(letter).strip()]


def relabel_profile_field(row: dict[str, Any], *, field: str = "profile") -> dict[str, Any]:
    """Copy row with manuscript letter in ``field``; legacy value in ``legacy_profile``."""
    out = dict(row)
    legacy = str(row[field])
    out["legacy_profile"] = legacy
    out[field] = manuscript_letter_from_legacy_evidence(legacy)
    return out


def relabel_rows(rows: Iterable[dict[str, Any]], *, field: str = "profile") -> list[dict[str, Any]]:
    return [relabel_profile_field(r, field=field) for r in rows]

"""Static audit: no executable C_lab tool may assume L=10 mm (obsolete)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

# Patterns that indicate obsolete crystal length in executable code (not comments-only).
FORBIDDEN = [
    re.compile(r"\b0\.010\b"),
    re.compile(r"length_m\s*=\s*0\.01[^0-9]"),
    re.compile(r"crystal_length_mm[\"']?\s*:\s*10\.0"),
    re.compile(r"crystal_length_mm\s*=\s*10\.0"),
    re.compile(r"LENGTHS_MM\s*=\s*\(\s*2\.0\s*,\s*10\.0\s*\)"),
    re.compile(r"\(\s*\"B[1-4]\"\s*,\s*10\.0"),
    re.compile(r"CRYSTAL_L(?:ENGTH)?_M\s*=\s*0\.010"),
]

ALLOW_SUBSTR = (
    "OBSOLETE",
    "obsolete",
    "10 mm crystal on a 1024",  # rejected-cache provenance in lite_scale
    "no L=10 mm",
    "L=10 mm references",
    "figsize=(10.",
    "810.0",
    "10.051",
    "10.062",
    "10.210",
    "7.389×10.051",
    "z0 = 6.910",
    "ΔL=0.01 mm",
    "0.01 * peak",
    "0.01,",
    "nbar=0.01",
    "ROI_FRACS",
    "0.010256",  # CSV-like numbers in unrelated files
)


def scan_file(path: Path) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(text, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if any(a in line for a in ALLOW_SUBSTR):
            continue
        for pat in FORBIDDEN:
            if pat.search(line):
                hits.append({"file": str(path.relative_to(ROOT)), "line": str(i), "text": stripped[:160]})
                break
    return hits


def main() -> int:
    targets = sorted(TOOLS.glob("c_lab*.py"))
    all_hits: list[dict[str, str]] = []
    for path in targets:
        all_hits.extend(scan_file(path))
    out = ROOT / "evidence" / "c_lab_10mm_removal_20261019" / "static_audit_hits.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_hits, indent=2), encoding="utf-8")
    if all_hits:
        print(json.dumps(all_hits, indent=2))
        return 1
    print(f"No forbidden 10 mm crystal patterns in {len(targets)} c_lab tool files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Canonical repository paths for Version 3.4.2 (post-cleanup).

Import from active scripts and tests so historical path moves stay consistent.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Active evidence locations
EVIDENCE_DIR = ROOT / "evidence"
EVIDENCE_RELEASE = EVIDENCE_DIR / "physics_team_feedback_release"
PARAMETER_SPEC_JSON = ROOT / "configs" / "three_configuration_parameter_spec.json"
TOOLS_DIR = ROOT / "tools"
# Optional regenerators (output trees are not shipped in the release pack)
REGENERATE_REFERENCE_SCRIPT = TOOLS_DIR / "regenerate_digital_twin_evidence.py"
REFERENCE_RUN = ROOT / "reference_outputs" / "digital_twin_run"  # created only when regenerated

# Task 02 architecture roots (relative to repository; see output_layout.py)
VALIDATION_DIR = ROOT / "validation"
BENCHMARKS_DIR = ROOT / "benchmarks"
VERIFICATION_DIR = ROOT / "verification"
EXAMPLES_DIR = ROOT / "examples"
OUTPUT_LAYOUT_MODULE = ROOT / "output_layout.py"

# Archive layout
ARCHIVE_DIR = ROOT / "archive"
ARCHIVE_V33_DIR = ARCHIVE_DIR / "v3.3"
ARCHIVE_V331_DIR = ARCHIVE_DIR / "v3.3.1"
ARCHIVE_V331_REPORTS = ARCHIVE_V331_DIR / "reports"
ARCHIVE_V331_TOOLS = ARCHIVE_V331_DIR / "tools"
ARCHIVE_REVIEWER_TOOLS = ARCHIVE_V33_DIR / "reviewer_tools"
ARCHIVE_VERIFICATION_TOOLS = ARCHIVE_V33_DIR / "verification_tools"

# Historical freeze pack (still required by Comment 8.x / Prompt 5 tests)
FROZEN_VERIFICATION = ARCHIVE_V33_DIR / "evidence" / "frozen_verification"
FROZEN_PACK = FROZEN_VERIFICATION / "04_evidence" / "frozen_verification_pack.json"
INDEPENDENT_REVIEW_DIR = FROZEN_VERIFICATION / "08_independent_review"

# Historical package docs still scanned by Comment 8.x label/governance tests
CHANGE_REPORT_V331_MD = ARCHIVE_V331_REPORTS / "CHANGE_REPORT_v3.3.1.md"
CHANGE_REPORT_V331_DOCX = ARCHIVE_V331_REPORTS / "CHANGE_REPORT_v3.3.1.docx"
DELIVERY_CHECKLIST_MD = ARCHIVE_V331_REPORTS / "DELIVERY_CHECKLIST_SUPERVISOR.md"
DELIVERY_CHECKLIST_HTML = ARCHIVE_V331_REPORTS / "DELIVERY_CHECKLIST_SUPERVISOR.html"
PHYSICALIZATION_CHANGELOG = ARCHIVE_V331_REPORTS / "PHYSICALIZATION_CHANGELOG.md"

# Historical scripts
SYNC_ACTION_PLAN = ARCHIVE_V331_TOOLS / "sync_action_plan_from_register.py"
SYNC_CHANGE_REPORT_DOCX = ARCHIVE_V331_TOOLS / "sync_change_report_to_docx.py"
RUN_FINAL_VERIFICATION = ARCHIVE_VERIFICATION_TOOLS / "run_final_verification.py"
PREPARE_INDEPENDENT_REVIEW = ARCHIVE_REVIEWER_TOOLS / "prepare_independent_review_package.py"
VERIFY_REVIEWER_EVIDENCE = ARCHIVE_REVIEWER_TOOLS / "verify_reviewer_evidence.py"
APPLY_INDEPENDENT_SIGNOFF = ARCHIVE_REVIEWER_TOOLS / "apply_independent_review_signoff.py"
GENERATE_FROZEN_VERIFICATION = ARCHIVE_VERIFICATION_TOOLS / "generate_frozen_verification_evidence.py"
AUDIT_PHYSICS_PARAMETER_SHEET = ARCHIVE_VERIFICATION_TOOLS / "audit_physics_parameter_sheet.py"


def ensure_import_paths() -> None:
    """Put active tools and archived tool directories on ``sys.path`` for imports."""
    for path in (
        TOOLS_DIR,
        ARCHIVE_V331_TOOLS,
        ARCHIVE_REVIEWER_TOOLS,
        ARCHIVE_VERIFICATION_TOOLS,
        ROOT,
    ):
        text = str(path)
        if path.is_dir() and text not in sys.path:
            sys.path.insert(0, text)

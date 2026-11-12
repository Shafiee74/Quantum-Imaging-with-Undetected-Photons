"""QIUP verification/detector — U12 validation of the existing EMCCD chain.

Does **not** implement a second detector model. Analytical mean/variance
formulas are expectations of ``detector.py`` only.
"""
from __future__ import annotations

from verification.detector.analytical import analytical_expectations, pre_poisson_lambda
from verification.detector.cases import case_configs, run_all_cases, run_case
from verification.detector.definitions import (
    CASE_IDS,
    CHAIN_STAGES_DOCUMENTED,
    DEFINITIONS_MANIFEST,
)

__all__ = [
    "CASE_IDS",
    "CHAIN_STAGES_DOCUMENTED",
    "DEFINITIONS_MANIFEST",
    "analytical_expectations",
    "case_configs",
    "pre_poisson_lambda",
    "run_all_cases",
    "run_case",
]

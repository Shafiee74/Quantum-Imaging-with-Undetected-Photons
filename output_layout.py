"""Task-scoped output layout helpers for QIUP paper / V&V work.

Creates standardized evidence directories under ``evidence/taskXX_<name>/`` and
returns Path objects. Does not hard-code machine-absolute repository paths:
the repository root is discovered from this file's location (or passed in).

Experimental data lives outside the repository and must be supplied via
CLI/config/environment — never assumed as a baked-in absolute path inside
reusable call sites.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

# Subdirectories created inside each evidence/taskXX_<name>/ root.
TASK_SUBDIRS: tuple[str, ...] = (
    "data",
    "figures",
    "examples",
    "tables",
    "logs",
    "configs",
    "provenance",
    "report",
)

EXPERIMENTAL_DATA_ENV = "QIUP_EXPERIMENTAL_DATA_ROOT"

_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9._+-]+")


def repository_root(explicit: Path | str | None = None) -> Path:
    """Return the repository root (directory containing this module)."""
    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"repository root does not exist: {root}")
        return root
    return Path(__file__).resolve().parent


def safe_token(value: str, *, max_len: int = 80, fallback: str = "x") -> str:
    """Collapse arbitrary text into a filesystem-safe token."""
    text = str(value).strip().replace(" ", "_")
    text = _SAFE_TOKEN.sub("_", text).strip("._")
    if not text:
        text = fallback
    return text[:max_len]


def safe_filename(
    stem: str,
    *,
    suffix: str = "",
    config: str | None = None,
    run_id: str | None = None,
    max_stem_len: int = 120,
) -> str:
    """Build a safe filename: ``[config__]stem[__run_id].ext``.

    ``suffix`` may be ``'.png'``, ``'png'``, or empty.
    """
    parts: list[str] = []
    if config:
        parts.append(safe_token(config, max_len=32))
    parts.append(safe_token(stem, max_len=max_stem_len))
    if run_id:
        parts.append(safe_token(run_id, max_len=64))
    name = "__".join(parts)
    ext = str(suffix or "").strip()
    if ext and not ext.startswith("."):
        ext = "." + ext
    return name + ext


def utc_timestamp(*, timespec: str = "seconds") -> str:
    """UTC timestamp suitable for filenames (``YYYYMMDDTHHMMSSZ``)."""
    now = datetime.now(timezone.utc)
    if timespec == "seconds":
        return now.strftime("%Y%m%dT%H%M%SZ")
    # microseconds
    return now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond:06d}Z"


def make_run_id(
    *,
    config: str | None = None,
    include_timestamp: bool = True,
    extra: str | None = None,
    explicit: str | None = None,
) -> str:
    """Compose a run identifier.

    Prefer ``explicit`` when provided (still sanitized). Otherwise join
    optional config, UTC timestamp, and extra tokens.
    """
    if explicit is not None:
        return safe_token(explicit, max_len=96)
    parts: list[str] = []
    if config:
        parts.append(safe_token(config, max_len=24))
    if include_timestamp:
        parts.append(utc_timestamp())
    if extra:
        parts.append(safe_token(extra, max_len=40))
    if not parts:
        parts.append("run")
    return "__".join(parts)


def task_slug(task_id: str, task_name: str) -> str:
    """Return ``taskXX_<name>`` directory slug."""
    tid = safe_token(task_id, max_len=24)
    if not tid.lower().startswith("task"):
        tid = f"task_{tid}"
    name = safe_token(task_name, max_len=64)
    return f"{tid}_{name}"


def evidence_root(repo_root: Path | str | None = None) -> Path:
    return repository_root(repo_root) / "evidence"


def task_evidence_dir(
    task_id: str,
    task_name: str,
    *,
    repo_root: Path | str | None = None,
) -> Path:
    """Path to ``evidence/taskXX_<name>/`` (not necessarily created yet)."""
    return evidence_root(repo_root) / task_slug(task_id, task_name)


@dataclass(frozen=True)
class TaskOutputLayout:
    """Standardized paths for one task evidence tree."""

    task_id: str
    task_name: str
    repo_root: Path
    root: Path
    data: Path
    figures: Path
    examples: Path
    tables: Path
    logs: Path
    configs: Path
    provenance: Path
    report: Path
    run_id: str | None = None
    created_subdirs: tuple[str, ...] = field(default_factory=tuple)

    @property
    def slug(self) -> str:
        return self.root.name

    def category(self, name: str) -> Path:
        key = str(name).strip().lower()
        mapping = {
            "data": self.data,
            "figures": self.figures,
            "figure": self.figures,
            "examples": self.examples,
            "example": self.examples,
            "tables": self.tables,
            "table": self.tables,
            "logs": self.logs,
            "log": self.logs,
            "configs": self.configs,
            "config": self.configs,
            "provenance": self.provenance,
            "report": self.report,
            "reports": self.report,
            "root": self.root,
        }
        if key not in mapping:
            raise KeyError(
                f"unknown output category {name!r}; "
                f"expected one of {sorted(set(mapping))}"
            )
        return mapping[key]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "slug": self.slug,
            "run_id": self.run_id,
            "repo_root": str(self.repo_root),
            "root": str(self.root),
            "categories": {name: str(self.category(name)) for name in TASK_SUBDIRS},
            "created_subdirs": list(self.created_subdirs),
        }


def ensure_task_layout(
    task_id: str,
    task_name: str,
    *,
    repo_root: Path | str | None = None,
    run_id: str | None = None,
    create: bool = True,
    subdirs: Iterable[str] = TASK_SUBDIRS,
) -> TaskOutputLayout:
    """Create (optional) and return the standard task evidence layout."""
    root_path = repository_root(repo_root)
    task_root = task_evidence_dir(task_id, task_name, repo_root=root_path)
    wanted = tuple(dict.fromkeys(str(s).strip().lower() for s in subdirs if str(s).strip()))
    created: list[str] = []
    if create:
        task_root.mkdir(parents=True, exist_ok=True)
        for name in wanted:
            path = task_root / name
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                created.append(name)
            elif not path.is_dir():
                raise NotADirectoryError(f"expected directory: {path}")
    else:
        for name in wanted:
            path = task_root / name
            if path.exists() and not path.is_dir():
                raise NotADirectoryError(f"expected directory: {path}")

    def _cat(name: str) -> Path:
        return task_root / name

    return TaskOutputLayout(
        task_id=str(task_id),
        task_name=str(task_name),
        repo_root=root_path,
        root=task_root,
        data=_cat("data"),
        figures=_cat("figures"),
        examples=_cat("examples"),
        tables=_cat("tables"),
        logs=_cat("logs"),
        configs=_cat("configs"),
        provenance=_cat("provenance"),
        report=_cat("report"),
        run_id=run_id,
        created_subdirs=tuple(created),
    )


def unique_path(
    path: Path,
    *,
    overwrite: bool = False,
    max_attempts: int = 1000,
) -> Path:
    """Return ``path`` or a non-colliding sibling when overwrite is forbidden."""
    target = Path(path)
    if overwrite or not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    parent = target.parent
    for index in range(1, max_attempts + 1):
        candidate = parent / f"{stem}__v{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(
        f"could not allocate unique path for {target} after {max_attempts} attempts"
    )


def artifact_path(
    layout: TaskOutputLayout,
    category: str,
    stem: str,
    *,
    suffix: str = "",
    config: str | None = None,
    run_id: str | None = None,
    overwrite: bool = False,
    create_parents: bool = True,
) -> Path:
    """Return a standardized artifact path under a task category directory."""
    directory = layout.category(category)
    if create_parents:
        directory.mkdir(parents=True, exist_ok=True)
    rid = run_id if run_id is not None else layout.run_id
    name = safe_filename(stem, suffix=suffix, config=config, run_id=rid)
    return unique_path(directory / name, overwrite=overwrite)


def write_json(
    path: Path,
    payload: Mapping[str, Any] | list[Any],
    *,
    overwrite: bool = False,
    indent: int = 2,
) -> Path:
    """Write JSON atomically-ish (temp then replace) with overwrite control."""
    target = unique_path(Path(path), overwrite=overwrite)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=indent, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(target)
    return target


def resolve_experimental_data_root(
    cli_or_config_path: Path | str | None = None,
    *,
    env_var: str = EXPERIMENTAL_DATA_ENV,
    must_exist: bool = True,
) -> Path | None:
    """Resolve external experimental-data root from CLI/config or environment.

    Returns ``None`` when neither source is provided. Never embeds a
    machine-absolute default path in reusable code.
    """
    candidates: list[Path] = []
    if cli_or_config_path is not None and str(cli_or_config_path).strip():
        candidates.append(Path(str(cli_or_config_path)).expanduser())
    env_value = os.environ.get(env_var, "").strip()
    if env_value:
        candidates.append(Path(env_value).expanduser())
    if not candidates:
        return None
    chosen = candidates[0].resolve()
    if must_exist and not chosen.is_dir():
        raise FileNotFoundError(
            f"experimental data root not found: {chosen} "
            f"(from CLI/config or env {env_var})"
        )
    return chosen


# ---------------------------------------------------------------------------
# Architecture package roots (relative discovery; no absolute machine paths)
# ---------------------------------------------------------------------------

ARCHITECTURE_PACKAGES: dict[str, tuple[str, ...]] = {
    "verification": (
        "physics",
        "spatial",
        "resolution",
        "convergence",
        "monte_carlo",
        "detector",
        "phase_reconstruction",
        "config_c",
    ),
    "validation": (
        "config_c",
        "config_c/data_audit",
        "config_c/calibration",
        "config_c/c1",
        "config_c/c2_longitudinal",
        "config_c/c2_transverse",
    ),
    "benchmarks": (
        "targets",
        "four_config",
        "sensitivity",
    ),
    "examples": (
        "physics_controls",
        "four_configurations",
        "phase_reconstruction",
        "resolution",
        "convergence",
        "detector",
        "config_c",
    ),
}


DIRECTORY_ROLES: dict[str, str] = {
    "verification": (
        "Independent analytical/numerical verification against production "
        "outputs. Existing modules at this level (parity, config_d_checks, "
        "independent_reference, …) remain the current entry points; new "
        "work packages land in the thematic subpackages below."
    ),
    "verification/physics": (
        "U1–U8 style physics controls: energy conservation, SPDC gain, "
        "phase matching, transparent/blocked, amplitude/phase/visibility."
    ),
    "verification/spatial": (
        "U9 spatial mapping, magnification, orientation, near-/far-field "
        "coordinate transfer checks for A/B/C/D."
    ),
    "verification/resolution": (
        "U10 resolution benchmarks: point/slit/knife-edge, PSF/ESF/LSF "
        "extraction under a named convention."
    ),
    "verification/convergence": (
        "U14 grid/sampling convergence studies and related numerical "
        "stability checks."
    ),
    "verification/monte_carlo": (
        "U13 stochastic convergence: mean/variance/visibility/reconstruction "
        "error versus realization count."
    ),
    "verification/detector": (
        "U12 detector-stage invariants and analytical mean/variance checks "
        "complementing tests/test_detector_*.py."
    ),
    "verification/phase_reconstruction": (
        "Shared four-step / N-step phase retrieval verification using the "
        "same algorithm intended for simulation and experiment."
    ),
    "verification/config_c": (
        "Configuration-C-specific numerical verification (topology, "
        "far-field mapping, lens-perturbation laws) independent of lab "
        "held-out validation."
    ),
    "validation": (
        "Apparatus-specific experimental validation. Currently focused on "
        "Configuration C. Does not store or rewrite experimental source "
        "TIFF archives; scripts read an externally supplied data root."
    ),
    "validation/config_c": (
        "Root for Configuration C laboratory validation workflows."
    ),
    "validation/config_c/data_audit": (
        "Recording manifests, dimension/ROI/clipping audits, usable-frame "
        "partitioning. Read-only relative to experimental source data."
    ),
    "validation/config_c/calibration": (
        "Calibration-only fits (phase-position, detector background, "
        "nuisance parameters) using designated no-object / development data."
    ),
    "validation/config_c/c1": (
        "C1 axial idler-mirror phase-step evaluation (held-out predictions)."
    ),
    "validation/config_c/c2_longitudinal": (
        "C2 longitudinal idler-lens (circular fringe / a(δ)) evaluation."
    ),
    "validation/config_c/c2_transverse": (
        "C2 transverse idler-lens (parallel fringe / κ(Δx)) evaluation. "
        "Requires corrected transverse phase law before quantitative use."
    ),
    "benchmarks": (
        "Reusable synthetic targets and cross-configuration comparison "
        "harnesses for manuscript Figures 3–4 and sensitivity studies."
    ),
    "benchmarks/targets": (
        "Canonical synthetic objects: transparent, blocked, knife-edge, "
        "point/slit, grid/bars, amplitude/phase steps."
    ),
    "benchmarks/four_config": (
        "Matched A/B/C/D benchmark runners and comparison tables."
    ),
    "benchmarks/sensitivity": (
        "Parameter sweeps (L, wp, focal lengths, overlap, detector settings) "
        "for robustness / Figure 7-style evidence."
    ),
    "examples": (
        "Runnable demonstration scripts and notebooks. Legacy generators "
        "remain at this level; new demos prefer thematic subfolders."
    ),
    "examples/physics_controls": (
        "Short demos for energy/gain/phase-matching/transparent-blocked "
        "controls feeding verification/physics."
    ),
    "examples/four_configurations": (
        "Four-arrangement imaging demos (successor home for new A–D "
        "example scripts)."
    ),
    "examples/phase_reconstruction": (
        "Demonstrations of N-step phase retrieval on synthetic frames."
    ),
    "examples/resolution": (
        "Point/edge resolution demo scripts."
    ),
    "examples/convergence": (
        "Grid and Monte-Carlo convergence demos."
    ),
    "examples/detector": (
        "Detector chain / ADU statistics demos."
    ),
    "examples/config_c": (
        "Configuration C demo entry points (simulation-side; lab I/O lives "
        "under validation/config_c)."
    ),
    "evidence": (
        "Immutable task evidence packs. Each major task writes only under "
        "evidence/taskXX_<name>/ with data/figures/examples/tables/logs/"
        "configs/provenance/report. Never dump artifacts at repository root."
    ),
    "configs": (
        "Registered YAML profiles and inventory/audit JSON (existing). "
        "New laboratory or paper profiles may be added later without "
        "moving stock A–D YAMLs."
    ),
    "tests": (
        "Regression and unit suite (existing). Architecture import/path "
        "tests for this layout live here."
    ),
    "reference_outputs": (
        "Canonical Comment-7.7 reference run pack (existing). Untouched by "
        "Task 02 structure work."
    ),
    "data": (
        "In-repo input datasets such as MNIST (existing). Experimental "
        "TIFF archives are NOT stored here."
    ),
}


def architecture_paths(repo_root: Path | str | None = None) -> dict[str, Path]:
    """Map relative architecture keys to absolute Paths under the repo root."""
    root = repository_root(repo_root)
    out: dict[str, Path] = {}
    for package, children in ARCHITECTURE_PACKAGES.items():
        out[package] = root / package
        for child in children:
            key = f"{package}/{child}"
            out[key] = root / package / Path(child)
    # Existing support trees reused by Task 02
    for name in ("evidence", "configs", "tests", "reference_outputs", "data"):
        out[name] = root / name
    return out


def ensure_architecture_directories(
    repo_root: Path | str | None = None,
    *,
    write_role_files: bool = True,
) -> dict[str, Path]:
    """Create missing architecture directories; never delete or move existing files."""
    root = repository_root(repo_root)
    paths = architecture_paths(root)
    package_roots = set(ARCHITECTURE_PACKAGES)

    for key, path in sorted(paths.items(), key=lambda kv: (kv[0].count("/"), kv[0])):
        path.mkdir(parents=True, exist_ok=True)
        if write_role_files and key in DIRECTORY_ROLES:
            role_path = path / "ROLE.md"
            if not role_path.exists():
                role_path.write_text(
                    f"# `{key}`\n\n"
                    f"{DIRECTORY_ROLES[key]}\n\n"
                    "Created/retained by Task 02 architecture setup. "
                    "Do not write paper artifacts here; use "
                    "`evidence/taskXX_<name>/` via `output_layout`.\n",
                    encoding="utf-8",
                )

        top = key.split("/", 1)[0]
        if top not in package_roots:
            continue
        # Never overwrite the existing verification package docstring/module.
        if key == "verification":
            continue
        init = path / "__init__.py"
        if not init.exists():
            init.write_text(
                f'"""QIUP {key} package (Task 02 architecture)."""\n',
                encoding="utf-8",
            )
    return paths


def build_directory_manifest(
    repo_root: Path | str | None = None,
    *,
    experimental_data_root: Path | str | None = None,
) -> dict[str, Any]:
    """Build a machine-readable directory manifest for Task 02 evidence."""
    root = repository_root(repo_root)
    paths = architecture_paths(root)
    entries: list[dict[str, Any]] = []
    for key, path in sorted(paths.items()):
        rel = path.relative_to(root).as_posix()
        role = DIRECTORY_ROLES.get(key, DIRECTORY_ROLES.get(rel, ""))
        entries.append(
            {
                "key": key,
                "relative_path": rel,
                "exists": path.is_dir(),
                "role": role,
                "status": "present" if path.is_dir() else "missing",
                "owned_by_task02_packages": key.split("/", 1)[0]
                in ARCHITECTURE_PACKAGES
                or key in ARCHITECTURE_PACKAGES,
            }
        )
    exp: dict[str, Any]
    try:
        resolved = resolve_experimental_data_root(
            experimental_data_root, must_exist=False
        )
        if resolved is None and experimental_data_root:
            resolved = Path(str(experimental_data_root)).expanduser()
        if resolved is None:
            exp = {
                "configured": False,
                "path": None,
                "exists": False,
                "note": (
                    "Pass CLI/config path or set "
                    f"{EXPERIMENTAL_DATA_ENV}; experimental source data "
                    "must not be moved or modified."
                ),
            }
        else:
            exp = {
                "configured": True,
                "path": str(resolved),
                "exists": resolved.is_dir(),
                "note": (
                    "External experimental-data root is referenced only; "
                    "Task 02 does not copy, move, or modify its contents."
                ),
            }
    except OSError as exc:
        exp = {
            "configured": True,
            "path": str(experimental_data_root),
            "exists": False,
            "error": str(exc),
        }
    return {
        "schema": "qiup_task02_directory_manifest_v1",
        "package_version_file": (root / "VERSION").read_text(encoding="utf-8").strip()
        if (root / "VERSION").is_file()
        else None,
        "repository_root": str(root),
        "task_output_subdirs": list(TASK_SUBDIRS),
        "output_layout_module": "output_layout.py",
        "experimental_data": exp,
        "directories": entries,
    }


__all__ = [
    "ARCHITECTURE_PACKAGES",
    "DIRECTORY_ROLES",
    "EXPERIMENTAL_DATA_ENV",
    "TASK_SUBDIRS",
    "TaskOutputLayout",
    "architecture_paths",
    "artifact_path",
    "build_directory_manifest",
    "ensure_architecture_directories",
    "ensure_task_layout",
    "evidence_root",
    "make_run_id",
    "repository_root",
    "resolve_experimental_data_root",
    "safe_filename",
    "safe_token",
    "task_evidence_dir",
    "task_slug",
    "unique_path",
    "utc_timestamp",
    "write_json",
]

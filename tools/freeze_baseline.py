#!/usr/bin/env python3
"""Freeze a reproducibility baseline for the current QIUP production tree.

Writes only under ``evidence/task03_baseline_freeze/``. Does **not** modify
simulator physics and does **not** overwrite repository-root ``SHA256SUMS.txt``.

Reuses ``provenance`` hashing / environment helpers and ``verify_sha256_manifest``
for raw SHA-256 and optional subset checks against the existing package manifest.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from output_layout import (  # noqa: E402
    ensure_task_layout,
    write_json,
)
from provenance import (  # noqa: E402
    BUILD_SCRIPT_NAME,
    CALIBRATION_STATUS,
    HASH_ALGORITHM,
    PACKAGE_VERSION,
    PHYSICS_MODEL_VERSION,
    PHYSICS_POLICY_VERSION,
    RELEASE_CLASSIFICATION,
    RELEASE_STATUS,
    SINGLE_FILE_DISTRIBUTION_NAME,
    SOURCE_MODULES,
    compute_package_hash_set,
    default_rng_scheme,
    get_environment,
    hash_source_file,
)
from verify_sha256_manifest import (  # noqa: E402
    read_manifest as read_sha256sums,
    sha256_file,
)

TASK_ID = "task03"
TASK_NAME = "baseline_freeze"
SCHEMA = "qiup_task03_baseline_freeze_v1"

PRODUCTION_PYTHON_FILES: tuple[str, ...] = tuple(SOURCE_MODULES) + (
    SINGLE_FILE_DISTRIBUTION_NAME,
)

SUPPORT_TEXT_FILES: tuple[str, ...] = (
    BUILD_SCRIPT_NAME,
    "VERSION",
    "README.md",
    "requirements.txt",
    "requirements.frozen.txt",
    "repo_paths.py",
    "output_layout.py",
    "verify_sha256_manifest.py",
    "build_parameter_spec.py",
    "parameter_inventory_audit.py",
    "generate_v3_4_1_evidence.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def collect_git_state(repo_root: Path) -> dict[str, Any]:
    """Return git commit / working-tree state, or an explicit unavailable record."""
    base: dict[str, Any] = {
        "available": False,
        "commit": None,
        "commit_short": None,
        "branch": None,
        "describe": None,
        "working_tree_clean": None,
        "status_porcelain": None,
        "status": "unavailable",
        "detail": "not a git repository (or git unavailable)",
    }

    def _run(args: list[str]) -> subprocess.CompletedProcess[str] | None:
        try:
            return subprocess.run(
                args,
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
            base["detail"] = f"git invocation failed: {exc}"
            return None

    probe = _run(["git", "rev-parse", "--is-inside-work-tree"])
    if probe is None:
        return base
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        err = (probe.stderr or probe.stdout or "").strip()
        if err:
            base["detail"] = err
        return base

    base["available"] = True
    base["status"] = "available"
    base["detail"] = None

    head = _run(["git", "rev-parse", "HEAD"])
    if head and head.returncode == 0:
        base["commit"] = head.stdout.strip()
        base["commit_short"] = base["commit"][:12]

    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if branch and branch.returncode == 0:
        base["branch"] = branch.stdout.strip()

    describe = _run(["git", "describe", "--always", "--dirty", "--tags"])
    if describe and describe.returncode == 0:
        base["describe"] = describe.stdout.strip()

    porcelain = _run(["git", "status", "--porcelain=v1"])
    if porcelain and porcelain.returncode == 0:
        text = porcelain.stdout
        base["status_porcelain"] = text
        base["working_tree_clean"] = text.strip() == ""
        base["status"] = "clean" if base["working_tree_clean"] else "dirty"
    return base


def list_config_files(repo_root: Path) -> list[Path]:
    configs = repo_root / "configs"
    files: list[Path] = []
    if not configs.is_dir():
        return files
    for path in sorted(configs.rglob("*")):
        if not path.is_file():
            continue
        if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        files.append(path)
    return files


def file_record(repo_root: Path, relative: str, *, text_normalized: bool) -> dict[str, Any]:
    path = repo_root / relative
    record: dict[str, Any] = {
        "relative_path": relative.replace("\\", "/"),
        "exists": path.is_file(),
        "bytes": None,
        "sha256_raw": None,
        "sha256_text_lf": None,
        "hash_note": (
            "sha256_raw = binary file digest (SHA256SUMS-compatible); "
            "sha256_text_lf = provenance LF-normalized text digest"
        ),
    }
    if not path.is_file():
        return record
    record["bytes"] = path.stat().st_size
    record["sha256_raw"] = sha256_file(path)
    if text_normalized:
        try:
            record["sha256_text_lf"] = hash_source_file(path)
        except UnicodeDecodeError:
            record["sha256_text_lf"] = None
            record["hash_note"] += "; text LF hash skipped (not UTF-8 text)"
    return record


def installed_dependency_rows(requirement_names: Iterable[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in requirement_names:
        name = raw.strip()
        if not name or name.startswith("#"):
            continue
        # strip extras / version pins for lookup
        for sep in ("[", "=", ">", "<", "!", "~", " "):
            if sep in name:
                name = name.split(sep, 1)[0]
        name = name.strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        try:
            version = importlib.metadata.version(name)
            status = "installed"
        except importlib.metadata.PackageNotFoundError:
            version = ""
            status = "not_installed"
        rows.append(
            {
                "package": name,
                "installed_version": version,
                "status": status,
            }
        )
    # Always include core scientific stack even if not named in requirements.
    for extra in ("numpy", "scipy", "matplotlib", "PyYAML", "Pillow", "openpyxl"):
        if extra.lower() not in seen:
            try:
                version = importlib.metadata.version(extra)
                status = "installed"
            except importlib.metadata.PackageNotFoundError:
                version = ""
                status = "not_installed"
            rows.append(
                {"package": extra, "installed_version": version, "status": status}
            )
            seen.add(extra.lower())
    return sorted(rows, key=lambda r: r["package"].lower())


def parse_requirement_names(text: str) -> list[str]:
    names: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        names.append(stripped)
    return names


def sha256sums_subset_check(
    repo_root: Path,
    relative_paths: Iterable[str],
) -> dict[str, Any]:
    """Compare selected files to existing SHA256SUMS.txt without rewriting it."""
    manifest_path = repo_root / "SHA256SUMS.txt"
    result: dict[str, Any] = {
        "manifest_path": "SHA256SUMS.txt",
        "manifest_exists": manifest_path.is_file(),
        "overwritten": False,
        "checked": [],
        "matched": [],
        "mismatched": [],
        "missing_from_manifest": [],
        "missing_on_disk": [],
    }
    if not manifest_path.is_file():
        result["status"] = "manifest_missing"
        return result
    entries = read_sha256sums(repo_root)
    for rel in relative_paths:
        key = rel.replace("\\", "/")
        result["checked"].append(key)
        path = repo_root / key
        if not path.is_file():
            result["missing_on_disk"].append(key)
            continue
        digest = sha256_file(path)
        if key not in entries:
            result["missing_from_manifest"].append(key)
            continue
        if entries[key] == digest:
            result["matched"].append(key)
        else:
            result["mismatched"].append(
                {"path": key, "manifest": entries[key], "current": digest}
            )
    if result["mismatched"] or result["missing_on_disk"]:
        result["status"] = "failed"
    elif result["missing_from_manifest"]:
        result["status"] = "partial_unlisted"
    else:
        result["status"] = "ok"
    return result


def verify_baseline_file_hashes(
    repo_root: Path,
    rows: Iterable[Mapping[str, str]],
    *,
    algorithm_field: str = "sha256_raw",
) -> dict[str, Any]:
    """Replay/verify a previously frozen file-hash table against the live tree."""
    matched: list[str] = []
    mismatched: list[dict[str, str]] = []
    missing: list[str] = []
    for row in rows:
        rel = str(row["relative_path"]).replace("\\", "/")
        expected = str(row[algorithm_field])
        path = repo_root / rel
        if not path.is_file():
            missing.append(rel)
            continue
        if algorithm_field == "sha256_text_lf":
            actual = hash_source_file(path)
        else:
            actual = sha256_file(path)
        if actual == expected:
            matched.append(rel)
        else:
            mismatched.append(
                {"path": rel, "expected": expected, "actual": actual}
            )
    ok = not mismatched and not missing
    return {
        "ok": ok,
        "algorithm_field": algorithm_field,
        "matched_count": len(matched),
        "mismatched": mismatched,
        "missing": missing,
    }


def load_file_hashes_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return path


def build_baseline(
    repo_root: Path,
    *,
    argv: list[str] | None = None,
) -> dict[str, Any]:
    argv = list(argv) if argv is not None else list(sys.argv)
    timestamp = _utc_now()
    version_file = (repo_root / "VERSION").read_text(encoding="utf-8").strip()
    git_state = collect_git_state(repo_root)
    environment = get_environment()
    rng_scheme = default_rng_scheme()

    req_txt = (repo_root / "requirements.txt").read_text(encoding="utf-8")
    req_frozen = (repo_root / "requirements.frozen.txt").read_text(encoding="utf-8")
    dep_rows = installed_dependency_rows(
        parse_requirement_names(req_txt) + parse_requirement_names(req_frozen)
    )

    production_records = [
        file_record(repo_root, rel, text_normalized=True)
        for rel in PRODUCTION_PYTHON_FILES
    ]
    support_records = [
        file_record(repo_root, rel, text_normalized=True)
        for rel in SUPPORT_TEXT_FILES
        if (repo_root / rel).is_file()
    ]
    config_paths = list_config_files(repo_root)
    config_records = [
        file_record(repo_root, p.relative_to(repo_root).as_posix(), text_normalized=True)
        for p in config_paths
    ]

    package_hashes = compute_package_hash_set(repo_root)
    package_hashes["git_commit"] = git_state.get("commit")
    package_hashes["git_commit_status"] = git_state.get("status")
    package_hashes["git"] = git_state

    sha_check = sha256sums_subset_check(
        repo_root,
        list(PRODUCTION_PYTHON_FILES)
        + [BUILD_SCRIPT_NAME, "VERSION", "requirements.txt", "requirements.frozen.txt"],
    )

    hash_rows: list[dict[str, Any]] = []
    for group, records in (
        ("production_python", production_records),
        ("support", support_records),
        ("configs", config_records),
    ):
        for rec in records:
            hash_rows.append(
                {
                    "group": group,
                    "relative_path": rec["relative_path"],
                    "exists": rec["exists"],
                    "bytes": rec["bytes"],
                    "sha256_raw": rec["sha256_raw"],
                    "sha256_text_lf": rec["sha256_text_lf"],
                }
            )

    command = {
        "argv": argv,
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "module_or_script": str(Path(__file__).resolve()),
        "timestamp_utc": timestamp,
    }

    repository_state = {
        "root": str(repo_root.resolve()),
        "version_file": version_file,
        "package_version": PACKAGE_VERSION,
        "physics_model_version": PHYSICS_MODEL_VERSION,
        "physics_policy_version": PHYSICS_POLICY_VERSION,
        "release_status": RELEASE_STATUS,
        "release_classification": RELEASE_CLASSIFICATION,
        "calibration_status": CALIBRATION_STATUS,
        "hash_algorithm": HASH_ALGORITHM,
        "production_physics_modified_by_this_task": False,
        "sha256sums_txt_overwritten": False,
        "git": git_state,
    }

    baseline = {
        "schema": SCHEMA,
        "timestamp_utc": timestamp,
        "version_file": version_file,
        "package_version": PACKAGE_VERSION,
        "physics_model_version": PHYSICS_MODEL_VERSION,
        "repository_state": repository_state,
        "git": git_state,
        "python": {
            "version": environment.get("python"),
            "full": environment.get("python_full"),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "os_platform": {
            "platform": environment.get("platform"),
            "system": environment.get("system"),
            "release": environment.get("release"),
            "machine": environment.get("machine"),
            "processor": environment.get("processor"),
        },
        "environment": environment,
        "dependency_versions": dep_rows,
        "requirements": {
            "requirements.txt": req_txt,
            "requirements.frozen.txt": req_frozen,
        },
        "random_seed_policy": rng_scheme,
        "production_files": {r["relative_path"]: r for r in production_records},
        "support_files": {r["relative_path"]: r for r in support_records},
        "configuration_inventory": {
            "count": len(config_records),
            "files": {r["relative_path"]: r for r in config_records},
        },
        "file_hashes": hash_rows,
        "package_hash_set": package_hashes,
        "sha256sums_txt": sha_check,
        "command": command,
        "notes": [
            "Production physics modules were not modified by Task 03.",
            "Repository-root SHA256SUMS.txt was not overwritten.",
            "Git unavailable is recorded explicitly when no repository is present.",
            "sha256_raw is SHA256SUMS-compatible; sha256_text_lf matches provenance.hash_source_file.",
        ],
    }
    return baseline


def write_baseline_evidence(
    baseline: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Path]:
    layout = ensure_task_layout(TASK_ID, TASK_NAME, repo_root=repo_root)
    paths: dict[str, Path] = {"layout_root": layout.root}

    paths["baseline_manifest"] = write_json(
        layout.data / "baseline_manifest.json",
        baseline,
        overwrite=True,
    )
    paths["file_hashes"] = write_csv(
        layout.data / "file_hashes.csv",
        [
            "group",
            "relative_path",
            "exists",
            "bytes",
            "sha256_raw",
            "sha256_text_lf",
        ],
        baseline["file_hashes"],
    )
    paths["dependency_versions"] = write_csv(
        layout.tables / "dependency_versions.csv",
        ["package", "installed_version", "status"],
        baseline["dependency_versions"],
    )
    paths["configuration_inventory"] = write_json(
        layout.configs / "configuration_inventory.json",
        baseline["configuration_inventory"],
        overwrite=True,
    )
    paths["environment"] = write_json(
        layout.provenance / "environment.json",
        {
            "python": baseline["python"],
            "os_platform": baseline["os_platform"],
            "environment": baseline["environment"],
            "random_seed_policy": baseline["random_seed_policy"],
            "dependency_versions": baseline["dependency_versions"],
        },
        overwrite=True,
    )
    paths["command"] = write_json(
        layout.provenance / "command.json",
        baseline["command"],
        overwrite=True,
    )

    # requirements snapshots (text)
    (layout.configs / "requirements.txt").write_text(
        baseline["requirements"]["requirements.txt"], encoding="utf-8"
    )
    (layout.configs / "requirements.frozen.txt").write_text(
        baseline["requirements"]["requirements.frozen.txt"], encoding="utf-8"
    )
    paths["requirements_txt"] = layout.configs / "requirements.txt"
    paths["requirements_frozen"] = layout.configs / "requirements.frozen.txt"

    report = _render_report(baseline, layout.root)
    report_path = layout.report / "baseline_freeze_report.md"
    report_path.write_text(report, encoding="utf-8")
    paths["report"] = report_path

    log_path = layout.logs / "baseline_freeze.log"
    log_lines = [
        f"timestamp_utc={baseline['timestamp_utc']}",
        f"schema={baseline['schema']}",
        f"version_file={baseline['version_file']}",
        f"package_version={baseline['package_version']}",
        f"git_status={baseline['git']['status']}",
        f"git_available={baseline['git']['available']}",
        f"production_files={len(baseline['production_files'])}",
        f"config_files={baseline['configuration_inventory']['count']}",
        f"sha256sums_status={baseline['sha256sums_txt']['status']}",
        f"sha256sums_overwritten={baseline['sha256sums_txt']['overwritten']}",
        f"evidence_root={layout.root}",
        "status=OK",
    ]
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    paths["log"] = log_path
    return paths


def _render_report(baseline: Mapping[str, Any], evidence_root: Path) -> str:
    git = baseline["git"]
    sha = baseline["sha256sums_txt"]
    lines = [
        "# Task 03 — Baseline Freeze Report",
        "",
        f"**Schema:** `{baseline['schema']}`  ",
        f"**Timestamp (UTC):** `{baseline['timestamp_utc']}`  ",
        f"**VERSION file:** `{baseline['version_file']}`  ",
        f"**Package version:** `{baseline['package_version']}`  ",
        f"**Physics model version:** `{baseline['physics_model_version']}`  ",
        f"**Evidence root:** `{evidence_root.as_posix()}`",
        "",
        "## Constraints",
        "",
        "- Simulator physics modules were **not** modified.",
        "- Repository-root `SHA256SUMS.txt` was **not** overwritten.",
        "- All generated artifacts were written under `evidence/task03_baseline_freeze/`.",
        "",
        "## Git",
        "",
    ]
    if not git.get("available"):
        lines.extend(
            [
                "**Git is unavailable** for this working tree.",
                "",
                f"- status: `{git.get('status')}`",
                f"- detail: `{git.get('detail')}`",
                "- commit: `null`",
                "- working-tree state: not applicable (no git repository)",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"- available: `{git.get('available')}`",
                f"- status: `{git.get('status')}`",
                f"- commit: `{git.get('commit')}`",
                f"- branch: `{git.get('branch')}`",
                f"- describe: `{git.get('describe')}`",
                f"- working_tree_clean: `{git.get('working_tree_clean')}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Environment",
            "",
            f"- Python: `{baseline['python']['version']}`",
            f"- Platform: `{baseline['os_platform']['platform']}`",
            f"- NumPy: `{baseline['environment'].get('numpy')}`",
            f"- SciPy: `{baseline['environment'].get('scipy')}`",
            f"- Pillow: `{baseline['environment'].get('pillow')}`",
            "",
            "## Random-seed policy",
            "",
            "```json",
            json.dumps(baseline["random_seed_policy"], indent=2),
            "```",
            "",
            "## Production Python SHA-256 (raw)",
            "",
            "| File | bytes | sha256_raw |",
            "|---|---:|---|",
        ]
    )
    for rel, rec in baseline["production_files"].items():
        lines.append(
            f"| `{rel}` | {rec.get('bytes')} | `{rec.get('sha256_raw')}` |"
        )

    lines.extend(
        [
            "",
            "## Provenance package hash set",
            "",
            f"- package_hash: `{baseline['package_hash_set'].get('package_hash')}`",
            f"- build_script_sha256: `{baseline['package_hash_set'].get('build_script_sha256')}`",
            f"- single_file_sha256: `{baseline['package_hash_set'].get('single_file_sha256')}`",
            "",
            "## Existing SHA256SUMS.txt subset check",
            "",
            f"- overwritten: `{sha.get('overwritten')}`",
            f"- status: `{sha.get('status')}`",
            f"- matched: {len(sha.get('matched') or [])}",
            f"- mismatched: {len(sha.get('mismatched') or [])}",
            f"- missing_from_manifest: {len(sha.get('missing_from_manifest') or [])}",
            "",
            "## Command-line invocation",
            "",
            "```json",
            json.dumps(baseline["command"], indent=2),
            "```",
            "",
            "## Artifacts",
            "",
            "- `data/baseline_manifest.json`",
            "- `data/file_hashes.csv`",
            "- `tables/dependency_versions.csv`",
            "- `configs/configuration_inventory.json`",
            "- `configs/requirements.txt`",
            "- `configs/requirements.frozen.txt`",
            "- `provenance/environment.json`",
            "- `provenance/command.json`",
            "- `logs/baseline_freeze.log`",
            "- `report/baseline_freeze_report.md`",
            "",
            "*End of Task 03 baseline freeze report.*",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: parent of tools/)",
    )
    args = parser.parse_args(argv)
    repo_root = (args.repo_root or ROOT).resolve()
    baseline = build_baseline(repo_root, argv=["python", str(Path(__file__).resolve()), *(argv or [])])
    paths = write_baseline_evidence(baseline, repo_root=repo_root)
    print(f"wrote baseline evidence under {paths['layout_root']}")
    print(f"git status: {baseline['git']['status']}")
    print(f"SHA256SUMS subset: {baseline['sha256sums_txt']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

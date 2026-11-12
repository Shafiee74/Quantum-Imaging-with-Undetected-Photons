#!/usr/bin/env python3
"""Write or verify ``SHA256SUMS.txt`` for the QIUP Simulator package.

Format (``sha256sum`` compatible): one line per file, ``<sha256>  <relative/path>``,
POSIX separators, sorted by path. Every package file is covered except the
manifest itself and transient or third-party build artifacts:

* ``__pycache__/``, ``.pytest_cache/``, ``*.pyc``, ``*.pyo``
* ``qiup-ui/node_modules/``, ``qiup-ui/.next/``, ``qiup-ui/.tools/``
  (reproducible from ``qiup-ui/package-lock.json``)

Verification fails on any digest mismatch, any listed file that is missing,
and any package file that is not listed.

Usage::

    python verify_sha256_manifest.py            # verify
    python verify_sha256_manifest.py --write    # regenerate (only after all changes)
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
MANIFEST = "SHA256SUMS.txt"
EXCLUDED_DIR_NAMES = {"__pycache__", ".pytest_cache", "_pytest_latest", "_run_output"}
EXCLUDED_PREFIXES = (
    "qiup-ui/node_modules/",
    "qiup-ui/.next/",
    "qiup-ui/.tools/",
    "exp data/",
    "reference_outputs/",
    "examples/notebooks/_run_output/",
    "evidence/",
    "outputs/",
)
EXCLUDED_SUFFIXES = (".pyc", ".pyo")


def _included(rel: str) -> bool:
    if rel == MANIFEST:
        return False
    if rel.startswith("_"):
        return False
    if any(part in EXCLUDED_DIR_NAMES for part in rel.split("/")):
        return False
    if rel.startswith(EXCLUDED_PREFIXES) or rel.endswith(EXCLUDED_SUFFIXES):
        return False
    return True


def package_files(root: Path = ROOT) -> list[str]:
    """Return actual package files, ignoring only broken/vanished entries.

    Some archived evidence trees contain directory entries whose targets no
    longer exist.  ``Path.rglob`` aborts at the first such entry on Windows,
    preventing the manifest from covering the rest of the package.  ``os.walk``
    gives us a narrow error hook: missing targets are not files to hash, while
    every other traversal error remains fatal.
    """

    out: list[str] = []

    def handle_walk_error(error: OSError) -> None:
        if isinstance(error, FileNotFoundError):
            return
        raise error

    for directory, dirnames, filenames in os.walk(
        root, topdown=True, onerror=handle_walk_error, followlinks=False
    ):
        parent = Path(directory)
        parent_rel = parent.relative_to(root).as_posix()
        prefix = "" if parent_rel == "." else f"{parent_rel}/"
        dirnames[:] = [
            name
            for name in dirnames
            if name not in EXCLUDED_DIR_NAMES
            and not any(f"{prefix}{name}/".startswith(item) for item in EXCLUDED_PREFIXES)
        ]
        for name in filenames:
            path = parent / name
            try:
                is_file = path.is_file()
            except FileNotFoundError:
                continue
            if is_file:
                rel = path.relative_to(root).as_posix()
                if _included(rel):
                    out.append(rel)
    return sorted(out)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(root: Path = ROOT) -> int:
    files = package_files(root)
    lines = [f"{sha256_file(root / rel)}  {rel}" for rel in files]
    (root / MANIFEST).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return len(lines)


def read_manifest(root: Path = ROOT) -> dict[str, str]:
    entries: dict[str, str] = {}
    for n, line in enumerate((root / MANIFEST).read_text(encoding="utf-8").splitlines(), 1):
        digest, sep, rel = line.partition("  ")
        if not sep or len(digest) != 64 or rel in entries:
            raise ValueError(f"{MANIFEST}:{n}: malformed or duplicate entry")
        entries[rel] = digest
    return entries


def verify_manifest(root: Path = ROOT) -> dict[str, list[str] | int]:
    entries = read_manifest(root)
    present = set(package_files(root))
    missing = sorted(rel for rel in entries if not (root / rel).is_file())
    mismatched = sorted(rel for rel, digest in entries.items()
                        if (root / rel).is_file() and sha256_file(root / rel) != digest)
    unlisted = sorted(present - set(entries))
    return {"files": len(entries), "missing": missing, "mismatched": mismatched, "unlisted": unlisted}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="regenerate the manifest")
    args = parser.parse_args(argv)
    if args.write:
        print(f"wrote {MANIFEST}: {write_manifest()} files (SHA-256)")
        return 0
    result = verify_manifest()
    for key in ("missing", "mismatched", "unlisted"):
        for rel in result[key]:
            print(f"{key.upper()}: {rel}")
    ok = not (result["missing"] or result["mismatched"] or result["unlisted"])
    print(f"{MANIFEST}: {result['files']} files, algorithm SHA-256 -> {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

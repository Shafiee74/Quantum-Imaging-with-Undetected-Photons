"""Physical-model change audit: received Version 3.4 sources vs active 3.4.1.

1. Static: SHA-256 of each received module against the reviewed 3.4 manifest,
   and an AST comparison (docstrings removed; comments are not in the AST) of
   received vs active modules, listing every differing top-level statement.
2. Behavioral: fixed-seed forward runs of A/B/C/D (the parity scenarios) with
   both source sets, each in its own interpreter. Arrays must be bit-identical;
   run metadata must be identical apart from declared release/provenance keys.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RECEIVED = ROOT / "archive" / "v3.4_superseded" / "received_v3_4_sources"
MANIFEST_V34 = ROOT / "archive" / "v3.4_superseded" / "PACKAGE_3.4_CORE_SHA256.json"
MODULES = ("setup_params.py", "latents.py", "optics.py", "detector.py", "provenance.py", "simulate.py")
# Metadata keys that carry release identity, source/build identity or the
# absolute location of the profile YAML, and so differ between the two source sets.
DECLARED_META_DIFFERENCES = frozenset({
    "timestamp", "release_version", "package_version", "release_classification",
    "calibration_status", "calibration_statement", "source_hashes", "build_hash",
    "package_hashes", "yaml_path",
})
PROFILES = ("A", "B", "C", "D")


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def _statement_label(node: ast.stmt) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return f"{type(node).__name__} {node.name}"
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return "assign " + ", ".join(ast.unparse(t) for t in targets)
    return type(node).__name__


def static_audit() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_V34.read_text(encoding="utf-8"))["files"]
    out: dict[str, Any] = {}
    for name in MODULES:
        received = (RECEIVED / name).read_bytes()
        active = (ROOT / name).read_bytes()
        rec = manifest.get(name)
        rec = rec.get("sha256") if isinstance(rec, dict) else rec
        old_body = _strip_docstrings(ast.parse(received.decode("utf-8"))).body
        new_body = _strip_docstrings(ast.parse(active.decode("utf-8"))).body
        old_map = {}
        for node in old_body:
            old_map.setdefault(_statement_label(node), []).append(ast.dump(node))
        new_map = {}
        for node in new_body:
            new_map.setdefault(_statement_label(node), []).append(ast.dump(node))
        changed = sorted(k for k in set(old_map) | set(new_map) if old_map.get(k) != new_map.get(k))
        out[name] = {
            "received_sha256": hashlib.sha256(received).hexdigest(),
            "active_sha256": hashlib.sha256(active).hexdigest(),
            "recorded_v3_4_sha256": rec,
            "received_matches_v3_4_manifest": None if rec is None else rec == hashlib.sha256(received).hexdigest(),
            "byte_identical": received == active,
            "ast_identical_ignoring_docstrings": ast.dump(ast.Module(old_body, [])) == ast.dump(ast.Module(new_body, [])),
            "changed_top_level_statements": changed,
        }
    return out


def _run_sources(modules_dir: Path, out_dir: Path) -> None:
    code = (
        "import sys, json, pickle\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from verification import parity\n"
        f"sys.path.insert(0, {str(modules_dir)!r})\n"
        "import numpy as np, latents, setup_params, simulate\n"
        f"assert setup_params.__file__.startswith({str(modules_dir)!r}), setup_params.__file__\n"
        "res = {}\n"
        "for p in parity.PROFILES:\n"
        "    for s, spec in parity.SCENARIOS.items():\n"
        "        setup = setup_params.resolve_setup_selection(p)\n"
        "        if spec['setup']: setup = setup.overridden(**spec['setup'])\n"
        "        st = latents.LatentState(setup=setup, **{**parity.BASE_STATE, **spec['state']})\n"
        "        r = simulate.simulate_forward(parity.object_reference(), st, seed=parity.SEED, detector_noise=True)\n"
        "        res[(p, s)] = ({k: getattr(r, k, None) for k in parity.ARRAY_OUTPUTS}, r.meta)\n"
        f"pickle.dump(res, open({str(out_dir / 'runs.pkl')!r}, 'wb'))\n"
    )
    proc = subprocess.run([sys.executable, "-B", "-c", code], cwd=str(modules_dir),
                          capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-4000:])


def _strip_declared(meta: Any) -> Any:
    if isinstance(meta, dict):
        return {k: _strip_declared(v) for k, v in meta.items() if k not in DECLARED_META_DIFFERENCES}
    if isinstance(meta, (list, tuple)):
        return [_strip_declared(v) for v in meta]
    return meta


def behavioral_audit() -> list[dict[str, Any]]:
    import pickle

    from verification import parity

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        received_dir = tmp_path / "received"
        (received_dir / "configs").mkdir(parents=True)
        for name in MODULES:
            shutil.copy2(RECEIVED / name, received_dir / name)
        for path in (ROOT / "configs").iterdir():
            if path.is_file() and path.suffix in (".yaml", ".yml", ".json"):
                shutil.copy2(path, received_dir / "configs" / path.name)
        for name in ("build_single_file.py", "qiup_digital_twin_full.py"):
            shutil.copy2(ROOT / name, received_dir / name)
        (tmp_path / "old").mkdir()
        (tmp_path / "new").mkdir()
        _run_sources(received_dir, tmp_path / "old")
        _run_sources(ROOT, tmp_path / "new")
        old = pickle.loads((tmp_path / "old" / "runs.pkl").read_bytes())
        new = pickle.loads((tmp_path / "new" / "runs.pkl").read_bytes())

    records = []
    for key in sorted(new):
        (arrays_new, meta_new), (arrays_old, meta_old) = new[key], old[key]
        worst, values, bad = 0.0, 0, []
        for name in parity.ARRAY_OUTPUTS:
            e, c, m = parity.compare_arrays(arrays_old[name], arrays_new[name], name)
            worst, values, bad = max(worst, e), values + c, bad + m
        me, mc, mm = parity.compare_tree(_strip_declared(meta_old), _strip_declared(meta_new), "meta")
        records.append({
            "configuration": key[0], "scenario": key[1], "seed": parity.SEED,
            "array_values_compared": values, "array_max_abs_error": worst,
            "metadata_values_compared": mc, "metadata_max_abs_error": me,
            "declared_metadata_differences_excluded": sorted(DECLARED_META_DIFFERENCES),
            "tolerance": 0.0,
            "status": "PASS" if max(worst, me) == 0.0 else "FAIL",
            "mismatches": (bad + mm)[:20],
        })
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    static = static_audit()
    behavior = behavioral_audit()
    for name, row in static.items():
        print(f"{name}: ast_identical={row['ast_identical_ignoring_docstrings']} "
              f"changed={row['changed_top_level_statements']}")
    for p in PROFILES:
        rows = [r for r in behavior if r["configuration"] == p]
        print(f"{p}: {len(rows)} scenarios, arrays max err "
              f"{max(r['array_max_abs_error'] for r in rows):.3e}, metadata max err "
              f"{max(r['metadata_max_abs_error'] for r in rows):.3e}, "
              f"{'PASS' if all(r['status'] == 'PASS' for r in rows) else 'FAIL'}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"static": static, "behavioral": behavior}, indent=2, default=str) + "\n",
                            encoding="utf-8", newline="\n")
    physics_modules_ok = all(static[m]["ast_identical_ignoring_docstrings"]
                             for m in MODULES if m != "provenance.py")
    return 0 if physics_modules_ok and all(r["status"] == "PASS" for r in behavior) else 1


if __name__ == "__main__":
    raise SystemExit(main())

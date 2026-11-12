"""CLI demo entry for shared four-step reconstruction examples."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verification.phase_reconstruction.examples import run_all_examples  # noqa: E402


def main() -> int:
    results = run_all_examples(shape=(32, 32))
    summary = []
    for r in results:
        summary.append(
            {
                "example_id": r["spec"].example_id,
                "method": r["reconstruction"].method,
                "rmse_rad": r["error_metrics"]["rmse_rad"],
                "max_abs_rad": r["error_metrics"]["max_abs_rad"],
                "mean_visibility": float(r["reconstruction"].visibility_V.mean()),
            }
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

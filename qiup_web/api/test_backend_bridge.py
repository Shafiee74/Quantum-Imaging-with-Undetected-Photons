"""Direct unit tests for qiup_ui_backend helpers used by the web API."""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from unittest import mock

from django.test import SimpleTestCase


class BackendBridgeTests(SimpleTestCase):
    def test_defaults_profiles_match_ui(self) -> None:
        import qiup_ui_backend as backend

        payload = backend.defaults()
        self.assertEqual(sorted(payload["profiles"]), ["A", "B", "C", "D"])
        for pid, info in payload["profiles"].items():
            self.assertIn("architecture", info)
            self.assertIn("imaging_mode", info)
            self.assertIn("crystals", info)
            self.assertTrue(info["crystals"], f"{pid} has empty crystals")

    def test_iter_run_events_display_only(self) -> None:
        import qiup_ui_backend as backend

        with mock.patch.dict(os.environ, {"QIUP_UI_PERSIST": ""}, clear=False):
            events = list(
                backend.iter_run_events(
                    {
                        "profiles": ["A"],
                        "modes": {"ideal": True, "noisy": False},
                        "object": {"source": "mask", "mask_id": 1},
                        "size": 32,
                        "seed": 7,
                    }
                )
            )
        types = [e["type"] for e in events]
        self.assertEqual(types.count("result"), 1)
        self.assertIn("done", types)
        start = next(e for e in events if e["type"] == "start")
        self.assertIsNone(start.get("output_dir"))
        self.assertFalse(start.get("persisted"))

    def test_upload_resize_preserves_square_grid(self) -> None:
        import numpy as np
        from PIL import Image
        import qiup_ui_backend as backend

        img = Image.fromarray(np.full((50, 20), 200, dtype=np.uint8), mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        ref, info = backend._build_reference(
            {"source": "upload", "png_base64": b64, "file_name": "t.png"},
            32,
        )
        self.assertEqual(info["source"], "upload")
        self.assertEqual(ref["T"].shape, (32, 32))
        self.assertGreater(float(ref["T"].mean()), 0.0)

"""Django API compatibility tests for the QIUP simulator UI backend."""
from __future__ import annotations

import json

from django.test import Client, SimpleTestCase, override_settings


@override_settings(
    ALLOWED_HOSTS=["testserver", "localhost", "127.0.0.1"],
    CORS_ALLOWED_ORIGINS=["http://localhost:3000"],
)
class QiupApiTests(SimpleTestCase):
    """No DB migrations required — API is stateless over the simulator."""

    def setUp(self) -> None:
        self.client = Client()

    def test_defaults_lists_abcd_profiles(self) -> None:
        response = self.client.get("/api/defaults")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(sorted(payload["profiles"]), ["A", "B", "C", "D"])
        self.assertEqual(payload["backend"]["framework"], "django")
        self.assertIn("package_version", payload)

    def test_preview_mask_object(self) -> None:
        body = {"object": {"source": "mask", "mask_id": 5}, "size": 48}
        response = self.client.post(
            "/api/preview",
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        event = response.json()
        self.assertEqual(event["type"], "object")
        self.assertEqual(event["source"], "mask")
        self.assertIn("png", event["image"])
        self.assertEqual(event["image"]["shape"], [48, 48])

    def test_preview_rejects_bad_size(self) -> None:
        body = {"object": {"source": "mask", "mask_id": 5}, "size": 8}
        response = self.client.post(
            "/api/preview",
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    def test_run_streams_ndjson_for_mask_ideal_only(self) -> None:
        body = {
            "profiles": ["A"],
            "modes": {"ideal": True, "noisy": False},
            "object": {"source": "mask", "mask_id": 5},
            "size": 48,
            "seed": 15000,
        }
        response = self.client.post(
            "/api/run",
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("ndjson", response["Content-Type"])
        raw = b"".join(response.streaming_content).decode("utf-8")
        events = [json.loads(line) for line in raw.splitlines() if line.strip()]
        types = [event["type"] for event in events]
        self.assertIn("start", types)
        self.assertIn("object", types)
        self.assertIn("result", types)
        self.assertIn("done", types)
        self.assertNotIn("error", types)
        result = next(event for event in events if event["type"] == "result")
        self.assertEqual(result["profile"], "A")
        self.assertEqual(result["mode"], "ideal")
        self.assertIn("png", result["image"])

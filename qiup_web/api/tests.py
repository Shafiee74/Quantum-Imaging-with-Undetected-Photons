"""Django API compatibility tests for the QIUP simulator UI backend."""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from unittest import mock

from django.test import Client, SimpleTestCase, override_settings


REPO = Path(__file__).resolve().parents[2]
RABBIT = REPO / "qiup-ui" / "public" / "test-masks" / "rabbit-slm-mask.png"


@override_settings(
    ALLOWED_HOSTS=["testserver", "localhost", "127.0.0.1"],
    CORS_ALLOWED_ORIGINS=[],
)
class QiupApiTests(SimpleTestCase):
    """No DB migrations required — API is stateless over the simulator."""

    def setUp(self) -> None:
        self.client = Client(enforce_csrf_checks=True)
        # Open the website UI first — issues qiup_ui_gate + csrftoken cookies.
        home = self.client.get("/")
        self.assertEqual(home.status_code, 200)
        self.assertIn("qiup_ui_gate", self.client.cookies)
        self.assertIn("csrftoken", self.client.cookies)

    def _csrf_header(self) -> dict:
        return {"HTTP_X_CSRFTOKEN": self.client.cookies["csrftoken"].value}

    def _ui_post_headers(self) -> dict:
        # Tests run with DEBUG=False; production API requires same-site Origin/Referer.
        return {
            **self._csrf_header(),
            "HTTP_ORIGIN": "http://testserver",
            "HTTP_REFERER": "http://testserver/",
        }

    def _json(self, path: str, body: dict, method: str = "post", **extra):
        caller = getattr(self.client, method)
        headers = self._ui_post_headers() if method.lower() == "post" else {}
        headers.update(extra)
        return caller(path, data=json.dumps(body), content_type="application/json", **headers)

    def _run_events(self, body: dict) -> list[dict]:
        response = self._json("/api/run", body)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.streaming)
        self.assertIn("ndjson", response["Content-Type"])
        raw = b"".join(response.streaming_content).decode("utf-8")
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    def test_api_rejects_without_ui_gate_cookie(self) -> None:
        bare = Client()
        response = bare.get("/api/defaults")
        self.assertEqual(response.status_code, 403)
        self.assertIn("website UI", response.json()["error"])

    def test_api_post_rejects_without_csrf(self) -> None:
        response = self.client.post(
            "/api/preview",
            data=json.dumps({"object": {"source": "mask", "mask_id": 5}, "size": 48}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 403)

    def test_defaults_lists_abcd_profiles(self) -> None:
        response = self.client.get("/api/defaults")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(sorted(payload["profiles"]), ["A", "B", "C", "D"])
        self.assertEqual(payload["backend"]["framework"], "django")
        self.assertIn("package_version", payload)

    def test_defaults_redacts_filesystem_root(self) -> None:
        payload = self.client.get("/api/defaults").json()
        self.assertEqual(payload["backend"]["root"], "local")
        self.assertNotIn("\\", payload["backend"]["root"])
        self.assertNotIn("/home/", str(payload["backend"]))

    def test_defaults_hides_model_notes_and_includes_ui_limits(self) -> None:
        payload = self.client.get("/api/defaults").json()
        self.assertNotIn("model_notes", payload)
        self.assertNotIn("calibration_status", payload)
        self.assertNotIn("release_classification", payload)
        self.assertEqual(payload["ui_limits"]["num_frames"], [1, 64])
        self.assertEqual(payload["ui_limits"]["thermal_C"], [20.0, 90.0])

    def test_preview_mask_object(self) -> None:
        body = {"object": {"source": "mask", "mask_id": 5}, "size": 48}
        response = self._json("/api/preview", body)
        self.assertEqual(response.status_code, 200)
        event = response.json()
        self.assertEqual(event["type"], "object")
        self.assertEqual(event["source"], "mask")
        self.assertIn("png", event["image"])
        self.assertEqual(event["image"]["shape"], [48, 48])

    def test_preview_rejects_bad_size(self) -> None:
        body = {"object": {"source": "mask", "mask_id": 5}, "size": 8}
        response = self._json("/api/preview", body)
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    def test_preview_rejects_non_json_content_type(self) -> None:
        response = self.client.post(
            "/api/preview",
            data='{"object":{"source":"mask","mask_id":5},"size":48}',
            content_type="text/plain",
            **self._ui_post_headers(),
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("application/json", response.json()["error"])

    def test_preview_rabbit_mask(self) -> None:
        if not RABBIT.is_file():
            self.skipTest(f"rabbit mask missing: {RABBIT}")
        response = self._json("/api/preview", {"object": {"source": "rabbit"}, "size": 32})
        self.assertEqual(response.status_code, 200)
        event = response.json()
        self.assertEqual(event["source"], "rabbit")
        self.assertEqual(event["image"]["shape"], [32, 32])

    def test_preview_upload_grayscale_mask(self) -> None:
        from PIL import Image
        import numpy as np

        arr = (np.linspace(0, 255, 40 * 24).reshape(40, 24)).astype("uint8")
        buf = io.BytesIO()
        Image.fromarray(arr, mode="L").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        response = self._json(
            "/api/preview",
            {"object": {"source": "upload", "png_base64": b64, "file_name": "g.png"}, "size": 32},
        )
        self.assertEqual(response.status_code, 200)
        event = response.json()
        self.assertEqual(event["source"], "upload")
        self.assertEqual(event["image"]["shape"], [32, 32])

    def test_preview_upload_rejects_rgb_photo(self) -> None:
        from PIL import Image
        import numpy as np

        rgb = np.zeros((24, 24, 3), dtype="uint8")
        rgb[..., 0] = 255
        rgb[..., 1] = 10
        rgb[..., 2] = 40
        buf = io.BytesIO()
        Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        response = self._json(
            "/api/preview",
            {"object": {"source": "upload", "png_base64": b64}, "size": 32},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("RGB", response.json()["error"])

    def test_run_streams_ndjson_for_mask_ideal_only(self) -> None:
        body = {
            "profiles": ["A"],
            "modes": {"ideal": True, "noisy": False},
            "object": {"source": "mask", "mask_id": 5},
            "size": 48,
            "seed": 15000,
        }
        events = self._run_events(body)
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
        self.assertIn("diagnostics", result)
        diag = result["diagnostics"]
        self.assertIn("profiles", diag)
        self.assertIn("histogram", diag)
        self.assertIn("snr", diag)
        self.assertIn("visibility", diag)
        self.assertIn("operating_point", diag)
        self.assertIn("nbar_1", diag["operating_point"])
        self.assertIn("plus_port", diag)
        self.assertIn("minus_port", diag)
        self.assertIn("png", diag["plus_port"])
        self.assertIn("png", diag["minus_port"])
        self.assertIn("T_eff_map", diag)
        self.assertIn("phi_eff_map", diag)
        self.assertIn("png", diag["phi_eff_map"])
        self.assertIn("thermal", result)
        self.assertIn("architecture", result)
        self.assertIn("imaging_mode", result["architecture"])
        self.assertIn("object_coefficient", result["architecture"])

    def test_run_does_not_persist_output_dir_by_default(self) -> None:
        with mock.patch.dict(os.environ, {"QIUP_UI_PERSIST": ""}, clear=False):
            events = self._run_events(
                {
                    "profiles": ["A"],
                    "modes": {"ideal": True, "noisy": False},
                    "object": {"source": "mask", "mask_id": 1},
                    "size": 32,
                    "seed": 1,
                }
            )
        start = next(e for e in events if e["type"] == "start")
        done = next(e for e in events if e["type"] == "done")
        self.assertIsNone(start.get("output_dir"))
        self.assertIsNone(done.get("output_dir"))
        self.assertFalse(start.get("persisted", True))

    def test_run_rejects_unknown_profile(self) -> None:
        response = self._json(
            "/api/run",
            {
                "profiles": ["Z"],
                "modes": {"ideal": True, "noisy": False},
                "object": {"source": "mask", "mask_id": 1},
                "size": 32,
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("unknown arrangement", response.json()["error"])

    def test_run_clamps_num_frames_above_ui_max(self) -> None:
        events = self._run_events(
            {
                "profiles": ["A"],
                "modes": {"ideal": True, "noisy": False},
                "object": {"source": "mask", "mask_id": 1},
                "size": 32,
                "seed": 2,
                "ideal_latents": {"num_frames": 999, "mean_photons_per_mode": 0.5},
            }
        )
        self.assertNotIn("error", [e["type"] for e in events])
        self.assertTrue(any(e["type"] == "result" for e in events))

    def test_run_rejects_non_json_content_type(self) -> None:
        response = self.client.post(
            "/api/run",
            data='{"profiles":["A"],"modes":{"ideal":true,"noisy":false},"object":{"source":"mask","mask_id":1},"size":32}',
            content_type="text/plain",
            **self._ui_post_headers(),
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("application/json", response.json()["error"])

    def test_run_all_profiles_ideal_smoke(self) -> None:
        events = self._run_events(
            {
                "profiles": ["A", "B", "C", "D"],
                "modes": {"ideal": True, "noisy": False},
                "object": {"source": "mask", "mask_id": 3},
                "size": 32,
                "seed": 11,
            }
        )
        results = [e for e in events if e["type"] == "result"]
        self.assertEqual(len(results), 4)
        self.assertEqual({(e["profile"], e["mode"]) for e in results}, {("A", "ideal"), ("B", "ideal"), ("C", "ideal"), ("D", "ideal")})
        self.assertTrue(all("png" in e["image"] for e in results))
        self.assertNotIn("error", [e["type"] for e in events])

    def test_run_noisy_profile_a_smoke(self) -> None:
        events = self._run_events(
            {
                "profiles": ["A"],
                "modes": {"ideal": False, "noisy": True},
                "object": {"source": "mask", "mask_id": 2},
                "size": 32,
                "seed": 42,
            }
        )
        types = [e["type"] for e in events]
        self.assertIn("start", types)
        self.assertIn("result", types)
        self.assertIn("done", types)
        self.assertNotIn("error", types)
        result = next(e for e in events if e["type"] == "result")
        self.assertEqual(result["profile"], "A")
        self.assertEqual(result["mode"], "noisy")
        self.assertIn("png", result["image"])

    def test_run_rejects_cross_site_origin(self) -> None:
        response = self._json(
            "/api/run",
            {
                "profiles": ["A"],
                "modes": {"ideal": True, "noisy": False},
                "object": {"source": "mask", "mask_id": 1},
                "size": 32,
            },
            HTTP_ORIGIN="https://evil.example",
            HTTP_REFERER="https://evil.example/",
        )
        self.assertEqual(response.status_code, 403)

    def test_run_ideal_noisy_includes_residual(self) -> None:
        events = self._run_events(
            {
                "profiles": ["A"],
                "modes": {"ideal": True, "noisy": True},
                "object": {"source": "mask", "mask_id": 2},
                "size": 32,
                "seed": 7,
            }
        )
        results = [e for e in events if e["type"] == "result"]
        self.assertEqual(len(results), 2)
        noisy = next(e for e in results if e["mode"] == "noisy")
        self.assertIn("residual_to_ideal", noisy["diagnostics"])
        self.assertIn("png", noisy["diagnostics"]["residual_to_ideal"])
        self.assertIn("residual_rms", noisy["diagnostics"])

    def _diagnostics_events(self, body: dict) -> list[dict]:
        response = self._json("/api/diagnostics", body)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.streaming)
        raw = b"".join(response.streaming_content).decode("utf-8")
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    def test_diagnostics_stage_ladder_streams(self) -> None:
        events = self._diagnostics_events(
            {
                "profiles": ["A"],
                "modes": {"ideal": True, "noisy": True},
                "object": {"source": "mask", "mask_id": 1},
                "size": 32,
                "seed": 3,
            }
        )
        types = [e["type"] for e in events]
        self.assertIn("start", types)
        self.assertIn("result", types)
        self.assertIn("done", types)
        self.assertNotIn("error", types)
        start = next(e for e in events if e["type"] == "start")
        self.assertEqual(start.get("diagnostic"), "stage_ladder")
        self.assertGreaterEqual(len(start.get("stages") or []), 4)
        stage_results = [e for e in events if e["type"] == "result" and e.get("mode") == "stage"]
        self.assertEqual(len(stage_results), len(start["stages"]))
        self.assertTrue(all(e.get("stage_id") for e in stage_results))
        self.assertTrue(all("diagnostics" in e for e in stage_results))
        ideal_stage = next(e for e in stage_results if e["stage_id"] == "ideal")
        full_stage = next(e for e in stage_results if e["stage_id"] == "full_noisy")
        self.assertIn("snr", ideal_stage["diagnostics"])
        self.assertIn("residual_rms", full_stage["diagnostics"])
        self.assertIn("visibility_map", ideal_stage["diagnostics"])
        self.assertIn("T_eff_map", ideal_stage["diagnostics"])
        self.assertIn("phi_eff_map", ideal_stage["diagnostics"])
        self.assertIn("transfer", ideal_stage["diagnostics"])

    def test_diagnostics_mechanism_matrix_streams(self) -> None:
        events = self._diagnostics_events(
            {
                "profiles": ["A"],
                "diagnostic": "mechanism_matrix",
                "modes": {"ideal": True, "noisy": False},
                "object": {"source": "mask", "mask_id": 1},
                "size": 32,
                "seed": 5,
            }
        )
        self.assertNotIn("error", [e["type"] for e in events])
        start = next(e for e in events if e["type"] == "start")
        self.assertEqual(start.get("diagnostic"), "mechanism_matrix")
        results = [e for e in events if e["type"] == "result" and e.get("mode") == "mechanism"]
        self.assertGreaterEqual(len(results), 5)
        self.assertTrue(any(e.get("stage_id") == "clean" for e in results))
        self.assertTrue(any(e.get("stage_id") == "read_noise" for e in results))

    def test_diagnostics_phase_step_streams(self) -> None:
        events = self._diagnostics_events(
            {
                "profiles": ["A"],
                "diagnostic": "phase_step",
                "modes": {"ideal": True, "noisy": False},
                "object": {"source": "mask", "mask_id": 2},
                "size": 32,
                "seed": 9,
            }
        )
        self.assertNotIn("error", [e["type"] for e in events])
        result = next(e for e in events if e["type"] == "result")
        self.assertEqual(result.get("mode"), "phase_step")
        self.assertIn("phase_map", result["diagnostics"])
        self.assertIn("phase_modulation_map", result["diagnostics"])

    def test_diagnostics_sweep_streams(self) -> None:
        events = self._diagnostics_events(
            {
                "profiles": ["A"],
                "diagnostic": "sweep",
                "sweep_param": "s_focus",
                "sweep_points": 3,
                "modes": {"ideal": True, "noisy": True},
                "object": {"source": "mnist", "digit": 5},
                "size": 32,
                "seed": 4,
            }
        )
        types = [e["type"] for e in events]
        self.assertIn("sweep_curve", types)
        self.assertNotIn("error", types)
        curve = next(e for e in events if e["type"] == "sweep_curve")
        self.assertEqual(curve.get("param"), "s_focus")
        points = curve.get("curve") or []
        self.assertEqual(len(points), 3)
        rms = [p.get("residual_rms") for p in points]
        self.assertTrue(all(v is not None for v in rms))
        self.assertGreater(max(rms) - min(rms), 1e-6)

    def test_diagnostics_sweep_thermal_moves(self) -> None:
        events = self._diagnostics_events(
            {
                "profiles": ["A"],
                "diagnostic": "sweep",
                "sweep_param": "s_thermal",
                "sweep_points": 3,
                "modes": {"ideal": True, "noisy": True},
                "object": {"source": "mnist", "digit": 5},
                "size": 32,
                "seed": 5,
            }
        )
        self.assertNotIn("error", [e["type"] for e in events])
        curve = next(e for e in events if e["type"] == "sweep_curve")
        points = curve.get("curve") or []
        self.assertEqual(len(points), 3)
        vis = [p.get("V_50_50") for p in points]
        rms = [p.get("residual_rms") for p in points]
        moved = (max(vis) - min(vis) > 1e-6) or (max(rms) - min(rms) > 1e-6)
        self.assertTrue(moved)

    def test_diagnostics_fringe_curve_streams(self) -> None:
        events = self._diagnostics_events(
            {
                "profiles": ["A"],
                "diagnostic": "fringe_curve",
                "fringe_points": 5,
                "modes": {"ideal": True, "noisy": False},
                "object": {"source": "mnist", "digit": 5},
                "size": 32,
                "seed": 6,
            }
        )
        types = [e["type"] for e in events]
        self.assertIn("fringe_curve", types)
        self.assertNotIn("error", types)
        start = next(e for e in events if e["type"] == "start")
        self.assertEqual(start.get("diagnostic"), "fringe_curve")
        curve_ev = next(e for e in events if e["type"] == "fringe_curve")
        points = curve_ev.get("curve") or []
        self.assertEqual(len(points), 5)
        self.assertTrue(all("phi_rad" in p and "mean_I_plus" in p for p in points))
        iplus = [p["mean_I_plus"] for p in points]
        iminus = [p["mean_I_minus"] for p in points]
        # Interferometric teaching curve should not be flat on both ports.
        moved = (max(iplus) - min(iplus) > 1e-9) or (max(iminus) - min(iminus) > 1e-9)
        self.assertTrue(moved)
        summary = curve_ev.get("summary") or {}
        self.assertIn("michelson_I_plus", summary)
        self.assertGreaterEqual(float(summary["michelson_I_plus"]), 0.0)
        self.assertLessEqual(float(summary["michelson_I_plus"]), 1.0 + 1e-9)


@override_settings(ALLOWED_HOSTS=["testserver", "localhost", "127.0.0.1"])
class ApiSanitizeUnitTests(SimpleTestCase):
    def test_sanitize_clamps_abuse_fields(self) -> None:
        from api.views import _sanitize_run_config

        cfg = _sanitize_run_config(
            {
                "profiles": ["A", "B"],
                "modes": {"ideal": True, "noisy": True},
                "object": {"source": "mask", "mask_id": 2},
                "size": 200,
                "seed": -3,
                "noisy_latents": {"num_frames": 1000, "mean_photons_per_mode": 500, "s_thermal": 1.0},
                "noisy_setup": {"pump_waist_um": 900, "pixel_um": 3.0},
                "thermal": {"A": {"T_pm0_NL1_C": 5, "dT_NL1_C": 100}},
            }
        )
        self.assertEqual(cfg["size"], 128)
        self.assertEqual(cfg["seed"], 0)
        self.assertEqual(cfg["noisy_latents"]["num_frames"], 64)
        self.assertEqual(cfg["noisy_latents"]["mean_photons_per_mode"], 20.0)
        self.assertEqual(cfg["noisy_latents"]["s_thermal"], 0.99)
        self.assertEqual(cfg["noisy_setup"]["pump_waist_um"], 350.0)
        self.assertEqual(cfg["noisy_setup"]["pixel_um"], 6.5)
        self.assertEqual(cfg["thermal"]["A"]["T_pm0_NL1_C"], 20.0)
        self.assertEqual(cfg["thermal"]["A"]["dT_NL1_C"], 30.0)

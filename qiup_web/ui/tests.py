"""UI page and static-asset smoke tests for the Django QIUP web app."""
from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.test import Client, SimpleTestCase, override_settings


@override_settings(ALLOWED_HOSTS=["testserver", "localhost", "127.0.0.1"])
class UiHomeTests(SimpleTestCase):
    def setUp(self) -> None:
        self.client = Client()

    def test_home_serves_simulator_page(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        body = response.content.decode("utf-8")
        self.assertIn("QIUP Simulator", body)
        self.assertIn("/static/ui/app.js", body)
        self.assertIn("Run simulation", body)

    def test_home_includes_logos_and_config_load(self) -> None:
        response = self.client.get("/")
        body = response.content.decode("utf-8")
        self.assertIn("CQST_Logo.png", body)
        self.assertIn("CIDAIT_Logo.png", body)
        self.assertIn("favicon.svg", body)
        self.assertIn('id="btn-config-load"', body)
        self.assertIn('id="config-upload-input"', body)
        self.assertNotIn('id="model-notes"', body)
        self.assertNotIn("prototype · pre-validation", body)
        self.assertIn('id="log"', body)
        self.assertIn('id="sim-panel"', body)
        self.assertIn('id="credits"', body)
        self.assertIn("aria-live", body)
        self.assertIn("qiup_ui_gate", response.cookies)
        self.assertIn("csrftoken", response.cookies)

    def test_home_cache_busts_static_assets(self) -> None:
        body = self.client.get("/").content.decode("utf-8")
        self.assertIn("app.css?v=", body)
        self.assertIn("app.js?v=", body)

    def test_static_app_js_and_css_contents(self) -> None:
        static_root = Path(settings.BASE_DIR) / "static" / "ui"
        js_path = static_root / "app.js"
        css_path = static_root / "app.css"
        self.assertTrue(js_path.is_file())
        self.assertTrue(css_path.is_file())
        js_text = js_path.read_text(encoding="utf-8")
        css_text = css_path.read_text(encoding="utf-8")
        self.assertIn("renderSidebar", js_text)
        self.assertIn("loadConfigFile", js_text)
        self.assertIn("updatePresetMetric", js_text)
        self.assertIn("enlarge-body-light", js_text)
        self.assertIn('textContent === "complete"', js_text)
        self.assertIn("apiFetch", js_text)
        self.assertIn("X-CSRFToken", js_text)
        self.assertIn("sidebar-scroll", css_text)
        self.assertIn("arr-card", css_text)
        self.assertIn("credits-card", css_text)
        self.assertNotIn("renderModelNotes", js_text)

    def test_architecture_and_logo_static_files_exist(self) -> None:
        static_root = Path(settings.BASE_DIR) / "static" / "ui"
        for name in (
            "img/CQST_Logo.png",
            "img/CIDAIT_Logo.png",
            "img/favicon.svg",
            "Architectures/Fig1-A.png",
            "Architectures/Fig1-B.png",
            "Architectures/Fig1-C.png",
            "Architectures/Fig1-D.png",
            "test-masks/rabbit-slm-mask.png",
            "app.js",
            "app.css",
        ):
            path = static_root / name
            self.assertTrue(path.is_file(), f"missing static file: {path}")
            self.assertGreater(path.stat().st_size, 0, name)

    def test_unknown_api_path_is_not_ui_200_html(self) -> None:
        # Django should 404 for unknown API routes (review B3 style).
        response = self.client.get("/api/doesnotexist")
        self.assertEqual(response.status_code, 404)

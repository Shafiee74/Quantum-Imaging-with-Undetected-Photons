from django.test import Client, SimpleTestCase, override_settings


@override_settings(ALLOWED_HOSTS=["testserver", "localhost", "127.0.0.1"])
class UiHomeTests(SimpleTestCase):
    def test_home_serves_simulator_page(self) -> None:
        response = Client().get("/")
        self.assertEqual(response.status_code, 200)
        body = response.content.decode("utf-8")
        self.assertIn("QIUP Simulator", body)
        self.assertIn("/static/ui/app.js", body)
        self.assertIn("Run simulation", body)

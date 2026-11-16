"""Flask prediction/model API: auth, authorization, validation, persistence, error handling.

The model behind these endpoints was trained on SYNTHETIC data purely to exercise the code paths.
"""
import os
import re
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# The app creates its database at import time: point it at a throw-away file first (never the real one).
os.environ["DATABASE_PATH"] = os.path.join(tempfile.mkdtemp(prefix="apitest_"), "import.db")

from werkzeug.security import generate_password_hash

import app as A
from machine_learning import model_config as cfg
from machine_learning.prediction_service import PredictionService
from tests.helpers import trained_env


def _login(client, email, password):
    page = client.get("/login").get_data(as_text=True)
    token = re.search(r'name="csrf_token" value="([^"]+)"', page).group(1)
    r = client.post("/login", data={"csrf_token": token, "email": email, "password": password})
    assert r.status_code == 302, "login failed"
    return re.search(r'data-csrf="([^"]+)"', client.get("/dashboard").get_data(as_text=True)).group(1)


class ApiCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # fresh_until_now: the newest measurement is only minutes old, so the API's own clock sees fresh data
        cls.env = trained_env(days=14, horizons=(30,), fresh_until_now=True)

    def setUp(self):
        d = Path(tempfile.mkdtemp())
        shutil.copy(self.env["db_path"], d / "api.db")
        self._patches = [mock.patch.dict(A.app.config, {"DATABASE": str(d / "api.db")}),
                         mock.patch.object(cfg, "MODELS_DIR", self.env["models_dir"]),
                         mock.patch.object(A, "PREDICTOR", PredictionService(models_dir=self.env["models_dir"]))]
        for p in self._patches:
            p.start()
        A.init_db()                                  # adds users/incidents/notifications to the copy
        with sqlite3.connect(d / "api.db") as c:
            for name, email, role in (("Admin A", "admin@test.co", "admin"), ("User U", "user@test.co", "commuter")):
                c.execute("INSERT INTO users(full_name,email,password_hash,role,created_at) VALUES(?,?,?,?,'2026-01-01 00:00:00')",
                          (name, email, generate_password_hash("passw0rd!"), role))
        self.db_path = d / "api.db"
        self.anon = A.app.test_client()
        self.user = A.app.test_client(); self.user_csrf = _login(self.user, "user@test.co", "passw0rd!")
        self.admin = A.app.test_client(); self.admin_csrf = _login(self.admin, "admin@test.co", "passw0rd!")

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()

    def count(self, table):
        with sqlite3.connect(self.db_path) as c:
            return c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


class TestAccessControl(ApiCase):
    def test_anonymous_users_are_rejected(self):
        for url in ("/api/predictions", "/api/predictions/1?horizon=30", "/api/predictions/1/history", "/api/models/status"):
            self.assertEqual(self.anon.get(url).status_code, 401, url)

    def test_model_metrics_are_admin_only(self):
        self.assertEqual(self.user.get("/api/models/metrics").status_code, 403)
        self.assertEqual(self.admin.get("/api/models/metrics").status_code, 200)

    def test_generating_predictions_is_admin_only_and_needs_csrf(self):
        h = {"X-CSRF-Token": self.user_csrf}
        self.assertEqual(self.user.post("/api/predictions/generate", json={}, headers=h).status_code, 403)
        self.assertEqual(self.admin.post("/api/predictions/generate", json={}).status_code, 400)       # no CSRF token
        self.assertEqual(self.count("traffic_predictions"), 0)


class TestCatalogueAndStatus(ApiCase):
    def test_catalogue_when_a_model_is_deployed(self):
        d = self.user.get("/api/predictions").get_json()
        self.assertTrue(d["available"]); self.assertEqual(d["model_status"], "ready"); self.assertEqual(d["horizons"], [30])
        self.assertEqual([x["name"] for x in d["locations"]], ["Test Road 1", "Test Road 2", "Test Road 3"])
        self.assertTrue(d["data"]["fresh"])

    def test_catalogue_without_a_model_says_so_and_invents_nothing(self):
        with mock.patch.object(A, "PREDICTOR", PredictionService(models_dir=tempfile.mkdtemp())):
            d = self.user.get("/api/predictions").get_json()
        self.assertFalse(d["available"]); self.assertEqual(d["model_status"], "not_connected")
        self.assertEqual(d["locations"], []); self.assertIn("train_model", d["message"])

    def test_models_status(self):
        d = self.user.get("/api/models/status").get_json()
        self.assertEqual(d["model_status"], "ready"); self.assertIn("30", d["models"]); self.assertIn("version", d["models"]["30"])


class TestPredictionEndpoints(ApiCase):
    def test_single_prediction_by_query_parameters(self):
        d = self.user.get("/api/predictions?location=1&horizon=30").get_json()
        self.assertTrue(d["available"]); self.assertEqual(d["road_name"], "Test Road 1")
        self.assertIn(d["predicted_congestion"], cfg.CLASS_LABELS)

    def test_single_prediction_by_road_path(self):
        d = self.user.get("/api/predictions/2?horizon=30").get_json()
        self.assertTrue(d["available"]); self.assertEqual(d["road_segment_id"], 2)

    def test_malformed_parameters_are_422(self):
        for url in ("/api/predictions?location=abc&horizon=30", "/api/predictions?location=1", "/api/predictions?location=-1&horizon=30",
                    "/api/predictions/1", "/api/predictions/1?horizon=abc", "/api/predictions?location=1&horizon=30.5"):
            self.assertEqual(self.user.get(url).status_code, 422, url)

    def test_unknown_road_and_unsupported_horizon_are_normal_unavailable_answers(self):
        a = self.user.get("/api/predictions/9999?horizon=30"); b = self.user.get("/api/predictions/1?horizon=45")
        self.assertEqual((a.status_code, b.status_code), (200, 200))
        self.assertEqual(a.get_json()["reason"], "unknown_segment"); self.assertEqual(b.get_json()["reason"], "unsupported_horizon")
        self.assertFalse(a.get_json()["available"])

    def test_sql_injection_in_parameters_is_harmless(self):
        r = self.user.get("/api/predictions?location=1%20OR%201=1&horizon=30")
        self.assertEqual(r.status_code, 422)
        self.assertEqual(self.count("road_segments"), 3)

    def test_stale_data_is_reported_not_predicted(self):
        with sqlite3.connect(self.db_path) as c:                     # age every measurement by 3 hours
            c.execute("UPDATE traffic_records SET recorded_at = datetime(recorded_at, '-3 hours')")
        d = self.user.get("/api/predictions/1?horizon=30").get_json()
        self.assertEqual(d["reason"], "stale_data"); self.assertNotIn("predicted_speed_kmh", d)


class TestStoredPredictions(ApiCase):
    def test_admin_can_generate_and_users_can_read_the_history(self):
        h = {"X-CSRF-Token": self.admin_csrf}
        r = self.admin.post("/api/predictions/generate", json={"road_id": 1, "horizon": 30}, headers=h)
        self.assertEqual(r.status_code, 200); self.assertTrue(r.get_json()["available"])
        self.assertEqual(self.count("traffic_predictions"), 1)
        hist = self.user.get("/api/predictions/1/history").get_json()
        self.assertEqual(hist["kind"], "model_predictions"); self.assertEqual(len(hist["predictions"]), 1)

    def test_generating_twice_does_not_duplicate(self):
        h = {"X-CSRF-Token": self.admin_csrf}
        for _ in range(2):
            self.admin.post("/api/predictions/generate", json={"road_id": 1, "horizon": 30}, headers=h)
        self.assertEqual(self.count("traffic_predictions"), 1)

    def test_generate_everything(self):
        r = self.admin.post("/api/predictions/generate", json={}, headers={"X-CSRF-Token": self.admin_csrf})
        self.assertEqual(r.get_json()["stored"], 3); self.assertEqual(self.count("traffic_predictions"), 3)

    def test_generate_validates_its_body(self):
        h = {"X-CSRF-Token": self.admin_csrf}
        for body in ({"road_id": "x", "horizon": 30}, {"road_id": 1}, {"horizon": 30}):
            self.assertEqual(self.admin.post("/api/predictions/generate", json=body, headers=h).status_code, 422, body)

    def test_history_for_a_road_with_no_predictions_is_empty(self):
        self.assertEqual(self.user.get("/api/predictions/3/history").get_json()["predictions"], [])


class TestMetricsEndpoint(ApiCase):
    def test_metrics_come_from_the_training_run(self):
        d = self.admin.get("/api/models/metrics").get_json()
        h = d["horizons"]["30"]
        self.assertEqual(h["version"], self.env["result"]["horizons"]["30"]["version"])
        self.assertTrue(any(r["kind"] == "baseline" for r in h["test"]))
        self.assertIn("realized", d)

    def test_realized_error_compares_stored_predictions_with_later_measurements(self):
        # store a prediction whose target time has a measurement, then check it is scored against it
        with sqlite3.connect(self.db_path) as c:
            row = c.execute("SELECT recorded_at, speed_kmh FROM traffic_records WHERE segment_id=1 ORDER BY recorded_at DESC LIMIT 1").fetchone()
            c.execute("INSERT INTO traffic_predictions(segment_id,horizon_min,predicted_speed_kmh,predicted_congestion,estimated_delay_min,"
                      "target_time,generated_at,source_observation_at,model_version) VALUES(1,30,?,?,NULL,?,?,?,'v-test')",
                      (row[1] + 4.0, "low", row[0], row[0], row[0]))
        d = self.admin.get("/api/models/metrics").get_json()["realized"]
        self.assertEqual(d["n"], 1); self.assertAlmostEqual(d["mae_kmh"], 4.0, places=6)


class TestErrorHandling(ApiCase):
    def test_internal_errors_do_not_leak_details(self):
        with mock.patch.object(A.PREDICTOR, "predict", side_effect=RuntimeError("secret path /srv/db password=hunter2")):
            r = self.user.get("/api/predictions/1?horizon=30")
        self.assertEqual(r.status_code, 500)
        self.assertNotIn("hunter2", r.get_data(as_text=True)); self.assertNotIn("/srv", r.get_data(as_text=True))
        self.assertNotIn("Traceback", r.get_data(as_text=True))

    def test_database_failure_gives_a_generic_error(self):
        with mock.patch.object(A, "ml_db", side_effect=sqlite3.OperationalError("unable to open /secret/path.db")):
            for url in ("/api/predictions", "/api/predictions/1?horizon=30", "/api/models/status", "/api/predictions/1/history"):
                r = self.user.get(url)
                self.assertEqual(r.status_code, 500, url)
                self.assertNotIn("/secret", r.get_data(as_text=True))

    def test_missing_model_file_is_a_normal_unavailable_answer(self):
        d = Path(tempfile.mkdtemp()) / "m"; shutil.copytree(self.env["models_dir"], d)
        next(d.glob("h30/*/bundle.joblib")).unlink()
        with mock.patch.object(A, "PREDICTOR", PredictionService(models_dir=str(d))):
            r = self.user.get("/api/predictions/1?horizon=30")
        self.assertEqual(r.status_code, 200); self.assertEqual(r.get_json()["reason"], "model_error")
        self.assertNotIn(str(d), r.get_data(as_text=True))


class TestRoutingUsesPredictions(ApiCase):
    def test_routes_are_not_invented_when_no_engine_is_connected(self):
        r = self.user.post("/api/routes", json={"origin": "A", "destination": "B"}, headers={"X-CSRF-Token": self.user_csrf})
        self.assertEqual(r.status_code, 501); self.assertEqual(r.get_json()["routes"], [])

    def test_a_connected_engine_gets_congestion_aware_ranking(self):
        # two candidate routes over the same two roads; the engine is a stub (no real routing is claimed)
        cands = [{"id": "r1", "name": "Via road 1", "distance_km": 3.0, "base_duration_min": 6.0, "segments": [{"segment_id": 1, "length_km": 3.0}]},
                 {"id": "r2", "name": "Via road 2", "distance_km": 4.0, "base_duration_min": 8.0, "segments": [{"segment_id": 2, "length_km": 4.0}]}]
        with mock.patch.object(A, "ROUTING_ENGINE", lambda o, d: cands):
            r = self.user.post("/api/routes", json={"origin": "A", "destination": "B"}, headers={"X-CSRF-Token": self.user_csrf})
        self.assertEqual(r.status_code, 200)
        routes = r.get_json()["routes"]
        self.assertEqual({x["id"] for x in routes}, {"r1", "r2"})
        self.assertEqual([x["recommended"] for x in routes], [True, False])
        self.assertLessEqual(routes[0]["duration_min"], routes[1]["duration_min"])
        self.assertTrue(all(x["traffic_coverage"] == 1.0 for x in routes))          # both roads priced from real speeds
        self.assertTrue(all("predicted" in x["speed_basis"] for x in routes))       # the model's predictions were used


if __name__ == "__main__":
    unittest.main()

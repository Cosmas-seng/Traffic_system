"""Prediction service: valid/invalid inputs, unavailable states, no future leakage, persistence.

Uses a model trained on SYNTHETIC data (mechanics only). Values are not real traffic predictions.
"""
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from machine_learning import data_loader as dl
from machine_learning import model_config as cfg
from machine_learning.prediction_service import PredictionService
from tests.helpers import trained_env


def _copy_db(env):
    """A private copy so a test can change data without touching the shared cached fixture."""
    p = Path(tempfile.mkdtemp()) / "t.db"
    shutil.copy(env["db_path"], p)
    return dl.Database(sqlite3.connect(str(p)))


class ServiceCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = trained_env(days=14, horizons=(30,))
        cls.now = cls.env["now"]                         # two minutes after the newest measurement: data is fresh
        cls.svc = PredictionService(models_dir=cls.env["models_dir"])

    def setUp(self):
        self.db = _copy_db(self.env)

    def tearDown(self):
        self.db.conn.close()


class TestValidPrediction(ServiceCase):
    def test_returns_a_complete_prediction(self):
        r = self.svc.predict(self.db, 1, 30, now=self.now)
        self.assertTrue(r["available"])
        self.assertEqual(r["prediction_status"], "available")
        for key in ("road_segment_id", "road_name", "prediction_horizon_minutes", "predicted_speed_kmh", "predicted_congestion",
                    "estimated_delay_minutes", "prediction_timestamp", "generated_at", "observed_at", "model", "history", "forecast"):
            self.assertIn(key, r, key)
        self.assertEqual(r["road_name"], "Test Road 1")
        self.assertIn(r["predicted_congestion"], cfg.CLASS_LABELS)
        self.assertGreaterEqual(r["predicted_speed_kmh"], 0)
        self.assertGreaterEqual(r["estimated_delay_minutes"], 0)

    def test_prediction_timestamp_is_latest_measurement_plus_horizon(self):
        r = self.svc.predict(self.db, 2, 30, now=self.now)
        self.assertEqual(pd.Timestamp(r["prediction_timestamp"]) - pd.Timestamp(r["observed_at"]), pd.Timedelta(minutes=30))

    def test_model_version_is_the_deployed_one(self):
        r = self.svc.predict(self.db, 1, 30, now=self.now)
        self.assertEqual(r["model"]["version"], self.env["result"]["horizons"]["30"]["version"])

    def test_history_is_measured_data_and_forecast_is_the_prediction(self):
        r = self.svc.predict(self.db, 1, 30, now=self.now)
        self.assertTrue(r["history"])
        self.assertEqual(len(r["forecast"]), 1)
        self.assertEqual(r["forecast"][0]["value"], r["predicted_speed_kmh"])
        self.assertLessEqual(pd.Timestamp(r["history"][-1]["t"]), pd.Timestamp(r["observed_at"]))

    def test_a_freshly_loaded_service_gives_the_same_answer(self):
        a = self.svc.predict(self.db, 1, 30, now=self.now)
        b = PredictionService(models_dir=self.env["models_dir"]).predict(self.db, 1, 30, now=self.now)
        self.assertEqual(a["predicted_speed_kmh"], b["predicted_speed_kmh"])
        self.assertEqual(a["predicted_congestion"], b["predicted_congestion"])

    def test_delay_is_unknown_when_road_length_is_unknown(self):
        self.db.execute("UPDATE road_segments SET length_km=NULL WHERE id=1"); self.db.commit()
        r = self.svc.predict(self.db, 1, 30, now=self.now)
        self.assertTrue(r["available"])
        self.assertIsNone(r["estimated_delay_minutes"])

    def test_catalogue_lists_only_roads_that_have_measurements(self):
        self.db.execute("INSERT INTO road_segments(id,name) VALUES(99,'No data road')"); self.db.commit()
        ids = [s["id"] for s in self.svc.catalogue(self.db)]
        self.assertEqual(ids, [1, 2, 3])


class TestInvalidInput(ServiceCase):
    def test_unknown_road(self):
        r = self.svc.predict(self.db, 9999, 30, now=self.now)
        self.assertFalse(r["available"]); self.assertEqual(r["reason"], "unknown_segment")

    def test_non_numeric_input(self):
        for bad in ("abc", None, "1; DROP TABLE road_segments"):
            r = self.svc.predict(self.db, bad, 30, now=self.now)
            self.assertEqual(r["reason"], "invalid_input", bad)
        self.assertEqual(self.svc.predict(self.db, 1, "x", now=self.now)["reason"], "invalid_input")
        self.assertTrue(self.db.has_table("road_segments"))

    def test_unsupported_horizon_lists_the_supported_ones(self):
        r = self.svc.predict(self.db, 1, 45, now=self.now)
        self.assertEqual(r["reason"], "unsupported_horizon")
        self.assertEqual(r["supported_horizons"], [30])

    def test_road_the_model_never_saw(self):
        self.db.execute("INSERT INTO road_segments(id,name,length_km) VALUES(50,'New road',2.0)")
        self.db.execute("INSERT INTO traffic_records(segment_id,recorded_at,speed_kmh) VALUES(50,?,30)",
                        (self.env["last_obs"].strftime(dl.TS_FMT),))
        self.db.commit()
        self.assertEqual(self.svc.predict(self.db, 50, 30, now=self.now)["reason"], "segment_not_in_model")


class TestUnavailableStates(ServiceCase):
    def test_no_model_deployed(self):
        r = PredictionService(models_dir=tempfile.mkdtemp()).predict(self.db, 1, 30, now=self.now)
        self.assertFalse(r["available"]); self.assertEqual(r["reason"], "no_model"); self.assertEqual(r["model_status"], "not_connected")

    def test_missing_model_file_is_reported_safely(self):
        d = Path(tempfile.mkdtemp()) / "models"; shutil.copytree(self.env["models_dir"], d)
        next(d.glob("h30/*/bundle.joblib")).unlink()
        r = PredictionService(models_dir=str(d)).predict(self.db, 1, 30, now=self.now)
        self.assertEqual(r["reason"], "model_error"); self.assertEqual(r["model_status"], "error")
        self.assertNotIn(str(d), str(r))                       # no file paths leak to the client

    def test_corrupt_model_file_is_refused(self):
        d = Path(tempfile.mkdtemp()) / "models"; shutil.copytree(self.env["models_dir"], d)
        p = next(d.glob("h30/*/bundle.joblib")); p.write_bytes(b"not a model")
        self.assertEqual(PredictionService(models_dir=str(d)).predict(self.db, 1, 30, now=self.now)["reason"], "model_error")

    def test_stale_observations_are_not_used(self):
        stale_now = self.env["last_obs"] + pd.Timedelta(minutes=cfg.MAX_OBSERVATION_AGE_MIN + 30)
        r = self.svc.predict(self.db, 1, 30, now=stale_now)
        self.assertEqual(r["reason"], "stale_data")
        self.assertGreater(r["data_age_minutes"], cfg.MAX_OBSERVATION_AGE_MIN)
        self.assertNotIn("predicted_speed_kmh", r)

    def test_a_gap_in_recent_data_means_insufficient_data(self):
        last = self.env["last_obs"]
        lo, hi = (last - pd.Timedelta(minutes=120)).strftime(dl.TS_FMT), (last - pd.Timedelta(minutes=5)).strftime(dl.TS_FMT)
        self.db.execute("DELETE FROM traffic_records WHERE segment_id=1 AND recorded_at>=? AND recorded_at<?", (lo, hi))
        self.db.commit()
        r = self.svc.predict(self.db, 1, 30, now=self.now)
        self.assertIn(r["reason"], ("insufficient_data", "stale_data"))
        self.assertFalse(r["available"])

    def test_road_without_any_measurements(self):
        self.db.execute("DELETE FROM traffic_records WHERE segment_id=3"); self.db.commit()
        self.assertEqual(self.svc.predict(self.db, 3, 30, now=self.now)["reason"], "no_data")


class TestNoFutureLeakage(ServiceCase):
    def test_measurements_after_the_prediction_moment_are_ignored(self):
        before = self.svc.predict(self.db, 1, 30, now=self.now)
        future = [(1, (self.now + pd.Timedelta(minutes=m)).strftime(dl.TS_FMT), 1.0, None, None, "future-test") for m in (1, 3, 5, 7)]
        self.db.conn.cursor().executemany(self.db.q("INSERT INTO traffic_records(segment_id,recorded_at,speed_kmh,vehicle_count,density,source) "
                                                    "VALUES(?,?,?,?,?,?)"), future)
        self.db.commit()
        after = self.svc.predict(self.db, 1, 30, now=self.now)
        self.assertEqual(before["predicted_speed_kmh"], after["predicted_speed_kmh"])
        self.assertEqual(before["observed_at"], after["observed_at"])


class TestPersistence(ServiceCase):
    def test_persist_stores_model_output_apart_from_measurements(self):
        n_obs = self.db.execute("SELECT COUNT(*) FROM traffic_records").fetchone()[0]
        r = self.svc.predict(self.db, 1, 30, persist=True, now=self.now)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM traffic_predictions").fetchone()[0], 1)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM traffic_records").fetchone()[0], n_obs)   # never mixed in
        row = self.db.execute("SELECT predicted_speed_kmh, model_version, horizon_min FROM traffic_predictions").fetchone()
        self.assertAlmostEqual(row[0], r["predicted_speed_kmh"], places=1)
        self.assertEqual((row[1], row[2]), (r["model"]["version"], 30))

    def test_repeating_the_same_prediction_does_not_duplicate_rows(self):
        for _ in range(3):
            self.svc.predict(self.db, 1, 30, persist=True, now=self.now)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM traffic_predictions").fetchone()[0], 1)

    def test_history_returns_stored_predictions_only(self):
        self.svc.predict(self.db, 1, 30, persist=True, now=self.now)
        hist = self.svc.history(self.db, 1)
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0]["horizon_min"], 30)
        self.assertEqual(self.svc.history(self.db, 2), [])

    def test_generate_all_covers_every_road_and_horizon(self):
        out = self.svc.generate_all(self.db, now=self.now)
        self.assertEqual(out["stored"], 3)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM traffic_predictions").fetchone()[0], 3)

    def test_unavailable_results_are_never_stored(self):
        self.svc.predict(self.db, 1, 45, persist=True, now=self.now)
        self.svc.predict(self.db, 9999, 30, persist=True, now=self.now)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM traffic_predictions").fetchone()[0], 0)


class TestStatus(ServiceCase):
    def test_status_reports_model_and_data_freshness(self):
        s = self.svc.status(self.db, now=self.now)
        self.assertEqual(s["model_status"], "ready"); self.assertEqual(s["horizons"], [30])
        self.assertTrue(s["data"]["fresh"])
        old = self.svc.status(self.db, now=self.now + pd.Timedelta(hours=5))
        self.assertFalse(old["data"]["fresh"])

    def test_status_without_a_model(self):
        s = PredictionService(models_dir=tempfile.mkdtemp()).status(self.db, now=self.now)
        self.assertEqual(s["model_status"], "not_connected"); self.assertEqual(s["models"], {})


if __name__ == "__main__":
    unittest.main()

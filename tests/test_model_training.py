"""Training pipeline, serialization, evaluation metrics and acceptance rules.

All data here is SYNTHETIC (tests/synthetic.py): these tests prove the mechanics work. Nothing below says
anything about how accurate a model would be on real Nairobi traffic.
"""
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from machine_learning import feature_engineering as fe
from machine_learning import model_config as cfg
from machine_learning import model_evaluation as ev
from machine_learning import model_registry as reg
from machine_learning import train_model as tm
from tests.helpers import memory_db, trained_env


def _train(days=10, models_dir=None, now="2026-02-01 00:00:00", force=False, seed=7, **kw):
    db, segs, obs = memory_db(days=days, seed=seed)
    models_dir = models_dir or tempfile.mkdtemp(prefix="mlmodels_")
    res = tm.train_all(db, [30], models_dir, fast=True, now=pd.Timestamp(now), plots=False, force=force,
                       log=lambda *a: None, **kw)
    return res, models_dir, db


class TestTrainingRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = trained_env(days=14, horizons=(30,))
        cls.res = cls.env["result"]
        cls.h = cls.res["horizons"]["30"]

    def test_pipeline_runs_and_deploys_a_model(self):
        self.assertEqual(self.res["status"], "ok")
        self.assertEqual(self.h["status"], "deployed")
        self.assertEqual(reg.deployed_horizons(self.env["models_dir"]), [30])
        self.assertTrue((Path(self.env["models_dir"]) / "training_report.json").exists())

    def test_baselines_and_candidates_are_all_reported(self):
        names = {r["model"] for r in self.h["test"]}
        self.assertTrue({"Baseline: persistence", "Baseline: seasonal mean", "Linear Regression",
                         "Random Forest", "Gradient Boosting"} <= names)
        for r in self.h["test"]:
            for k in ("mae", "rmse", "r2"):
                self.assertTrue(math.isfinite(r[k]), (r["model"], k))

    def test_metrics_are_computed_on_the_test_period_only(self):
        self.assertTrue(all(r["n"] == self.h["rows"]["test"] for r in self.h["test"]))
        self.assertTrue(all(r["n"] == self.h["rows"]["validation"] for r in self.h["validation"]))

    def test_split_dates_are_chronological(self):
        s = self.res["split"]
        order = [pd.Timestamp(s[k]) for k in ("first", "train_end", "validation_end", "last")]
        self.assertEqual(order, sorted(order))
        self.assertEqual(len(set(order)), 4)

    def test_best_model_is_chosen_on_validation_not_test(self):
        best_val = min((r for r in self.h["validation"] if r["kind"] == "model"), key=lambda r: r["mae"])["model"]
        self.assertEqual(self.h["selected"]["regressor"], best_val)

    def test_reported_improvement_matches_the_reported_numbers(self):
        base = min(r["mae"] for r in self.h["test"] if r["kind"] == "baseline")
        best = next(r["mae"] for r in self.h["test"] if r["model"] == self.h["selected"]["regressor"])
        self.assertAlmostEqual(self.h["acceptance"]["improvement_vs_best_baseline"], (base - best) / base, places=9)

    def test_classification_uses_per_class_metrics(self):
        c = self.h["classification"]
        self.assertTrue(c["supported"])
        row = c["test"][0]["details"]
        self.assertEqual(row["labels"], list(cfg.CLASS_LABELS))
        self.assertIn("macro_f1", c["test"][0])
        self.assertEqual(np.array(row["confusion_matrix"]).shape, (3, 3))

    def test_rolling_validation_is_reported(self):
        self.assertGreaterEqual(len(self.h["rolling_validation"]), 1)


class TestInsufficientData(unittest.TestCase):
    def test_empty_database_gives_no_data_status(self):
        import sqlite3
        from machine_learning import data_loader as dl
        db = dl.Database(sqlite3.connect(":memory:")); db.ensure_schema()
        res = tm.train_all(db, [30], tempfile.mkdtemp(), fast=True, plots=False, log=lambda *a: None)
        self.assertEqual(res["status"], "no_data")

    def test_too_few_days_refuses_to_train(self):
        res, models_dir, _ = _train(days=3)
        self.assertEqual(res["status"], "insufficient_data")
        self.assertEqual(reg.deployed_horizons(models_dir), [])

    def test_unsupported_horizon_is_skipped_not_trained(self):
        db, _, _ = memory_db(days=10)
        res = tm.train_all(db, [45], tempfile.mkdtemp(), fast=True, plots=False, log=lambda *a: None)
        self.assertEqual(res["horizons"]["45"]["status"], "skipped")


class TestReproducibility(unittest.TestCase):
    def test_same_data_same_seed_gives_identical_results(self):
        a, _, _ = _train(days=10)
        b, _, _ = _train(days=10)
        ha, hb = a["horizons"]["30"], b["horizons"]["30"]
        self.assertEqual(ha["selected"], hb["selected"])
        for ra, rb in zip(ha["test"], hb["test"]):
            self.assertEqual(ra["model"], rb["model"])
            self.assertAlmostEqual(ra["mae"], rb["mae"], places=9)


class TestAcceptanceRules(unittest.TestCase):
    def test_model_that_does_not_beat_the_baseline_is_not_deployed(self):
        with mock.patch.object(cfg, "MIN_MAE_IMPROVEMENT_VS_PERSISTENCE", 0.99):
            res, models_dir, _ = _train(days=10)
        h = res["horizons"]["30"]
        self.assertEqual(h["status"], "rejected")
        self.assertTrue(h["acceptance"]["reasons"])
        self.assertEqual(reg.deployed_horizons(models_dir), [])
        self.assertTrue(list(Path(models_dir).glob("h30/rejected/*/metrics.json")))
        self.assertEqual(res["status"], "no_model_deployed")

    def test_no_deploy_flag_evaluates_but_does_not_deploy(self):
        res, models_dir, _ = _train(days=10, deploy=False)
        self.assertEqual(res["horizons"]["30"]["status"], "trained_not_deployed")
        self.assertEqual(reg.deployed_horizons(models_dir), [])

    def test_candidate_not_better_than_deployed_model_is_rejected_and_old_one_kept(self):
        res1, models_dir, _ = _train(days=10, now="2026-02-01 00:00:00")
        v1 = res1["horizons"]["30"]["version"]
        self.assertEqual(reg.current_version(30, models_dir), v1)
        res2, _, _ = _train(days=10, models_dir=models_dir, now="2026-02-02 00:00:00")   # identical data -> not better
        h2 = res2["horizons"]["30"]
        self.assertEqual(h2["status"], "rejected")
        self.assertIn("not better than deployed", " ".join(h2["acceptance"]["reasons"]))
        self.assertEqual(reg.current_version(30, models_dir), v1)             # still serving the old one

    def test_forced_deploy_and_rollback_keep_the_previous_version(self):
        res1, models_dir, _ = _train(days=10, now="2026-02-01 00:00:00")
        v1 = res1["horizons"]["30"]["version"]
        res2, _, _ = _train(days=10, models_dir=models_dir, now="2026-02-02 00:00:00", force=True)
        v2 = res2["horizons"]["30"]["version"]
        self.assertNotEqual(v1, v2)
        self.assertEqual(reg.current_version(30, models_dir), v2)
        self.assertEqual(reg.rollback(30, models_dir), v1)
        self.assertEqual(reg.current_version(30, models_dir), v1)
        self.assertEqual(reg.load_bundle(30, models_dir)["version"], v1)


class TestSerialization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = trained_env(days=14, horizons=(30,))

    def test_bundle_contains_everything_needed_for_serving(self):
        b = reg.load_bundle(30, self.env["models_dir"])
        for key in ("version", "trained_at", "horizon", "feature_config", "feature_columns", "regressor", "metrics", "library_versions"):
            self.assertIn(key, b)
        self.assertEqual(b["horizon"], 30)
        self.assertIn("test_mae_kmh", b["metrics"])
        self.assertEqual(b["feature_columns"], fe.feature_columns(b["feature_config"]))

    def test_reloading_gives_identical_predictions(self):
        b1, b2 = reg.load_bundle(30, self.env["models_dir"]), reg.load_bundle(30, self.env["models_dir"])
        rng = np.random.default_rng(0)
        X = pd.DataFrame({c: rng.normal(40, 5, 25) for c in b1["feature_columns"]})
        X["segment_id"] = 1
        np.testing.assert_allclose(b1["regressor"].predict(X), b2["regressor"].predict(X))

    def _copy(self):
        d = Path(tempfile.mkdtemp()) / "models"
        shutil.copytree(self.env["models_dir"], d)
        return d

    def test_tampered_model_file_is_refused(self):
        d = self._copy()
        path = next(d.glob("h30/*/bundle.joblib"))
        path.write_bytes(path.read_bytes() + b"x")
        with self.assertRaises(reg.ModelError):
            reg.load_bundle(30, d)

    def test_version_string_cannot_escape_the_models_directory(self):
        with self.assertRaises(reg.ModelError):
            reg.load_bundle(30, self.env["models_dir"], version="../../../etc")

    def test_missing_model_raises_a_clean_error(self):
        with self.assertRaises(reg.ModelError):
            reg.load_bundle(30, tempfile.mkdtemp())

    def test_manifest_records_a_checksum(self):
        m = json.loads(next(Path(self.env["models_dir"]).glob("h30/*/manifest.json")).read_text())
        self.assertEqual(len(m["sha256"]), 64)

    def test_signing_key_rejects_unsigned_models(self):
        with mock.patch.dict("os.environ", {"ML_MODEL_SIGNING_KEY": "secret-for-test"}):
            with self.assertRaises(reg.ModelError):
                reg.load_bundle(30, self.env["models_dir"])     # model was saved without a signature


class TestMetrics(unittest.TestCase):
    def test_regression_metrics_match_hand_calculation(self):
        m = ev.regression_metrics([10, 20, 30], [12, 18, 33])
        self.assertAlmostEqual(m["mae"], 7 / 3, places=9)
        self.assertAlmostEqual(m["rmse"], math.sqrt(17 / 3), places=9)
        self.assertAlmostEqual(m["r2"], 1 - 17 / 200, places=9)

    def test_classification_metrics_match_hand_calculation(self):
        m = ev.classification_metrics(["low", "low", "moderate", "heavy"], ["low", "moderate", "moderate", "heavy"])
        self.assertAlmostEqual(m["accuracy"], 0.75)
        self.assertAlmostEqual(m["macro_precision"], (1 + 0.5 + 1) / 3)
        self.assertAlmostEqual(m["macro_recall"], (0.5 + 1 + 1) / 3)
        self.assertAlmostEqual(m["macro_f1"], (2 / 3 + 2 / 3 + 1) / 3)
        self.assertEqual(m["confusion_matrix"], [[1, 1, 0], [0, 1, 0], [0, 0, 1]])

    def test_macro_f1_exposes_a_majority_class_predictor_that_accuracy_flatters(self):
        y = ["low"] * 90 + ["moderate"] * 5 + ["heavy"] * 5
        m = ev.classification_metrics(y, ["low"] * 100)
        self.assertAlmostEqual(m["accuracy"], 0.90)
        self.assertLess(m["macro_f1"], 0.40)

    def test_rolling_validation_never_trains_on_the_future(self):
        n = 6000
        ts = pd.Timestamp("2026-01-01") + pd.to_timedelta(np.arange(n) * 5, unit="min")
        sup = pd.DataFrame({"ts": ts, "target_time": ts + pd.Timedelta(minutes=30),
                            "target_speed": np.arange(n) % 50 + 10.0, "speed_now": np.arange(n) % 50 + 11.0})
        seen = []

        def fit_fn(tr, te):
            seen.append((tr["target_time"].max(), te["ts"].min()))
            return te["speed_now"].to_numpy()

        out = ev.rolling_validation(sup, fit_fn, ts.min(), ts.max())
        self.assertGreaterEqual(len(out), 2)
        for last_train_target, first_test_input in seen:
            self.assertLessEqual(last_train_target, first_test_input)      # purged: no label from the test window
        cuts = [pd.Timestamp(o["train_until"]) for o in out]
        self.assertEqual(cuts, sorted(cuts))


if __name__ == "__main__":
    unittest.main()

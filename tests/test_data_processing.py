import os
import sqlite3
import tempfile
import unittest

import numpy as np
import pandas as pd

from machine_learning import data_cleaning as dc
from machine_learning import data_loader as dl
from machine_learning import feature_engineering as fe
from machine_learning import model_evaluation as ev
from tests.helpers import memory_db
from tests.synthetic import make_synthetic_traffic


def _segs(n=2):
    return pd.DataFrame({"id": range(1, n + 1), "name": [f"R{i}" for i in range(1, n + 1)], "latitude": [-1.3] * n, "longitude": [36.8] * n,
                         "length_km": [2.0] * n, "free_flow_speed_kmh": [np.nan] * n, "speed_limit_kmh": [50.0] * n, "geometry_json": [None] * n})


class TestDatabaseExtraction(unittest.TestCase):
    def test_extraction_filters_and_sorts(self):
        db, segs, obs = memory_db(days=2)
        all_rows = dl.load_observations(db)
        self.assertEqual(len(all_rows), len(obs))
        one = dl.load_observations(db, [2], start="2026-01-05 12:00:00", end="2026-01-05 13:00:00")
        self.assertTrue((one["segment_id"] == 2).all())
        self.assertTrue(one["recorded_at"].min() >= "2026-01-05 12:00:00" and one["recorded_at"].max() <= "2026-01-05 13:00:00")
        self.assertTrue(all_rows.groupby("segment_id")["recorded_at"].apply(lambda s: s.is_monotonic_increasing).all())
        self.assertEqual(len(dl.load_observations(db, [])), 0)

    def test_parameterized_sql_resists_injection(self):
        db, _, _ = memory_db(days=1)
        # a hostile "id" must be rejected by int(), never concatenated into SQL
        with self.assertRaises(ValueError):
            dl.load_observations(db, ["1; DROP TABLE traffic_records"])
        self.assertTrue(db.has_table("traffic_records"))

    def test_csv_import_reports_rejects_and_skips_duplicates(self):
        db = dl.Database(sqlite3.connect(":memory:"))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.csv")
            open(p, "w").write("segment_name,recorded_at,speed_kmh,vehicle_count\n"
                               "Thika Rd,2026-01-05 08:00:00,30,10\nThika Rd,2026-01-05 08:00:00,30,10\n"
                               "Thika Rd,not a date,30,10\nThika Rd,2026-01-05 08:05:00,abc,10\nUhuru Hwy,2026-01-05 08:00:00,40,\n")
            rep = dl.import_traffic_csv(db, p, "unit-test", utc_offset_hours=3)
        self.assertEqual(rep["rows_read"], 5)
        self.assertEqual(rep["rows_rejected"], {"invalid_timestamp": 1, "non_numeric_speed": 1})
        self.assertEqual(rep["rows_inserted"], 2); self.assertEqual(rep["duplicates_skipped"], 1); self.assertEqual(rep["segments_created"], 2)
        # naive local times were converted to UTC (3 h earlier)
        self.assertEqual(db.execute("SELECT MIN(recorded_at) FROM traffic_records").fetchone()[0], "2026-01-05 05:00:00")


class TestCleaning(unittest.TestCase):
    def setUp(self):
        t = pd.Timestamp("2026-01-05 08:00:00")
        self.segs = _segs()
        self.obs = pd.DataFrame({
            "segment_id": [1, 1, 1, 1, 1, 9, 2],
            "recorded_at": [t, t + pd.Timedelta(minutes=5), t + pd.Timedelta(minutes=5), "garbage", t + pd.Timedelta(minutes=10), t, t],
            "speed_kmh": [30, 35, 36, 40, 500, 30, -5],
            "vehicle_count": [10, 10, 10, 10, 10, 10, 10], "density": [np.nan] * 7})

    def test_each_problem_is_handled_and_counted(self):
        clean, rep = dc.clean_observations(self.obs, self.segs, now="2026-12-31")
        self.assertEqual(rep["removed"]["invalid_timestamp"], 1)
        self.assertEqual(rep["removed"]["unknown_segment_id"], 1)
        self.assertEqual(rep["removed"]["duplicate_segment_timestamp"], 1)
        self.assertEqual(rep["removed"]["impossible_speed_set_missing"], 2)      # 500 and -5
        self.assertEqual(rep["removed"]["no_usable_speed"], 2)
        self.assertEqual(rep["rows_out"], 2)                                     # (1,08:00)=30 and (1,08:05)=36 (last duplicate kept)
        self.assertEqual(clean.loc[clean["recorded_at"] == pd.Timestamp("2026-01-05 08:05:00"), "speed_kmh"].iloc[0], 36)

    def test_output_is_chronological_per_segment(self):
        db, segs, obs = memory_db(days=2)
        shuffled = dl.load_observations(db).sample(frac=1, random_state=1)
        clean, _ = dc.clean_observations(shuffled, dl.load_segments(db), now="2026-12-31")
        self.assertTrue(clean.groupby("segment_id")["recorded_at"].apply(lambda s: s.is_monotonic_increasing).all())

    def test_unusual_but_possible_values_are_flagged_not_deleted(self):
        segs, obs = make_synthetic_traffic(days=3, n_segments=1)
        obs.loc[obs.index[100], "speed_kmh"] = 2.0          # a real jam can crawl
        obs.loc[obs.index[200], "speed_kmh"] = 115.0
        clean, rep = dc.clean_observations(obs, segs, now="2026-12-31")
        self.assertEqual(len(clean), len(obs))              # nothing removed
        self.assertGreaterEqual(rep["flagged"]["statistical_speed_outliers_kept"], 1)

    def test_missing_values_in_grid(self):
        t0 = pd.Timestamp("2026-01-05 00:00:00")
        times = [t0 + pd.Timedelta(minutes=5 * i) for i in (0, 1, 4, 11, 12)]          # gaps: 2 bins, then 6 bins
        df = pd.DataFrame({"segment_id": 1, "recorded_at": times, "speed_kmh": [30, 31, 32, 33, 34.0], "vehicle_count": np.nan, "density": np.nan})
        g = dc.to_grid(df, 5, 2).set_index("ts")
        self.assertEqual(len(g), 13)
        self.assertTrue(g.loc[t0 + pd.Timedelta(minutes=10), "speed_kmh"] == 31 and not g.loc[t0 + pd.Timedelta(minutes=10), "observed"])  # bridged, not 'observed'
        self.assertTrue(np.isnan(g.loc[t0 + pd.Timedelta(minutes=40), "speed_kmh"]))                                                          # long gap stays empty
        self.assertEqual(int(g["observed"].sum()), 5)


class TestFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db, segs, _ = memory_db(days=14)
        cls.segs = dl.load_segments(db)
        clean, _ = dc.clean_observations(dl.load_observations(db), cls.segs, now="2026-12-31")
        cls.grid = dc.to_grid(clean)
        cls.train_end = cls.grid["ts"].min() + (cls.grid["ts"].max() - cls.grid["ts"].min()) * 0.7
        cls.refs = fe.reference_speeds(cls.grid[cls.grid["ts"] <= cls.train_end], cls.segs)
        cls.inc = pd.DataFrame(columns=["latitude", "longitude", "created_at"])
        cls.fcfg = fe.make_feature_config(cls.grid[cls.grid["ts"] <= cls.train_end], cls.segs, cls.inc, cls.refs)

    def test_lag_features_are_correct(self):
        f = fe.build_features(self.grid, self.fcfg)
        g = self.grid[self.grid["segment_id"] == 1].reset_index(drop=True)
        r = f[f["segment_id"] == 1].reset_index(drop=True)
        i = 200
        self.assertAlmostEqual(r.loc[i, "speed_lag_3"], g.loc[i - 3, "speed_kmh"])
        self.assertAlmostEqual(r.loc[i, "speed_roll_mean_6"], g.loc[i - 5:i, "speed_kmh"].mean())
        self.assertTrue(np.isnan(r.loc[2, "speed_lag_12"]))          # not enough history -> missing, never invented

    def test_no_future_leakage_in_features(self):
        cut = self.grid["ts"].min() + pd.Timedelta(days=9)
        base = fe.build_features(self.grid, self.fcfg)
        tampered = self.grid.copy()
        tampered.loc[tampered["ts"] > cut, ["speed_kmh", "vehicle_count"]] = 999.0          # rewrite the FUTURE
        alt = fe.build_features(tampered, self.fcfg)
        cols = [c for c in fe.feature_columns(self.fcfg) if not c.startswith("target_")]
        past_a, past_b = base[base["ts"] <= cut], alt[alt["ts"] <= cut]
        pd.testing.assert_frame_equal(past_a[cols + ["segment_id", "ts"]].reset_index(drop=True), past_b[cols + ["segment_id", "ts"]].reset_index(drop=True))

    def test_targets_are_measured_future_values(self):
        sup = fe.make_supervised(fe.build_features(self.grid, self.fcfg), self.grid, 30, self.fcfg)
        lookup = self.grid.set_index(["segment_id", "ts"])
        for r in sup.sample(50, random_state=0).itertuples():
            row = lookup.loc[(r.segment_id, r.ts + pd.Timedelta(minutes=30))]
            self.assertTrue(row["observed"])                                                  # a real measurement, not a carried-forward value
            self.assertAlmostEqual(row["speed_kmh"], r.target_speed)
        self.assertTrue((sup["target_time"] - sup["ts"] == pd.Timedelta(minutes=30)).all())
        with self.assertRaises(ValueError):
            fe.make_supervised(fe.build_features(self.grid, self.fcfg), self.grid, 7, self.fcfg)   # not a multiple of the grid

    def test_reference_speed_uses_training_period_only(self):
        tampered = self.grid.copy()
        tampered.loc[tampered["ts"] > self.train_end, "speed_kmh"] = 150.0
        self.assertEqual(fe.reference_speeds(tampered[tampered["ts"] <= self.train_end], self.segs), self.refs)

    def test_optional_features_only_when_supported(self):
        self.assertFalse(self.fcfg["use_density"])           # no density in the data
        self.assertFalse(self.fcfg["use_incidents"])         # no incidents supplied
        self.assertFalse(self.fcfg["use_month"])             # single month of data
        self.assertTrue(self.fcfg["use_vehicle_count"])
        self.assertNotIn("density_now", fe.feature_columns(self.fcfg))

    def test_chronological_split_has_no_overlap(self):
        sup = fe.make_supervised(fe.build_features(self.grid, self.fcfg), self.grid, 30, self.fcfg)
        t_min, t_max = self.grid["ts"].min(), self.grid["ts"].max()
        tr_end, va_end = ev.time_cutoffs(t_min, t_max)
        train, val, test = ev.chronological_split(sup, tr_end, va_end)
        self.assertLessEqual(train["target_time"].max(), tr_end)             # training labels all lie in the training period
        self.assertGreater(val["ts"].min(), tr_end); self.assertLessEqual(val["target_time"].max(), va_end)
        self.assertGreater(test["ts"].min(), va_end)
        self.assertLess(len(train), len(sup)); self.assertGreater(len(test), 0)


if __name__ == "__main__":
    unittest.main()

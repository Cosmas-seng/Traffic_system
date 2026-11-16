"""Route ranking cost rules (services/routing_service.py). Pure functions: no model or database needed.

The numbers below are hand-made inputs chosen so the correct answer can be worked out on paper.
"""
import sqlite3
import unittest

from machine_learning import data_loader as dl
from services import routing_service as rs


def route(id_, dist, base, segs=()):
    return {"id": id_, "name": id_, "distance_km": dist, "base_duration_min": base,
            "segments": [{"segment_id": s, "length_km": k} for s, k in segs]}


class TestRanking(unittest.TestCase):
    def test_time_is_length_over_speed(self):
        out = rs.rank_routes([route("a", 6, 12, [(1, 6)])], {1: (30.0, "measured")})
        self.assertAlmostEqual(out[0]["duration_min"], 12.0)             # 6 km at 30 km/h = 12 min
        self.assertEqual(out[0]["traffic_coverage"], 1.0)

    def test_shortest_distance_is_not_assumed_fastest(self):
        short_but_jammed = route("short", 5, 7, [(1, 5)])            # 5 km at 10 km/h = 30 min
        long_but_clear = route("long", 8, 10, [(2, 8)])              # 8 km at 60 km/h = 8 min
        out = rs.rank_routes([short_but_jammed, long_but_clear], {1: (10.0, "predicted"), 2: (60.0, "predicted")})
        self.assertEqual([r["id"] for r in out], ["long", "short"])
        self.assertTrue(out[0]["recommended"]); self.assertFalse(out[1]["recommended"])

    def test_unmonitored_stretches_use_the_engines_free_flow_time(self):
        out = rs.rank_routes([route("none", 10, 20)], {})
        self.assertAlmostEqual(out[0]["duration_min"], 20.0)
        self.assertEqual(out[0]["traffic_coverage"], 0.0)
        self.assertIsNone(out[0]["level"])                               # no traffic data -> no congestion claim

    def test_partly_monitored_route_mixes_both_sources_pro_rata(self):
        # 10 km route, base 20 min. 4 km monitored at 20 km/h = 12 min. Other 6 km: 20 * 6/10 = 12 min.
        out = rs.rank_routes([route("mix", 10, 20, [(1, 4)])], {1: (20.0, "measured")})
        self.assertAlmostEqual(out[0]["duration_min"], 24.0)
        self.assertAlmostEqual(out[0]["traffic_coverage"], 0.4)

    def test_each_reported_incident_adds_the_documented_penalty(self):
        base = rs.rank_routes([route("a", 6, 12, [(1, 6)])], {1: (30.0, "measured")})[0]["duration_min"]
        with_inc = rs.rank_routes([route("a", 6, 12, [(1, 6)])], {1: (30.0, "measured")}, {1: 3})[0]
        self.assertAlmostEqual(with_inc["duration_min"], base + 3 * rs.INCIDENT_PENALTY_MIN)
        self.assertEqual(with_inc["incidents_considered"], 3)

    def test_speed_basis_is_reported(self):
        out = rs.rank_routes([route("a", 6, 12, [(1, 3), (2, 3)])], {1: (30.0, "measured"), 2: (30.0, "predicted")})
        self.assertEqual(out[0]["speed_basis"], ["measured", "predicted"])

    def test_invalid_candidates_are_skipped_not_guessed(self):
        out = rs.rank_routes([route("zero", 0, 5), route("ok", 4, 8)], {})
        self.assertEqual([r["id"] for r in out], ["ok"])

    def test_no_candidates_means_no_routes(self):
        self.assertEqual(rs.rank_routes([], {}), [])
        self.assertEqual(rs.rank_routes(None, {}), [])

    def test_segment_length_is_capped_at_route_distance(self):
        out = rs.rank_routes([route("cap", 2, 4, [(1, 50)])], {1: (60.0, "measured")})
        self.assertAlmostEqual(out[0]["duration_min"], 2.0)             # 2 km at 60 km/h, not 50 km


class TestSpeedLookup(unittest.TestCase):
    def test_prediction_beats_measurement_and_stale_data_is_ignored(self):
        cur = [{"id": 1, "fresh": True, "speed_kmh": 40}, {"id": 2, "fresh": False, "speed_kmh": 55}, {"id": 3, "fresh": True, "speed_kmh": None}]
        out = rs.speed_lookup(cur, {1: 25.0})
        self.assertEqual(out[1], (25.0, "predicted"))
        self.assertNotIn(2, out)                                       # stale
        self.assertNotIn(3, out)                                       # no speed
        self.assertEqual(rs.speed_lookup(cur)[1], (40.0, "measured"))

    def test_non_positive_predictions_are_ignored(self):
        self.assertEqual(rs.speed_lookup([], {1: 0, 2: -5}), {})


class TestIncidentCounts(unittest.TestCase):
    def test_only_recent_nearby_incidents_count(self):
        import pandas as pd
        db = dl.Database(sqlite3.connect(":memory:")); db.ensure_schema()
        db.execute("INSERT INTO road_segments(id,name,latitude,longitude) VALUES(1,'A',-1.2864,36.8172)")
        now = pd.Timestamp.now("UTC").tz_localize(None)
        fmt = lambda m: (now - pd.Timedelta(minutes=m)).strftime("%Y-%m-%d %H:%M:%S")
        db.execute("INSERT INTO incidents(latitude,longitude,created_at) VALUES(-1.2870,36.8175,?)", (fmt(10),))     # near, recent
        db.execute("INSERT INTO incidents(latitude,longitude,created_at) VALUES(-1.2870,36.8175,?)", (fmt(600),))    # near, too old
        db.execute("INSERT INTO incidents(latitude,longitude,created_at) VALUES(-1.4000,36.9500,?)", (fmt(10),))     # recent, far away
        db.commit()
        self.assertEqual(rs.incident_counts(db, [1]), {1: 1})
        self.assertEqual(rs.incident_counts(db, []), {})


if __name__ == "__main__":
    unittest.main()

"""SYNTHETIC TRAFFIC DATA - FOR AUTOMATED TESTS ONLY.

This is NOT Nairobi traffic. It exists so the pipeline's mechanics (cleaning, features, no-leakage,
split, save/load, API) can be tested without real data. Nothing trained on it is ever deployed, and no
accuracy figure obtained from it says anything about real-world performance.
"""
import numpy as np
import pandas as pd

from machine_learning.data_loader import Database, TS_FMT


def make_synthetic_traffic(days=28, n_segments=3, seed=7, start="2026-01-05 00:00:00", interval_min=5,
                           missing_frac=0.02, with_counts=True):
    rng = np.random.default_rng(seed)
    segs = pd.DataFrame({
        "id": range(1, n_segments + 1), "name": [f"Test Road {i}" for i in range(1, n_segments + 1)],
        "latitude": [-1.28 - 0.01 * i for i in range(n_segments)], "longitude": [36.82 + 0.01 * i for i in range(n_segments)],
        "length_km": [3.0 + i for i in range(n_segments)], "free_flow_speed_kmh": [np.nan] * n_segments,
        "speed_limit_kmh": [50.0] * n_segments, "geometry_json": [None] * n_segments})
    times = pd.date_range(start, periods=int(
        days * 24 * 60 / interval_min), freq=f"{interval_min}min")  # UTC
    local_hour = ((times + pd.Timedelta(hours=3)).hour +
                  (times + pd.Timedelta(hours=3)).minute / 60).to_numpy()
    weekday = ((times + pd.Timedelta(hours=3)).day_of_week < 5).astype(float)
    frames = []
    for i in range(n_segments):
        base = 45 + 5 * i
        dip = (np.exp(-((local_hour - 8) ** 2) / 2.0) +
               np.exp(-((local_hour - 17.5) ** 2) / 3.0)) * weekday * (0.5 + 0.1 * i)
        noise = np.zeros(len(times))
        e = rng.normal(0, 2.2, len(times))
        # AR(1): traffic conditions persist
        for k in range(1, len(times)):
            noise[k] = 0.92 * noise[k - 1] + e[k]
        speed = np.clip(base * (1 - dip) + noise, 3, 120)
        df = pd.DataFrame(
            {"segment_id": i + 1, "recorded_at": times, "speed_kmh": speed})
        if with_counts:
            df["vehicle_count"] = np.clip(
                60 + 200 * (1 - speed / base) + rng.normal(0, 8, len(df)), 0, None)
        else:
            df["vehicle_count"] = np.nan
        df["density"] = np.nan
        frames.append(df[rng.random(len(df)) > missing_frac])
    return segs, pd.concat(frames, ignore_index=True)


def populate_db(db: Database, segs, obs, source="synthetic-test"):
    db.ensure_schema()
    for r in segs.itertuples():
        db.execute("INSERT INTO road_segments(id,name,latitude,longitude,length_km,free_flow_speed_kmh,speed_limit_kmh) VALUES(?,?,?,?,?,?,?)",
                   (int(r.id), r.name, r.latitude, r.longitude, r.length_km, None if pd.isna(r.free_flow_speed_kmh) else r.free_flow_speed_kmh, r.speed_limit_kmh))
    rows = [(int(r.segment_id), pd.Timestamp(r.recorded_at).strftime(TS_FMT), float(r.speed_kmh),
             None if pd.isna(r.vehicle_count) else float(r.vehicle_count), None if pd.isna(r.density) else float(r.density), source)
            for r in obs.itertuples()]
    db.conn.cursor().executemany(db.q(
        "INSERT INTO traffic_records(segment_id,recorded_at,speed_kmh,vehicle_count,density,source) VALUES(?,?,?,?,?,?)"), rows)
    db.commit()


import json

import pandas as pd

from machine_learning import model_config as cfg
from machine_learning import model_registry as reg
from machine_learning.congestion_classifier import label_ratio

LATEST_SQL = ("SELECT r.segment_id, r.recorded_at, r.speed_kmh FROM traffic_records r "
              "JOIN (SELECT segment_id, MAX(recorded_at) AS m FROM traffic_records GROUP BY segment_id) x "
              "ON r.segment_id = x.segment_id AND r.recorded_at = x.m")


def _reference_from_models(models_dir=None):
    """Reference speeds recorded in the deployed models (if any)."""
    for h in reg.deployed_horizons(models_dir):
        try:
            return {int(k): v for k, v in reg.load_bundle(h, models_dir)["feature_config"]["ref_speeds"].items()}
        except reg.ModelError:
            continue
    return {}


def _coords(geometry_json):
    try:
        pts = json.loads(geometry_json) if geometry_json else None
        return pts if isinstance(pts, list) and len(pts) >= 2 else None
    except (TypeError, ValueError):
        return None


def current_conditions(db, now=None, models_dir=None):
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(
        "UTC").tz_localize(None)
    empty = {"available": False, "is_live": False, "summary": None, "segments": [], "updated_at": now.strftime("%Y-%m-%d %H:%M:%S") + " UTC",
             "last_observation": None, "message": "No traffic measurements are stored yet. Import a real dataset or connect a traffic data provider."}
    if not (db.has_table("traffic_records") and db.has_table("road_segments")):
        return empty
    latest = db.frame(LATEST_SQL)
    if latest.empty:
        return empty
    segs = db.frame(
        "SELECT id, name, length_km, free_flow_speed_kmh, geometry_json FROM road_segments").set_index("id")
    model_refs = _reference_from_models(models_dir)
    items, fresh_levels, delays, newest = [], [], [], None
    for r in latest.itertuples():
        if r.segment_id not in segs.index:
            continue
        s = segs.loc[r.segment_id]
        ts = pd.Timestamp(r.recorded_at)
        newest = ts if newest is None or ts > newest else newest
        age = (now - ts).total_seconds() / 60
        ref = float(s["free_flow_speed_kmh"]) if pd.notna(
            s["free_flow_speed_kmh"]) else model_refs.get(int(r.segment_id))
        fresh = age <= cfg.MAX_OBSERVATION_AGE_MIN
        level = str(label_ratio(r.speed_kmh / ref)
                    ) if (ref and fresh) else None
        if level:
            fresh_levels.append(level)
            if pd.notna(s["length_km"]) and r.speed_kmh > 0:
                delays.append(max(0.0, float(
                    s["length_km"]) / r.speed_kmh * 60 - float(s["length_km"]) / ref * 60))
        items.append({"id": int(r.segment_id), "name": s["name"], "level": level, "speed_kmh": round(float(r.speed_kmh), 1),
                      "reference_speed_kmh": None if not ref else round(ref, 1), "observed_at": ts.strftime("%Y-%m-%d %H:%M:%S"),
                      "age_minutes": round(age, 1), "fresh": bool(fresh), "coordinates": _coords(s["geometry_json"])})
    live = bool(fresh_levels) or any(i["fresh"] for i in items)
    out = {"available": live, "is_live": live, "segments": items, "updated_at": now.strftime("%Y-%m-%d %H:%M:%S") + " UTC",
           "last_observation": newest.strftime("%Y-%m-%d %H:%M:%S") if newest is not None else None}
    if live:
        out["summary"] = {"heavy": fresh_levels.count("heavy"), "moderate": fresh_levels.count("moderate"), "low": fresh_levels.count("low"),
                          "avg_delay_min": round(sum(delays) / len(delays), 1) if delays else None}
        out["message"] = None if fresh_levels else ("Fresh measurements exist but no reference speed is known, so congestion levels cannot be "
                                                    "shown. Set free_flow_speed_kmh on the roads or deploy a trained model.")
    else:
        out["summary"] = None
        out["message"] = (f"Newest stored measurement is from {out['last_observation']} UTC. No live traffic feed is connected, so current "
                          "conditions are not shown (historical data is used for model training only).")
    return out


def trend_report(db, start, end):
    """Aggregates of MEASURED speeds for [start, end) ('YYYY-MM-DD'). Done in SQL so large tables are not loaded."""
    off = cfg.LOCAL_UTC_OFFSET_HOURS
    if not (db.has_table("traffic_records") and db.has_table("road_segments")):
        return {"available": False, "message": "No traffic data tables exist yet."}
    n = db.execute(
        "SELECT COUNT(*) FROM traffic_records WHERE recorded_at>=? AND recorded_at<?", (start, end)).fetchone()[0]
    if not n:
        return {"available": False, "message": "No traffic measurements exist for this date range. Import a real dataset or connect a provider."}
    # fixed SQL fragments, no user input
    day = "substr(recorded_at,1,10)" if db.engine == "sqlite" else "DATE(recorded_at)"
    hour = "CAST(substr(recorded_at,12,2) AS INTEGER)" if db.engine == "sqlite" else "HOUR(recorded_at)"
    by_day = db.frame(
        f"SELECT {day} AS d, AVG(speed_kmh) AS v, COUNT(*) AS n FROM traffic_records WHERE recorded_at>=? AND recorded_at<? GROUP BY d ORDER BY d", (start, end))
    by_hour = db.frame(
        f"SELECT {hour} AS h, AVG(speed_kmh) AS v, COUNT(*) AS n FROM traffic_records WHERE recorded_at>=? AND recorded_at<? GROUP BY h ORDER BY h", (start, end))
    roads = db.frame("SELECT r.segment_id AS id, s.name AS name, AVG(r.speed_kmh) AS v, COUNT(*) AS n, s.free_flow_speed_kmh AS ff "
                     "FROM traffic_records r JOIN road_segments s ON s.id = r.segment_id WHERE r.recorded_at>=? AND r.recorded_at<? "
                     "GROUP BY r.segment_id, s.name, s.free_flow_speed_kmh", (start, end))
    refs, road_rows = _reference_from_models(), []
    for r in roads.itertuples():
        ref = float(r.ff) if pd.notna(r.ff) else refs.get(int(r.id))
        road_rows.append({"road": r.name, "avg_speed_kmh": round(float(r.v), 1), "n": int(r.n),
                          "speed_ratio": round(float(r.v) / ref, 2) if ref else None})
    rated = [r for r in road_rows if r["speed_ratio"] is not None]
    return {"available": True, "records": int(n),
            "by_day": [{"date": r.d, "avg_speed_kmh": round(float(r.v), 1)} for r in by_day.itertuples()],
            "by_hour": sorted(({"hour": (int(r.h) + off) % 24, "avg_speed_kmh": round(float(r.v), 1), "n": int(r.n)} for r in by_hour.itertuples()),
                              key=lambda x: x["hour"]),
            "roads": sorted(road_rows, key=lambda r: r["avg_speed_kmh"]),
            "most_congested": min(rated, key=lambda r: r["speed_ratio"])["road"] if rated else None,
            "note": None if rated else "Congestion ranking needs a reference speed per road (free_flow_speed_kmh or a trained model)."}

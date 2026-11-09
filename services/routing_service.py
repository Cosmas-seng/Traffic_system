
from machine_learning.congestion_classifier import label_ratio

INCIDENT_PENALTY_MIN = 2.0


def speed_lookup(current_segments, predictions=None):
    """{segment_id: (speed_kmh, source)}. `current_segments` as returned by traffic_service; `predictions`
    an optional {segment_id: predicted_speed_kmh} from the prediction service."""
    out = {}
    for s in current_segments or []:
        if s.get("fresh") and s.get("speed_kmh"):
            out[int(s["id"])] = (float(s["speed_kmh"]), "measured")
    for sid, v in (predictions or {}).items():
        if v and v > 0:
            out[int(sid)] = (float(v), "predicted")
    return out


def rank_routes(candidates, speeds, incident_counts=None, penalty=INCIDENT_PENALTY_MIN):
    """candidates: [{id, name, distance_km, base_duration_min, segments:[{segment_id, length_km}], geometry?}]
    Returns routes sorted best-first in the shape the frontend expects."""
    ranked = []
    for c in candidates or []:
        dist, base = float(c["distance_km"]), float(c["base_duration_min"])
        if dist <= 0 or base <= 0:
            continue
        priced_km = time_min = 0.0
        sources = set()
        for seg in c.get("segments", []):
            sp = speeds.get(int(seg["segment_id"]))
            if sp and sp[0] > 0:
                km = min(float(seg["length_km"]), dist - priced_km)
                if km <= 0:
                    continue
                time_min += km / sp[0] * 60
                priced_km += km
                sources.add(sp[1])
        time_min += base * max(0.0, dist - priced_km) / \
            dist                       # rule 2
        incidents = sum((incident_counts or {}).get(
            int(seg["segment_id"]), 0) for seg in c.get("segments", []))
        total = time_min + penalty * incidents                                       # rule 3
        coverage = priced_km / dist
        level = str(label_ratio(base / total)) if coverage > 0 else None
        ranked.append({"id": c["id"], "name": c.get("name"), "duration_min": round(total, 1), "distance_km": round(dist, 1),
                       "level": level, "geometry": c.get("geometry"), "incidents_considered": int(incidents),
                       "traffic_coverage": round(coverage, 2), "speed_basis": sorted(sources)})
    ranked.sort(key=lambda r: r["duration_min"])
    for i, r in enumerate(ranked):
        r["recommended"] = i == 0
    return ranked


def incident_counts(db, segment_ids, hours=2, radius_km=1.5):
    """Reported incidents near each segment's representative point during the last `hours` hours."""
    import pandas as pd
    from machine_learning import feature_engineering as fe
    if not segment_ids or not db.has_table("incidents"):
        return {}
    since = (pd.Timestamp.now("UTC").tz_localize(None) -
             pd.Timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    inc = db.frame(
        "SELECT latitude, longitude FROM incidents WHERE created_at>=?", (since,))
    ids = [int(i) for i in segment_ids]
    segs = db.frame("SELECT id, latitude, longitude FROM road_segments WHERE id IN (" +
                    ",".join("?" * len(ids)) + ")", ids)
    out = {}
    for s in segs.itertuples():
        if inc.empty or pd.isna(s.latitude) or pd.isna(s.longitude):
            continue
        d = fe._haversine_km(float(s.latitude), float(
            s.longitude), inc["latitude"].astype(float), inc["longitude"].astype(float))
        out[int(s.id)] = int((d <= radius_km).sum())
    return out

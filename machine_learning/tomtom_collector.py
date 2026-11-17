"""Collect licensed TomTom Flow Segment Data for configured road segments.

Run: python -m machine_learning.tomtom_collector --limit 5
The collector samples each segment at its midpoint and writes measured speeds only.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from . import data_loader as dl

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=False)
FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/{zoom}/json"


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def fetch_flow(session: requests.Session, api_key: str, lat: float, lon: float,
               zoom: int = 18, timeout: int = 20) -> dict:
    response = session.get(
        FLOW_URL.format(zoom=zoom),
        params={"key": api_key, "point": f"{lat:.7f},{lon:.7f}", "unit": "KMPH"},
        timeout=timeout,
    )
    if response.status_code == 429:
        raise RuntimeError("TomTom rate limit reached (HTTP 429). Reduce collection frequency or road count.")
    if response.status_code in (401, 403):
        raise RuntimeError("TomTom rejected the API key or product access (HTTP 401/403). Check your TomTom account.")
    response.raise_for_status()
    payload = response.json()
    flow = payload.get("flowSegmentData")
    if not isinstance(flow, dict):
        raise RuntimeError("TomTom response did not contain flowSegmentData for this road point.")
    speed = flow.get("currentSpeed")
    if speed is None:
        raise RuntimeError("TomTom response did not include currentSpeed.")
    return flow


def collect(limit: int | None = None, zoom: int | None = None) -> dict:
    api_key = os.getenv("TOMTOM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TOMTOM_API_KEY is missing from your .env file.")
    if not _truthy(os.getenv("TOMTOM_STORAGE_LICENSE_CONFIRMED")):
        raise RuntimeError(
            "Collection is disabled until you confirm that your TomTom plan/license permits "
            "storing these traffic observations. After confirming with TomTom, set "
            "TOMTOM_STORAGE_LICENSE_CONFIRMED=True in .env."
        )
    zoom = zoom or int(os.getenv("TOMTOM_FLOW_ZOOM", "18"))
    db = dl.Database.from_env()
    result = {"provider": "TomTom Flow Segment Data", "checked": 0, "stored": 0, "skipped": 0, "errors": []}
    session = requests.Session()
    try:
        segments = dl.load_segments(db)
        if limit:
            segments = segments.head(limit)
        for seg in segments.itertuples(index=False):
            result["checked"] += 1
            sid = int(seg.id)
            # Existing schema stores endpoints, not a road polyline. Sample the midpoint.
            lat = float(seg.latitude)
            lon = float(seg.longitude)
            if hasattr(seg, "end_lat") and hasattr(seg, "end_lng"):
                lat = (lat + float(seg.end_lat)) / 2
                lon = (lon + float(seg.end_lng)) / 2
            try:
                flow = fetch_flow(session, api_key, lat, lon, zoom=zoom)
                speed = float(flow["currentSpeed"])
                if speed < 0 or speed > 250:
                    raise ValueError(f"TomTom returned an implausible speed: {speed}")
                recorded_at = datetime.now(timezone.utc).replace(tzinfo=None).strftime(dl.TS_FMT)
                level = dl.classify_traffic(speed)
                if db.engine == "mysql":
                    db.execute(
                        "INSERT INTO traffic_records(segment_id,recorded_at,traffic_level,vehicle_count,average_speed,data_source,density) "
                        "VALUES(?,?,?,?,?,?,NULL)",
                        (sid, recorded_at, level, None, speed, "TomTom Traffic API"),
                    )
                else:
                    db.execute(
                        "INSERT INTO traffic_records(segment_id,recorded_at,speed_kmh,vehicle_count,density,source) "
                        "VALUES(?,?,?,?,NULL,?)",
                        (sid, recorded_at, speed, None, "TomTom Traffic API"),
                    )
                db.commit()
                result["stored"] += 1
            except Exception as exc:
                db.rollback()
                result["skipped"] += 1
                result["errors"].append({"segment_id": sid, "road": str(seg.name), "error": str(exc)})
        return result
    finally:
        session.close()
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect TomTom traffic speed for configured road segments")
    parser.add_argument("--limit", type=int, default=None, help="Only sample the first N road segments")
    parser.add_argument("--zoom", type=int, default=None, help="TomTom zoom (0-22; default from TOMTOM_FLOW_ZOOM or 18)")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.zoom is not None and not 0 <= args.zoom <= 22:
        parser.error("--zoom must be between 0 and 22")
    try:
        report = collect(args.limit, args.zoom)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Checked {report['checked']} road(s); stored {report['stored']}; skipped {report['skipped']}.")
    for item in report["errors"]:
        print(f"  Segment {item['segment_id']} ({item['road']}): {item['error']}")
    return 0 if report["stored"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

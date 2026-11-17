"""Presentation-safe prediction fallback.

Uses the real trained PredictionService whenever a deployed model exists.
When no model is deployed, provides clearly labelled heuristic estimates for
software demonstrations. These estimates are NOT trained ML predictions and
must not be represented as validated forecasts.
"""
from __future__ import annotations

import os

import pandas as pd

from . import data_loader as dl
from . import model_config as cfg
from .prediction_service import PredictionService


class PresentationPredictionService:
    def __init__(self, real_service=None):
        self.real = real_service or PredictionService()
        self.models_dir = self.real.models_dir

    @staticmethod
    def demo_enabled():
        return os.getenv("PREDICTION_DEMO_MODE", "False").strip().lower() in {"1", "true", "yes", "on"}

    def horizons(self):
        deployed = self.real.horizons()
        if deployed:
            return deployed
        return list(cfg.SUPPORTED_HORIZONS) if self.demo_enabled() else []

    def catalogue(self, db):
        locations = self.real.catalogue(db)
        if locations:
            return locations
        try:
            segments = dl.load_segments(db)
            return [{"id": int(r.id), "name": str(r.name)}
                    for r in segments.itertuples()]
        except Exception:
            return []

    def status(self, db, now=None):
        real_status = self.real.status(db, now=now)
        if real_status.get("horizons") or not self.demo_enabled():
            return real_status
        now = pd.Timestamp(now) if now is not None else pd.Timestamp.now("UTC").tz_localize(None)
        latest = None
        if db.has_table("traffic_records"):
            row = db.execute("SELECT MAX(recorded_at) FROM traffic_records").fetchone()
            latest = row[0] if row else None
        age = None if latest is None else round((now - pd.Timestamp(latest)).total_seconds() / 60, 1)
        return {
            "model_status": "demo",
            "prediction_mode": "presentation_demo",
            "is_demo": True,
            "models": {},
            "horizons": list(cfg.SUPPORTED_HORIZONS),
            "data": {
                "latest_observation": latest,
                "age_minutes": age,
                "fresh": bool(age is not None and 0 <= age <= cfg.MAX_OBSERVATION_AGE_MIN),
            },
            "message": "Presentation demo estimates are enabled. No trained model is deployed; results are heuristic and not validated ML forecasts.",
        }

    @staticmethod
    def _level(speed):
        return dl.classify_traffic(max(0.0, float(speed)))

    def predict(self, db, segment_id, horizon, persist=False, now=None):
        deployed = self.real.horizons()
        if deployed or not self.demo_enabled():
            return self.real.predict(db, segment_id, horizon, persist=persist, now=now)

        try:
            sid, horizon = int(segment_id), int(horizon)
        except (TypeError, ValueError):
            return {"available": False, "prediction_status": "unavailable", "reason": "invalid_input",
                    "message": "Road and horizon must be valid integers."}
        if horizon not in cfg.SUPPORTED_HORIZONS:
            return {"available": False, "prediction_status": "unavailable", "reason": "unsupported_horizon",
                    "message": "Choose 15, 30 or 60 minutes."}

        if db.engine == "sqlite":
            seg = db.frame("SELECT id, name, length_km FROM road_segments WHERE id=?", (sid,))
        else:
            seg = db.frame("SELECT segment_id AS id, road_name AS name, length_km FROM road_segments WHERE segment_id=?", (sid,))
        if seg.empty:
            return {"available": False, "prediction_status": "unavailable", "reason": "unknown_segment",
                    "message": "The selected road segment was not found."}

        now_ts = pd.Timestamp(now) if now is not None else pd.Timestamp.now("UTC").tz_localize(None)
        observations = dl.load_observations(db, [sid])
        observations = observations.dropna(subset=["speed_kmh"]).sort_values("recorded_at")
        has_observation = not observations.empty
        if has_observation:
            latest = observations.iloc[-1]
            current_speed = max(0.0, float(latest["speed_kmh"]))
            observed_at = pd.Timestamp(latest["recorded_at"])
            age = max(0.0, (now_ts - observed_at).total_seconds() / 60)
            recent = observations.tail(3).copy()
            trend_per_min = 0.0
            if len(recent) >= 2:
                recent["minute"] = pd.to_datetime(recent["recorded_at"]).map(lambda x: pd.Timestamp(x).timestamp() / 60)
                elapsed = float(recent["minute"].iloc[-1] - recent["minute"].iloc[0])
                if elapsed > 0:
                    trend_per_min = (float(recent["speed_kmh"].iloc[-1]) - float(recent["speed_kmh"].iloc[0])) / elapsed
            # Damp the observed trend heavily; this is a transparent demo heuristic.
            projected_speed = current_speed + max(-0.15, min(0.15, trend_per_min)) * horizon * 0.35
        else:
            # Explicit synthetic fallback for an empty road history.
            current_speed = 35.0
            projected_speed = 35.0
            observed_at = None
            age = None

        projected_speed = round(max(5.0, min(100.0, projected_speed)), 1)
        current_level = self._level(current_speed)
        predicted_level = self._level(projected_speed)
        length = seg.iloc[0]["length_km"]
        delay = None
        if pd.notna(length) and float(length) > 0 and projected_speed > 0:
            # A fixed 60 km/h comparison is a demo reference, not a measured free-flow speed.
            delay = round(max(0.0, float(length) / projected_speed * 60 - float(length) / 60 * 60), 1)

        history = []
        if has_observation:
            for row in observations.tail(12).itertuples():
                history.append({"t": pd.Timestamp(row.recorded_at).strftime(dl.TS_FMT),
                                "value": round(float(row.speed_kmh), 1)})
        target = now_ts + pd.Timedelta(minutes=horizon)
        target_text = target.strftime(dl.TS_FMT)
        now_text = now_ts.strftime(dl.TS_FMT)
        return {
            "available": True,
            "prediction_status": "demo",
            "prediction_mode": "presentation_demo",
            "is_demo": True,
            "reason": None,
            "message": "DEMO ESTIMATE: heuristic projection, not a trained or validated machine-learning forecast.",
            "model_status": "demo",
            "road_segment_id": sid,
            "road_name": str(seg.iloc[0]["name"]),
            "prediction_horizon_minutes": horizon,
            "current_speed_kmh": round(current_speed, 1),
            "current_congestion": current_level,
            "current_level": current_level,
            "predicted_speed_kmh": projected_speed,
            "predicted_congestion": predicted_level,
            "predicted_level": predicted_level,
            "estimated_delay_minutes": delay,
            "estimated_delay_min": delay,
            "reference_speed_kmh": 60.0,
            "observed_at": observed_at.strftime(dl.TS_FMT) if observed_at is not None else None,
            "prediction_timestamp": target_text,
            "generated_at": now_text,
            "data_age_minutes": round(age, 1) if age is not None else None,
            "data_warning": ("Using an old traffic observation; treat this as a demonstration only."
                             if age is not None and age > cfg.MAX_OBSERVATION_AGE_MIN else
                             "No recorded observation exists; a demonstration baseline is being used."
                             if not has_observation else None),
            "model": {"name": "Presentation heuristic (not trained ML)", "version": "DEMO",
                      "trained_at": None, "congestion_from": "speed thresholds"},
            "history": history,
            "forecast": [{"t": target_text, "value": projected_speed}],
            "persisted": False,
        }

    def history(self, db, segment_id, limit=200):
        # Demo estimates are intentionally not inserted into the production model_predictions table.
        if self.real.horizons() or not self.demo_enabled():
            return self.real.history(db, segment_id, limit)
        return []

    def generate_all(self, db, now=None):
        if self.real.horizons() or not self.demo_enabled():
            return self.real.generate_all(db, now=now)
        result = {"mode": "presentation_demo", "stored": 0, "generated": 0, "unavailable": {}}
        for road in self.catalogue(db):
            for horizon in cfg.SUPPORTED_HORIZONS:
                prediction = self.predict(db, road["id"], horizon, persist=False, now=now)
                if prediction.get("available"):
                    result["generated"] += 1
                else:
                    reason = prediction.get("reason", "unknown")
                    result["unavailable"][reason] = result["unavailable"].get(reason, 0) + 1
        result["message"] = "Demo estimates generated in memory only; no demo rows were saved to traffic_predictions."
        return result

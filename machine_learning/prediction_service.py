
import threading

import pandas as pd

from . import data_cleaning as dc
from . import data_loader as dl
from . import feature_engineering as fe
from . import model_config as cfg
from . import model_registry as reg
from .congestion_classifier import label_ratio

REASONS = {
    "no_model": "No trained model is deployed yet.",
    "unsupported_horizon": "That prediction horizon is not available.",
    "unknown_segment": "Unknown road segment.",
    "segment_not_in_model": "This road was not part of the data the model was trained on.",
    "no_data": "No traffic measurements exist for this road.",
    "stale_data": "The newest measurement is too old for a reliable short-term prediction.",
    "insufficient_data": "Not enough consecutive recent measurements to build the model inputs.",
    "model_error": "The prediction model could not be loaded.",
    "invalid_input": "Invalid request.",
}


def _fmt(ts):
    return pd.Timestamp(ts).strftime(dl.TS_FMT)


def _unavailable(reason, segment_id=None, horizon=None, extra=None, message=None):
    out = {"available": False, "prediction_status": "unavailable", "reason": reason,
           "message": message or REASONS.get(reason, "Prediction unavailable."),
           "road_segment_id": segment_id, "prediction_horizon_minutes": horizon, "model_status": "ready" if reason not in ("no_model", "model_error") else ("not_connected" if reason == "no_model" else "error")}
    out.update(extra or {})
    return out


class PredictionService:
    def __init__(self, models_dir=None, log=None):
        self.models_dir = models_dir
        self._cache, self._lock = {}, threading.Lock()
        self._log = log or (lambda *a: None)

    # ------------------------------------------------------------------ model access (cached, hot-reload)
    def _bundle(self, horizon):
        stamp = reg.current_pointer_mtime(horizon, self.models_dir)
        with self._lock:
            hit = self._cache.get(horizon)
            if hit and hit[0] == stamp:
                return hit[1]
        # may raise ModelError
        bundle = reg.load_bundle(horizon, self.models_dir)
        with self._lock:
            self._cache[horizon] = (stamp, bundle)
        return bundle

    def horizons(self):
        return reg.deployed_horizons(self.models_dir)

    # ------------------------------------------------------------------ catalogue / status
    def status(self, db, now=None):
        now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(
            "UTC").tz_localize(None)
        models = {}
        for h in self.horizons():
            try:
                b = self._bundle(h)
                models[str(h)] = {"version": b["version"], "trained_at": b["trained_at"], "model": b["regressor_name"],
                                  "congestion_source": "classifier" if b.get("classifier") is not None else "derived_from_speed"}
            except reg.ModelError as e:
                self._log("model load failed", h, e)
                models[str(h)] = {"error": "model could not be loaded"}
        latest = db.execute("SELECT MAX(recorded_at) FROM traffic_records").fetchone()[
            0] if db.has_table("traffic_records") else None
        age = None if latest is None else round(
            (now - pd.Timestamp(latest)).total_seconds() / 60, 1)
        state = "not_connected" if not models else ("error" if any(
            "error" in m for m in models.values()) else "ready")
        return {"model_status": state, "models": models, "horizons": self.horizons(),
                "data": {"latest_observation": latest, "age_minutes": age,
                         "fresh": bool(age is not None and age <= cfg.MAX_OBSERVATION_AGE_MIN)}}

    def catalogue(self, db):
        segs = db.frame("SELECT s.id, s.name FROM road_segments s WHERE EXISTS (SELECT 1 FROM traffic_records r WHERE r.segment_id = s.id) ORDER BY s.name") \
            if db.has_table("road_segments") and db.has_table("traffic_records") else pd.DataFrame(columns=["id", "name"])
        return [{"id": int(r.id), "name": r.name} for r in segs.itertuples()]

    # ------------------------------------------------------------------ prediction
    def predict(self, db, segment_id, horizon, persist=False, now=None):
        try:
            segment_id, horizon = int(segment_id), int(horizon)
        except (TypeError, ValueError):
            return _unavailable("invalid_input")
        now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(
            "UTC").tz_localize(None)
        deployed = self.horizons()
        if not deployed:
            return _unavailable("no_model", segment_id, horizon)
        if horizon not in deployed:
            return _unavailable("unsupported_horizon", segment_id, horizon, {"supported_horizons": deployed})
        seg = db.frame(
            "SELECT id, name, length_km FROM road_segments WHERE id=?", (segment_id,))
        if seg.empty:
            return _unavailable("unknown_segment", segment_id, horizon)
        name, length_km = seg.at[0, "name"], seg.at[0, "length_km"]
        base = {"road_segment_id": segment_id, "road_name": name}
        try:
            bundle = self._bundle(horizon)
        except reg.ModelError as e:
            self._log("model load failed", horizon, e)
            return _unavailable("model_error", segment_id, horizon, base)
        fcfg = bundle["feature_config"]
        ref = fcfg["ref_speeds"].get(str(segment_id))
        if ref is None:
            return _unavailable("segment_not_in_model", segment_id, horizon, base)

        # Only measurements up to `now` count: nothing recorded after the prediction moment may be used.
        latest = db.execute("SELECT MAX(recorded_at) FROM traffic_records WHERE segment_id=? AND recorded_at<=?",
                            (segment_id, _fmt(now))).fetchone()[0]
        if latest is None:
            return _unavailable("no_data", segment_id, horizon, base)
        latest_ts = pd.Timestamp(latest)
        age_min = (now - latest_ts).total_seconds() / 60
        if age_min > cfg.MAX_OBSERVATION_AGE_MIN:
            return _unavailable("stale_data", segment_id, horizon, {**base, "observed_at": _fmt(latest_ts), "data_age_minutes": round(age_min, 1)},
                                f"The newest measurement for this road is from {_fmt(latest_ts)} UTC ({age_min:.0f} min old; "
                                f"limit {cfg.MAX_OBSERVATION_AGE_MIN} min). Short-term predictions need fresh data.")

        bins = fe.history_bins(fcfg)
        start = latest_ts - \
            pd.Timedelta(minutes=(bins + 6) * fcfg["bin_minutes"])
        obs = dl.load_observations(
            db, [segment_id], start=start, end=latest_ts)
        clean, _ = dc.clean_observations(obs, seg.rename(
            columns={"length_km": "length_km"}), now=now)
        grid = dc.to_grid(clean, fcfg["bin_minutes"])
        inc = dl.load_incidents(
            db, start=start, end=latest_ts) if fcfg["use_incidents"] else None
        feats = fe.build_features(grid, fcfg, inc)
        row = fe.add_target_calendar(feats.tail(
            1), horizon, fcfg) if len(feats) else pd.DataFrame()
        need = fe.essential_columns(fcfg)
        if row.empty or row[need].isna().any(axis=None):
            return _unavailable("insufficient_data", segment_id, horizon, {**base, "observed_at": _fmt(latest_ts)},
                                f"Need about {bins * fcfg['bin_minutes']} minutes of consecutive recent measurements; gaps prevent that.")

        X = row[bundle["feature_columns"] + ["segment_id"]]
        speed = float(max(0.0, bundle["regressor"].predict(X)[0]))
        cong = str(bundle["classifier"].predict(X)[0]) if bundle.get("classifier") is not None else \
            str(label_ratio(speed / ref, **fcfg["thresholds"]))
        now_speed = float(row["speed_now"].iloc[0])
        delay = None
        if pd.notna(length_km) and length_km > 0 and speed > 0:
            delay = round(max(0.0, float(length_km) / speed *
                          60 - float(length_km) / ref * 60), 1)
        last_ts = pd.Timestamp(row["ts"].iloc[0])
        target = last_ts + pd.Timedelta(minutes=horizon)
        obs_hist = grid[grid["observed"]].tail(24)
        out = {**base, "available": True, "prediction_status": "available", "reason": None, "message": None, "model_status": "ready",
               "prediction_horizon_minutes": horizon,
               "current_speed_kmh": round(now_speed, 1), "current_congestion": str(label_ratio(now_speed / ref, **fcfg["thresholds"])),
               "predicted_speed_kmh": round(speed, 1), "predicted_congestion": cong, "estimated_delay_minutes": delay,
               "reference_speed_kmh": round(ref, 1), "observed_at": _fmt(latest_ts), "prediction_timestamp": _fmt(target),
               "generated_at": _fmt(now), "data_age_minutes": round(age_min, 1),
               "model": {"name": bundle["regressor_name"], "version": bundle["version"], "trained_at": bundle["trained_at"],
                         "congestion_from": "classifier" if bundle.get("classifier") is not None else "predicted speed"},
               "history": [{"t": _fmt(r.ts), "value": round(float(r.speed_kmh), 1)} for r in obs_hist.itertuples()],
               "forecast": [{"t": _fmt(target), "value": round(speed, 1)}]}
        if persist:
            db.execute("INSERT OR IGNORE INTO traffic_predictions(segment_id,horizon_min,predicted_speed_kmh,predicted_congestion,estimated_delay_min,"
                       "target_time,generated_at,source_observation_at,model_version) VALUES(?,?,?,?,?,?,?,?,?)" if db.engine == "sqlite" else
                       "INSERT IGNORE INTO traffic_predictions(segment_id,horizon_min,predicted_speed_kmh,predicted_congestion,estimated_delay_min,"
                       "target_time,generated_at,source_observation_at,model_version) VALUES(?,?,?,?,?,?,?,?,?)",
                       (segment_id, horizon, speed, cong, delay, _fmt(target), _fmt(now), _fmt(latest_ts), bundle["version"]))
            db.commit()
        return out

    def generate_all(self, db, now=None):
        """Predict every road x deployed horizon and store the results (run from a scheduler / cron)."""
        done = {"stored": 0, "unavailable": {}}
        for seg in self.catalogue(db):
            for h in self.horizons():
                r = self.predict(db, seg["id"], h, persist=True, now=now)
                if r["available"]:
                    done["stored"] += 1
                else:
                    done["unavailable"][r["reason"]] = done["unavailable"].get(
                        r["reason"], 0) + 1
        return done

    def history(self, db, segment_id, limit=200):
        """Stored predictions (model output), newest first. Kept apart from measured observations."""
        if not db.has_table("traffic_predictions"):
            return []
        lim = max(1, min(int(limit), 1000))
        df = db.frame("SELECT horizon_min, predicted_speed_kmh, predicted_congestion, estimated_delay_min, target_time, generated_at, model_version "
                      "FROM traffic_predictions WHERE segment_id=? ORDER BY target_time DESC LIMIT " + str(lim), (int(segment_id),))
        return df.to_dict("records")

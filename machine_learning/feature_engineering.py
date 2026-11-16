
import numpy as np
import pandas as pd

from . import model_config as cfg
from .congestion_classifier import label_ratio


# --------------------------------------------------------------------------------------- configuration
def reference_speeds(grid_train, segments):
    """Per-road reference ('free-flow') speed: segment's own value if given, else the 85th percentile of
    its TRAINING-period measured speeds. Never computed from validation/test data."""
    refs = {}
    for sid, g in grid_train[grid_train["observed"]].groupby("segment_id"):
        refs[int(sid)] = float(np.percentile(
            g["speed_kmh"], cfg.REFERENCE_PERCENTILE))
    for r in segments.itertuples():
        if pd.notna(r.free_flow_speed_kmh) and r.free_flow_speed_kmh > 0:
            refs[int(r.id)] = float(r.free_flow_speed_kmh)
    return refs


def make_feature_config(grid_train, segments, incidents, refs, bin_minutes=cfg.BIN_MINUTES):
    """Decide, from the TRAINING data only, which optional features the dataset supports."""
    obs = grid_train[grid_train["observed"]]
    coords = {str(int(r.id)): ([float(r.latitude), float(r.longitude)] if pd.notna(r.latitude) and pd.notna(r.longitude) else None)
              for r in segments.itertuples()}
    local = obs["ts"] + pd.Timedelta(hours=cfg.LOCAL_UTC_OFFSET_HOURS)
    return {
        "bin_minutes": int(bin_minutes), "lag_bins": list(cfg.LAG_BINS), "roll_windows": list(cfg.ROLL_WINDOWS),
        "utc_offset_hours": cfg.LOCAL_UTC_OFFSET_HOURS, "rush_hours": [list(r) for r in cfg.RUSH_HOURS],
        "ref_speeds": {str(k): v for k, v in refs.items()},
        "use_vehicle_count": bool(len(obs) and obs["vehicle_count"].notna().mean() >= 0.5),
        "use_density": bool(len(obs) and obs["density"].notna().mean() >= 0.5),
        "use_incidents": bool(len(incidents) >= cfg.MIN_INCIDENT_ROWS_FOR_FEATURE and any(coords.values())),
        "use_month": bool(local.dt.to_period("M").nunique() >= 2),
        "segment_coords": coords, "incident_radius_km": cfg.INCIDENT_RADIUS_KM,
        "incident_window_min": cfg.INCIDENT_WINDOW_MIN,
        "thresholds": {"low": cfg.LOW_RATIO, "heavy": cfg.HEAVY_RATIO},
    }


def history_bins(fcfg):
    """How many bins (including 'now') must be available to build one feature row."""
    return max(max(fcfg["lag_bins"]), max(fcfg["roll_windows"]) - 1) + 1


def feature_columns(fcfg):
    cols = ["speed_now", "speed_ratio_now"] + \
        [f"speed_lag_{k}" for k in fcfg["lag_bins"]]
    for w in fcfg["roll_windows"]:
        cols += [f"speed_roll_mean_{w}", f"speed_roll_std_{w}"]
    if fcfg["use_vehicle_count"]:
        cols += ["vehicle_count_now", "vehicle_count_roll_mean_6"]
    if fcfg["use_density"]:
        cols += ["density_now"]
    if fcfg["use_incidents"]:
        cols += ["incidents_nearby"]
    cols += _calendar_names("", fcfg) + _calendar_names("target_", fcfg)
    return cols


def essential_columns(fcfg):
    """Rows missing any of these cannot be turned into a trustworthy feature row."""
    cols = ["speed_now"] + [f"speed_lag_{k}" for k in fcfg["lag_bins"]]
    for w in fcfg["roll_windows"]:
        cols += [f"speed_roll_mean_{w}", f"speed_roll_std_{w}"]
    return cols


# ------------------------------------------------------------------------------------------ features
def _calendar_names(prefix, fcfg):
    names = ["hour_sin", "hour_cos", "dow", "is_weekend",
             "is_rush"] + (["month"] if fcfg["use_month"] else [])
    return [prefix + n for n in names]


def calendar_features(ts, fcfg, prefix=""):
    local = ts + pd.Timedelta(hours=fcfg["utc_offset_hours"])
    hf = local.dt.hour + local.dt.minute / 60.0
    dow = local.dt.day_of_week
    rush = np.zeros(len(ts), dtype=bool)
    for a, b in fcfg["rush_hours"]:
        rush |= ((hf >= a) & (hf < b)).to_numpy()
    out = {prefix + "hour_sin": np.sin(2 * np.pi * hf / 24), prefix + "hour_cos": np.cos(2 * np.pi * hf / 24),
           prefix + "dow": dow, prefix + "is_weekend": (dow >= 5).astype(int), prefix + "is_rush": rush.astype(int)}
    if fcfg["use_month"]:
        out[prefix + "month"] = local.dt.month
    if prefix == "target_":
        out["target_hour"] = local.dt.hour
    return pd.DataFrame(out, index=ts.index)


def _haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * \
        np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def _incident_counts(ts, coord, incidents, fcfg):
    """Incidents reported within the radius during the window ENDING at each ts (never after it)."""
    if coord is None or incidents is None or incidents.empty:
        return np.full(len(ts), np.nan if coord is None else 0.0)
    near = incidents[_haversine_km(coord[0], coord[1], incidents["latitude"].astype(float), incidents["longitude"].astype(float))
                     <= fcfg["incident_radius_km"]]
    if near.empty:
        return np.zeros(len(ts))
    times = np.sort(pd.to_datetime(near["created_at"]).to_numpy().astype(
        "datetime64[ns]").astype("int64"))
    t = ts.to_numpy().astype("datetime64[ns]").astype("int64")
    window = int(fcfg["incident_window_min"] * 60 * 1e9)
    return (np.searchsorted(times, t, side="right") - np.searchsorted(times, t - window, side="right")).astype(float)


def build_features(grid, fcfg, incidents=None):
    """One feature row per (segment, bin) using only data at or before that bin."""
    parts = []
    for sid, g in grid.groupby("segment_id", sort=False):
        g = g.sort_values("ts").reset_index(drop=True)
        s = g["speed_kmh"]
        ref = fcfg["ref_speeds"].get(str(int(sid)))
        f = pd.DataFrame({"segment_id": int(
            sid), "ts": g["ts"], "speed_now": s, "observed_now": g["observed"]})
        f["speed_ratio_now"] = s / ref if ref else np.nan
        for k in fcfg["lag_bins"]:
            f[f"speed_lag_{k}"] = s.shift(k)
        # trailing windows: include now and the past only
        for w in fcfg["roll_windows"]:
            r = s.rolling(w, min_periods=w)
            f[f"speed_roll_mean_{w}"], f[f"speed_roll_std_{w}"] = r.mean(
            ), r.std()
        if fcfg["use_vehicle_count"]:
            f["vehicle_count_now"] = g["vehicle_count"]
            f["vehicle_count_roll_mean_6"] = g["vehicle_count"].rolling(
                6, min_periods=3).mean()
        if fcfg["use_density"]:
            f["density_now"] = g["density"]
        if fcfg["use_incidents"]:
            f["incidents_nearby"] = _incident_counts(
                g["ts"], fcfg["segment_coords"].get(str(int(sid))), incidents, fcfg)
        f = pd.concat([f, calendar_features(g["ts"], fcfg)], axis=1)
        parts.append(f)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def add_target_calendar(feats, horizon_min, fcfg):
    target_time = feats["ts"] + pd.Timedelta(minutes=horizon_min)
    out = feats.copy()
    out["target_time"] = target_time
    return pd.concat([out, calendar_features(target_time, fcfg, prefix="target_")], axis=1)


def make_supervised(feats, grid, horizon_min, fcfg):
    """Attach the true future speed/class as the target. Only MEASURED target bins are used (observed=True),
    so the model is never trained to reproduce a carried-forward value."""
    if horizon_min % fcfg["bin_minutes"]:
        raise ValueError(
            f"horizon {horizon_min} min is not a multiple of the {fcfg['bin_minutes']}-min grid")
    d = add_target_calendar(feats, horizon_min, fcfg)
    tgt = grid[["segment_id", "ts", "speed_kmh", "observed"]].rename(
        columns={"ts": "target_time", "speed_kmh": "target_speed", "observed": "target_observed"})
    d = d.merge(tgt, on=["segment_id", "target_time"], how="inner")
    d = d[d["target_observed"].astype(bool)].drop(columns="target_observed")
    d = d.dropna(subset=essential_columns(fcfg) + ["target_speed"])
    ref = d["segment_id"].map(lambda s: fcfg["ref_speeds"].get(str(int(s))))
    d = d[ref.notna()].copy()
    d["target_class"] = label_ratio(d["target_speed"] / ref[ref.notna()], **{
        "low": fcfg["thresholds"]["low"], "heavy": fcfg["thresholds"]["heavy"]})
    return d.sort_values(["ts", "segment_id"]).reset_index(drop=True)

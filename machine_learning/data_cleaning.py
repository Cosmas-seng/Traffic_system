
import numpy as np
import pandas as pd

from . import model_config as cfg


def clean_observations(obs, segments, now=None):
    """Return (clean_df, report). `obs` columns: segment_id, recorded_at, speed_kmh, vehicle_count, density."""
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(
        "UTC").tz_localize(None)
    rep = {"rows_in": int(len(obs)), "removed": {}, "flagged": {}}
    df = obs.copy()

    def drop(mask, reason):
        n = int(mask.sum())
        if n:
            rep["removed"][reason] = rep["removed"].get(reason, 0) + n
        return df[~mask].copy()

    df["recorded_at"] = pd.to_datetime(df["recorded_at"], errors="coerce")
    bad_ts = df["recorded_at"].isna() | (df["recorded_at"] < pd.Timestamp(
        "2000-01-01")) | (df["recorded_at"] > now + pd.Timedelta(days=1))
    df = drop(bad_ts, "invalid_timestamp")

    df["segment_id"] = pd.to_numeric(df["segment_id"], errors="coerce")
    known = set(segments["id"].astype(int)) if len(segments) else set()
    df = drop(df["segment_id"].isna() | ~
              df["segment_id"].isin(known), "unknown_segment_id")
    df["segment_id"] = df["segment_id"].astype(int)

    for col in ("speed_kmh", "vehicle_count", "density"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    impossible_speed = (df["speed_kmh"] < 0) | (
        df["speed_kmh"] > cfg.MAX_PLAUSIBLE_SPEED_KMH)
    rep["removed"]["impossible_speed_set_missing"] = int(
        impossible_speed.sum())
    df.loc[impossible_speed, "speed_kmh"] = np.nan
    for col, hi in (("vehicle_count", cfg.MAX_PLAUSIBLE_COUNT), ("density", None)):
        bad = (df[col] < 0) | ((df[col] > hi) if hi else False)
        rep["removed"][f"impossible_{col}_set_missing"] = int(bad.sum())
        df.loc[bad, col] = np.nan

    df = df.sort_values(["segment_id", "recorded_at"], kind="mergesort")
    dup = df.duplicated(["segment_id", "recorded_at"], keep="last")
    df = drop(dup, "duplicate_segment_timestamp")
    df = drop(df["speed_kmh"].isna(), "no_usable_speed")

    # ---- flags (kept, not removed) ----
    flagged, sample = 0, []
    for sid, g in df.groupby("segment_id"):
        med = g["speed_kmh"].median()
        mad = (g["speed_kmh"] - med).abs().median() * 1.4826
        if mad > 0:
            out = g[(g["speed_kmh"] - med).abs() > 6 * mad]
            flagged += len(out)
            sample += out.head(3).assign(segment_id=sid)[
                ["segment_id", "recorded_at", "speed_kmh"]].astype(str).values.tolist()
    rep["flagged"]["statistical_speed_outliers_kept"] = int(flagged)
    rep["flagged"]["outlier_sample_for_review"] = sample[:10]
    stuck = 0
    for sid, g in df.groupby("segment_id"):
        run = (g["speed_kmh"] != g["speed_kmh"].shift()).cumsum()
        stuck += int((g.groupby(run)
                     ["speed_kmh"].transform("size") >= 24).sum())
    rep["flagged"]["rows_in_possible_stuck_sensor_runs_kept"] = stuck

    df = df.reset_index(drop=True)
    rep["rows_out"] = int(len(df))
    rep["segments"] = int(df["segment_id"].nunique()) if len(df) else 0
    if len(df):
        rep["first"], rep["last"] = str(
            df["recorded_at"].min()), str(df["recorded_at"].max())
        rep["span_days"] = round(
            (df["recorded_at"].max() - df["recorded_at"].min()).total_seconds() / 86400, 2)
        iv = df.groupby("segment_id")[
            "recorded_at"].diff().dt.total_seconds().div(60)
        rep["median_interval_min"] = None if iv.dropna(
        ).empty else round(float(iv.median()), 2)
        rep["per_segment_rows"] = {int(k): int(
            v) for k, v in df.groupby("segment_id").size().items()}
    else:
        rep.update(first=None, last=None, span_days=0.0,
                   median_interval_min=None, per_segment_rows={})
    return df, rep


def to_grid(df, bin_minutes=cfg.BIN_MINUTES, max_ffill=cfg.MAX_FFILL_BINS):
    """Average observations onto a regular time grid per segment.

    `observed` is True only for bins that contain a real measurement. Short gaps are bridged by carrying
    the last value forward (inputs only): such bins stay observed=False so they are never used as a
    training TARGET. Longer gaps stay empty (NaN).
    """
    cols = ["ts", "segment_id", "speed_kmh",
            "vehicle_count", "density", "observed"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    d = df.copy()
    d["bin"] = d["recorded_at"].dt.floor(f"{bin_minutes}min")
    agg = d.groupby(["segment_id", "bin"]).agg(speed_kmh=("speed_kmh", "mean"), vehicle_count=("vehicle_count", "mean"),
                                               density=("density", "mean")).reset_index()
    parts = []
    for sid, g in agg.groupby("segment_id"):
        idx = pd.date_range(g["bin"].min(), g["bin"].max(),
                            freq=f"{bin_minutes}min")
        g = g.set_index("bin").reindex(idx)
        g["observed"] = g["speed_kmh"].notna()
        value_cols = ["speed_kmh", "vehicle_count", "density"]
        g[value_cols] = g[value_cols].ffill(limit=max_ffill)
        g["segment_id"] = sid
        parts.append(g.rename_axis("ts").reset_index())
    return pd.concat(parts, ignore_index=True)[cols]


def grid_quality(grid):
    """Share of grid bins holding a real measurement, per segment (a coverage report for the assessment)."""
    if grid.empty:
        return {}
    return {int(k): round(float(v), 3) for k, v in grid.groupby("segment_id")["observed"].mean().items()}

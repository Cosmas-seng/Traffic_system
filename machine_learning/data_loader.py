
import os
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MIGRATION = ROOT / "migrations" / "001_traffic_ml.sql"
TS_FMT = "%Y-%m-%d %H:%M:%S"
OBS_COLS = ["segment_id", "recorded_at",
            "speed_kmh", "vehicle_count", "density"]


def fmt_ts(value):
    """datetime/Timestamp -> storage string."""
    return pd.Timestamp(value).strftime(TS_FMT)


class Database:
    def __init__(self, conn, engine="sqlite"):
        self.conn, self.engine = conn, engine

    @classmethod
    def from_env(cls, path=None):
        """DB_ENGINE=mysql with DB_HOST/DB_USER/DB_PASSWORD/DB_NAME, otherwise SQLite at DATABASE_PATH."""
        if os.environ.get("DB_ENGINE", "sqlite").lower() == "mysql":
            import pymysql  # pip install PyMySQL  (not required for SQLite)
            conn = pymysql.connect(host=os.environ["DB_HOST"], user=os.environ["DB_USER"],
                                   password=os.environ["DB_PASSWORD"], database=os.environ["DB_NAME"],
                                   charset="utf8mb4")
            return cls(conn, "mysql")
        path = path or os.environ.get(
            "DATABASE_PATH") or str(ROOT / "traffic.db")
        return cls(sqlite3.connect(path), "sqlite")

    # -- helpers --------------------------------------------------------------------------------
    def q(self, sql):
        return sql.replace("?", "%s") if self.engine == "mysql" else sql

    def execute(self, sql, params=()):
        cur = self.conn.cursor()
        cur.execute(self.q(sql), tuple(params))
        return cur

    def frame(self, sql, params=()):
        cur = self.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return pd.DataFrame([tuple(r) for r in cur.fetchall()], columns=cols)

    def commit(self):
        self.conn.commit()

    def ensure_schema(self):
        """Apply the ADDITIVE migration (CREATE ... IF NOT EXISTS only). SQLite only; for MySQL run a
        translated copy of migrations/001_traffic_ml.sql by hand (see the notes at the top of that file)."""
        if self.engine != "sqlite":
            raise RuntimeError(
                "Apply migrations/001_traffic_ml.sql to MySQL/MariaDB manually (translated syntax).")
        self.conn.executescript(MIGRATION.read_text())
        self.conn.commit()

    def has_table(self, name):
        if self.engine == "sqlite":
            return self.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None
        return self.execute("SELECT 1 FROM information_schema.tables WHERE table_schema=DATABASE() AND table_name=?",
                            (name,)).fetchone() is not None


# ------------------------------------------------------------------------------------------------
def inventory(db):
    """Row counts of the tables the ML component cares about (used by the data assessment)."""
    out = {}
    for t in ("road_segments", "traffic_records", "traffic_predictions", "incidents"):
        out[t] = db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[
            0] if db.has_table(t) else None  # t is a fixed literal
    return out


def load_segments(db):
    cols = "id, name, latitude, longitude, length_km, free_flow_speed_kmh, speed_limit_kmh, geometry_json"
    return db.frame(f"SELECT {cols} FROM road_segments ORDER BY id")


def load_observations(db, segment_ids=None, start=None, end=None):
    """Measured traffic, chronologically sorted. `start`/`end` bound the range (inclusive)."""
    sql = "SELECT segment_id, recorded_at, speed_kmh, vehicle_count, density FROM traffic_records WHERE 1=1"
    params = []
    if segment_ids is not None:
        ids = [int(i) for i in segment_ids]
        if not ids:
            return pd.DataFrame(columns=OBS_COLS)
        sql += " AND segment_id IN (" + ",".join("?" * len(ids)) + ")"
        params += ids
    if start is not None:
        sql += " AND recorded_at >= ?"
        params.append(fmt_ts(start))
    if end is not None:
        sql += " AND recorded_at <= ?"
        params.append(fmt_ts(end))
    return db.frame(sql + " ORDER BY segment_id, recorded_at", params)


def load_incidents(db, start=None, end=None):
    if not db.has_table("incidents"):
        return pd.DataFrame(columns=["latitude", "longitude", "created_at"])
    sql, params = "SELECT latitude, longitude, created_at FROM incidents WHERE 1=1", []
    if start is not None:
        sql += " AND created_at >= ?"
        params.append(fmt_ts(start))
    if end is not None:
        sql += " AND created_at <= ?"
        params.append(fmt_ts(end))
    return db.frame(sql + " ORDER BY created_at", params)


# ------------------------------------------------------------------------------------------------
def import_traffic_csv(db, csv_path, source, utc_offset_hours=0, segments_csv=None):
    """Load a REAL historical dataset into traffic_records. Returns a report; nothing is invented.

    Required columns: recorded_at, speed_kmh, and segment_id OR segment_name.
    Optional: vehicle_count, density.  `utc_offset_hours`: local-time offset of naive timestamps
    (3 for Nairobi local time); timestamps that carry a timezone are converted to UTC automatically.
    """
    db.ensure_schema()
    report = {"rows_read": 0, "rows_rejected": {}, "rows_inserted": 0,
              "duplicates_skipped": 0, "segments_created": 0}
    if segments_csv:
        _upsert_segments(db, pd.read_csv(segments_csv), report)

    df = pd.read_csv(csv_path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    report["rows_read"] = len(df)
    for need in ("recorded_at", "speed_kmh"):
        if need not in df.columns:
            raise ValueError(f"CSV is missing the required column '{need}'.")
    if "segment_id" not in df.columns and "segment_name" not in df.columns:
        raise ValueError("CSV needs a 'segment_id' or 'segment_name' column.")

    def reject(mask, reason):
        n = int(mask.sum())
        if n:
            report["rows_rejected"][reason] = report["rows_rejected"].get(
                reason, 0) + n
        return df[~mask]

    ts = pd.to_datetime(df["recorded_at"], errors="coerce",
                        utc=False, format="mixed")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        ts = ts - pd.Timedelta(hours=utc_offset_hours)
    df["tmp_ts"] = ts
    df = reject(df["tmp_ts"].isna(), "invalid_timestamp")
    df["tmp_speed"] = pd.to_numeric(df["speed_kmh"], errors="coerce")
    df = reject(df["tmp_speed"].isna(), "non_numeric_speed")
    for col in ("vehicle_count", "density"):
        df["tmp_" + col] = pd.to_numeric(
            df[col], errors="coerce") if col in df.columns else float("nan")

    # resolve segment ids
    if "segment_id" in df.columns:
        df["tmp_sid"] = pd.to_numeric(df["segment_id"], errors="coerce")
    else:
        ids = {}
        for name in df["segment_name"].dropna().str.strip().unique():
            row = db.execute(
                "SELECT id FROM road_segments WHERE name=?", (name,)).fetchone()
            if row is None:
                cur = db.execute(
                    "INSERT INTO road_segments(name) VALUES(?)", (name,))
                ids[name] = cur.lastrowid
                report["segments_created"] += 1
            else:
                ids[name] = row[0]
        df["tmp_sid"] = df["segment_name"].str.strip().map(ids)
    df = reject(df["tmp_sid"].isna(), "unknown_segment")

    verb = "INSERT OR IGNORE" if db.engine == "sqlite" else "INSERT IGNORE"
    sql = (f"{verb} INTO traffic_records(segment_id, recorded_at, speed_kmh, vehicle_count, density, source) "
           "VALUES(?,?,?,?,?,?)")
    before = db.conn.total_changes if db.engine == "sqlite" else 0
    rows = [(int(r.tmp_sid), r.tmp_ts.strftime(TS_FMT), float(r.tmp_speed),
             None if pd.isna(r.tmp_vehicle_count) else float(
                 r.tmp_vehicle_count),
             None if pd.isna(r.tmp_density) else float(r.tmp_density), source) for r in df.itertuples()]
    cur = db.conn.cursor()
    cur.executemany(db.q(sql), rows)
    db.commit()
    inserted = (db.conn.total_changes -
                before) if db.engine == "sqlite" else cur.rowcount
    report["rows_inserted"] = int(inserted)
    report["duplicates_skipped"] = len(rows) - int(inserted)
    return report


def _upsert_segments(db, seg, report):
    seg.columns = [c.strip().lower() for c in seg.columns]
    if "name" not in seg.columns:
        raise ValueError("segments CSV needs a 'name' column.")
    fields = [c for c in ("latitude", "longitude", "length_km", "free_flow_speed_kmh", "speed_limit_kmh", "geometry_json")
              if c in seg.columns]
    for _, r in seg.iterrows():
        name = str(r["name"]).strip()
        vals = [None if pd.isna(r[f]) else r[f] for f in fields]
        row = db.execute(
            "SELECT id FROM road_segments WHERE name=?", (name,)).fetchone()
        if row is None:
            db.execute("INSERT INTO road_segments(name" + "".join("," + f for f in fields) + ") VALUES(?" + ",?" * len(fields) + ")",
                       [name] + vals)
            report["segments_created"] += 1
        else:
            # field names come from the fixed whitelist above
            for f, v in zip(fields, vals):
                if v is not None:
                    db.execute(
                        f"UPDATE road_segments SET {f}=? WHERE id=?", (v, row[0]))
    db.commit()

from dotenv import load_dotenv
import os
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TS_FMT = "%Y-%m-%d %H:%M:%S"
OBS_COLS = ["segment_id", "recorded_at",
            "speed_kmh", "vehicle_count", "density"]
load_dotenv(ROOT / ".env")


def fmt_ts(value):
    return pd.Timestamp(value).strftime(TS_FMT)


def classify_traffic(speed):
    """Map speed to the existing MySQL traffic_level ENUM."""
    speed = float(speed)
    if speed >= 50:
        return "low"
    if speed >= 30:
        return "moderate"
    if speed >= 15:
        return "heavy"
    return "severe"


class Database:
    def __init__(self, conn, engine=None):
        self.conn = conn

        # Detect SQLite connections automatically. This is important for tests
        # that construct Database(sqlite_connection) without passing an engine.
        # Connections created by from_env() still pass their engine explicitly.
        if engine is None:
            engine = "sqlite" if isinstance(
                conn, sqlite3.Connection) else "mysql"

        self.engine = str(engine).lower()
        if self.engine not in {"mysql", "sqlite"}:
            raise ValueError("engine must be either 'mysql' or 'sqlite'.")

    @classmethod
    def from_env(cls, path=None):
        load_dotenv(ROOT / ".env", override=False)
        engine = os.environ.get("DB_ENGINE", "mysql").lower()
        if engine == "mysql":
            import pymysql
            conn = pymysql.connect(
                host=os.environ.get("DB_HOST", "127.0.0.1"),
                port=int(os.environ.get("DB_PORT", "3306")),
                user=os.environ["DB_USER"],
                password=os.environ.get("DB_PASSWORD", ""),
                database=os.environ.get("DB_NAME", "ai_system"),
                charset="utf8mb4",
                autocommit=False,
                cursorclass=pymysql.cursors.Cursor,
            )
            return cls(conn, "mysql")
        if engine == "sqlite":
            db_path = path or os.environ.get(
                "DATABASE_PATH") or str(ROOT / "traffic.db")
            return cls(sqlite3.connect(db_path), "sqlite")
        raise ValueError("DB_ENGINE must be either 'mysql' or 'sqlite'.")

    def q(self, sql):
        """Translate qmark placeholders for MySQL; SQLite uses qmarks natively."""
        return sql if self.engine == "sqlite" else sql.replace("?", "%s")

    def execute(self, sql, params=()):
        cur = self.conn.cursor()
        cur.execute(self.q(sql), tuple(params))
        return cur

    def frame(self, sql, params=()):
        cur = self.execute(sql, params)
        columns = [d[0] for d in cur.description] if cur.description else []
        return pd.DataFrame(cur.fetchall(), columns=columns)

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()

    def rollback(self):
        self.conn.rollback()

    def has_table(self, name):
        if self.engine == "sqlite":
            sql = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?"
            return self.execute(sql, (name,)).fetchone() is not None
        sql = """SELECT 1 FROM information_schema.tables
                 WHERE table_schema=DATABASE() AND table_name=?"""
        return self.execute(sql, (name,)).fetchone() is not None

    def columns(self, table):
        # table names passed here are fixed internal identifiers, not user input.
        if self.engine == "sqlite":
            return {row[1] for row in self.execute(f"PRAGMA table_info({table})").fetchall()}
        rows = self.execute(
            """SELECT column_name FROM information_schema.columns
               WHERE table_schema=DATABASE() AND table_name=?""", (table,)
        ).fetchall()
        return {row[0] for row in rows}

    def ensure_schema(self):
        """Create the isolated SQLite test schema, or validate the existing MySQL schema."""
        if self.engine == "sqlite":
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS road_segments (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    length_km REAL,
                    free_flow_speed_kmh REAL,
                    speed_limit_kmh REAL,
                    geometry_json TEXT
                );
                CREATE TABLE IF NOT EXISTS traffic_records (
                    traffic_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    segment_id INTEGER NOT NULL,
                    recorded_at TEXT NOT NULL,
                    speed_kmh REAL,
                    vehicle_count REAL,
                    density REAL,
                    source TEXT
                );
                CREATE TABLE IF NOT EXISTS incidents (
                    incident_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    latitude REAL,
                    longitude REAL,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS traffic_predictions (
                    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    segment_id INTEGER,
                    horizon_min INTEGER,
                    predicted_speed_kmh REAL,
                    predicted_congestion TEXT,
                    estimated_delay_min REAL,
                    target_time TEXT,
                    generated_at TEXT,
                    source_observation_at TEXT,
                    model_version TEXT,
                    UNIQUE(segment_id, horizon_min, target_time, model_version)
                );
            """)
            self.commit()
            return

        required = {
            "road_segments": {"segment_id", "road_name", "start_lat", "start_lng",
                              "end_lat", "end_lng", "length_km"},
            "traffic_records": {"traffic_id", "segment_id", "recorded_at",
                                "traffic_level", "vehicle_count", "average_speed",
                                "data_source"},
            "incidents": {"latitude", "longitude", "reported_at"},
        }
        for table, expected in required.items():
            if not self.has_table(table):
                raise RuntimeError(
                    f"Required table '{table}' was not found in database "
                    f"{os.environ.get('DB_NAME', 'ai_system')}. Check DB_NAME."
                )
            missing = expected - self.columns(table)
            if missing:
                raise RuntimeError(
                    f"Table '{table}' is missing expected columns: " +
                    ", ".join(sorted(missing))
                )
        if "density" not in self.columns("traffic_records"):
            self.execute(
                "ALTER TABLE traffic_records ADD COLUMN density DECIMAL(10,3) NULL")
            self.commit()


def inventory(db):
    tables = ("road_segments", "traffic_records",
              "traffic_predictions", "incidents")
    return {
        table: (db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                if db.has_table(table) else None)
        for table in tables
    }


def load_segments(db):
    db.ensure_schema()
    if db.engine == "sqlite":
        return db.frame("""
            SELECT id, name, latitude, longitude, length_km,
                   free_flow_speed_kmh, speed_limit_kmh, geometry_json
            FROM road_segments ORDER BY id
        """)
    return db.frame("""
        SELECT segment_id AS id, road_name AS name,
               start_lat AS latitude, start_lng AS longitude, length_km,
               NULL AS free_flow_speed_kmh, NULL AS speed_limit_kmh,
               NULL AS geometry_json, end_lat, end_lng
        FROM road_segments ORDER BY segment_id
    """)


def load_observations(db, segment_ids=None, start=None, end=None):
    db.ensure_schema()
    speed_col = "speed_kmh" if db.engine == "sqlite" else "average_speed"
    sql = f"""SELECT segment_id, recorded_at, {speed_col} AS speed_kmh,
                     vehicle_count, density
              FROM traffic_records WHERE 1=1"""
    params = []
    if segment_ids is not None:
        ids = [int(v) for v in segment_ids]
        if not ids:
            return pd.DataFrame(columns=OBS_COLS)
        sql += " AND segment_id IN (" + ",".join(["?"] * len(ids)) + ")"
        params.extend(ids)
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
    time_col = "created_at" if db.engine == "sqlite" else "reported_at"
    sql = f"SELECT latitude, longitude, {time_col} AS created_at FROM incidents WHERE 1=1"
    params = []
    if start is not None:
        sql += f" AND {time_col} >= ?"
        params.append(fmt_ts(start))
    if end is not None:
        sql += f" AND {time_col} <= ?"
        params.append(fmt_ts(end))
    return db.frame(sql + f" ORDER BY {time_col}", params)


def _upsert_segments(db, segments, report):
    segments = segments.copy()
    segments.columns = [str(c).strip().lower() for c in segments.columns]
    required = {"name", "start_lat", "start_lng", "end_lat", "end_lng"}
    missing = required - set(segments.columns)
    if missing:
        raise ValueError(
            "roads.csv is missing required columns: " + ", ".join(sorted(missing)))

    for _, row in segments.iterrows():
        name = str(row["name"]).strip()
        if not name or name.lower() == "nan":
            report["rows_rejected"]["road_missing_name"] = report["rows_rejected"].get(
                "road_missing_name", 0) + 1
            continue
        try:
            start_lat, start_lng = float(
                row["start_lat"]), float(row["start_lng"])
            end_lat, end_lng = float(row["end_lat"]), float(row["end_lng"])
        except (TypeError, ValueError):
            report["rows_rejected"]["road_invalid_coordinates"] = report["rows_rejected"].get(
                "road_invalid_coordinates", 0) + 1
            continue
        length = row.get("length_km")
        length = None if pd.isna(length) else float(length)

        if db.engine == "sqlite":
            existing = db.execute(
                "SELECT id FROM road_segments WHERE name=? LIMIT 1", (name,)).fetchone()
            if existing is None:
                db.execute("""INSERT INTO road_segments
                    (name, latitude, longitude, length_km)
                    VALUES (?, ?, ?, ?)""", (name, start_lat, start_lng, length))
                report["segments_created"] += 1
            else:
                db.execute("""UPDATE road_segments SET latitude=?, longitude=?,
                    length_km=COALESCE(?, length_km) WHERE id=?""",
                           (start_lat, start_lng, length, existing[0]))
        else:
            existing = db.execute(
                "SELECT segment_id FROM road_segments WHERE road_name=? LIMIT 1", (name,)).fetchone()
            if existing is None:
                db.execute("""INSERT INTO road_segments
                    (road_name, start_lat, start_lng, end_lat, end_lng, length_km)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                           (name, start_lat, start_lng, end_lat, end_lng, length))
                report["segments_created"] += 1
            else:
                db.execute("""UPDATE road_segments SET start_lat=?, start_lng=?,
                    end_lat=?, end_lng=?, length_km=COALESCE(?, length_km)
                    WHERE segment_id=?""",
                           (start_lat, start_lng, end_lat, end_lng, length, existing[0]))
    db.commit()


def import_traffic_csv(db, csv_path, source, utc_offset_hours=0, segments_csv=None):
    """Import traffic CSV rows. Naive timestamps are converted from local time to UTC."""
    db.ensure_schema()
    report = {"rows_read": 0, "rows_rejected": {}, "rows_inserted": 0,
              "duplicates_skipped": 0, "segments_created": 0}
    try:
        if segments_csv:
            _upsert_segments(db, pd.read_csv(segments_csv), report)
        df = pd.read_csv(csv_path, dtype=str)
        df.columns = [str(c).strip().lower() for c in df.columns]
        report["rows_read"] = len(df)
        for col in ("recorded_at", "speed_kmh", "segment_name"):
            if col not in df.columns:
                raise ValueError(
                    f"traffic.csv is missing required column '{col}'")

        def reject(mask, reason):
            count = int(mask.sum())
            if count:
                report["rows_rejected"][reason] = report["rows_rejected"].get(
                    reason, 0) + count
            return df.loc[~mask].copy()

        ts = pd.to_datetime(df["recorded_at"], errors="coerce", format="mixed")
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            ts = ts - pd.Timedelta(hours=float(utc_offset_hours))
        df["tmp_ts"] = ts
        df = reject(df["tmp_ts"].isna(), "invalid_timestamp")
        df["tmp_speed"] = pd.to_numeric(df["speed_kmh"], errors="coerce")
        bad_numeric_speed = df["tmp_speed"].isna()
        df = reject(bad_numeric_speed, "non_numeric_speed")
        df = reject(df["tmp_speed"] < 0, "invalid_speed")
        for col in ("vehicle_count", "density"):
            df["tmp_" + col] = pd.to_numeric(df[col],
                                             errors="coerce") if col in df else float("nan")
        df["tmp_name"] = df["segment_name"].fillna("").astype(str).str.strip()
        df = reject(df["tmp_name"].eq(""), "missing_segment_name")

        sid_col, name_col = (("id", "name") if db.engine ==
                             "sqlite" else ("segment_id", "road_name"))
        segment_map = {}
        for name in df["tmp_name"].drop_duplicates():
            found = db.execute(
                f"SELECT {sid_col} FROM road_segments WHERE {name_col}=? LIMIT 1", (name,)).fetchone()
            if found is not None:
                segment_map[name] = int(found[0])
            elif db.engine == "sqlite":
                db.execute("INSERT INTO road_segments(name) VALUES(?)", (name,))
                segment_map[name] = int(db.execute(
                    "SELECT id FROM road_segments WHERE name=?", (name,)).fetchone()[0])
                report["segments_created"] += 1
        df["tmp_sid"] = df["tmp_name"].map(segment_map)
        df = reject(df["tmp_sid"].isna(), "unknown_segment")

        for row in df.itertuples(index=False):
            sid = int(row.tmp_sid)
            recorded_at = row.tmp_ts.strftime(TS_FMT)
            speed = float(row.tmp_speed)
            count = None if pd.isna(row.tmp_vehicle_count) else int(
                float(row.tmp_vehicle_count))
            density = None if pd.isna(
                row.tmp_density) else float(row.tmp_density)
            if db.engine == "sqlite":
                duplicate = db.execute("""SELECT 1 FROM traffic_records
                    WHERE segment_id=? AND recorded_at=? AND source=? LIMIT 1""",
                                       (sid, recorded_at, source)).fetchone()
                if duplicate:
                    report["duplicates_skipped"] += 1
                    continue
                db.execute("""INSERT INTO traffic_records
                    (segment_id, recorded_at, speed_kmh, vehicle_count, density, source)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                           (sid, recorded_at, speed, count, density, source))
            else:
                duplicate = db.execute("""SELECT traffic_id FROM traffic_records
                    WHERE segment_id=? AND recorded_at=? AND data_source=? LIMIT 1""",
                                       (sid, recorded_at, source)).fetchone()
                if duplicate:
                    report["duplicates_skipped"] += 1
                    continue
                db.execute("""INSERT INTO traffic_records
                    (segment_id, recorded_at, traffic_level, vehicle_count,
                     average_speed, data_source, density)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                           (sid, recorded_at, classify_traffic(speed), count, speed, source, density))
            report["rows_inserted"] += 1
        db.commit()
        return report
    except Exception:
        db.rollback()
        raise

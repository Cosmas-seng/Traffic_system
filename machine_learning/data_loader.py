from dotenv import load_dotenv
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
TS_FMT = "%Y-%m-%d %H:%M:%S"
OBS_COLS = ["segment_id", "recorded_at",
            "speed_kmh", "vehicle_count", "density"]
load_dotenv(ROOT / ".env")


def fmt_ts(value):
    """Convert a datetime-like value to the database's UTC-naive timestamp format."""
    return pd.Timestamp(value).strftime(TS_FMT)


def classify_traffic(speed):
    """Map speed to the existing traffic_level ENUM.

    These are configurable starter thresholds, not official Nairobi standards.
    Tune them using the actual road speed limits and training data.
    """
    speed = float(speed)
    if speed >= 50:
        return "low"
    if speed >= 30:
        return "moderate"
    if speed >= 15:
        return "heavy"
    return "severe"


class Database:
    def __init__(self, conn, engine="mysql"):
        self.conn = conn
        self.engine = engine.lower()

    @classmethod
    def from_env(cls, path=None):
        """Connect using DB_ENGINE=mysql and DB_HOST/DB_USER/DB_PASSWORD/DB_NAME.

        Loads the project's .env file when python-dotenv is installed.
        MySQL is the default for this project. Set DB_ENGINE=sqlite only if
        intentionally using a separate SQLite database.
        """
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env", override=False)
        except ImportError:
            pass
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
            import sqlite3
            db_path = path or os.environ.get(
                "DATABASE_PATH") or str(ROOT / "traffic.db")
            return cls(sqlite3.connect(db_path), "sqlite")

        raise ValueError("DB_ENGINE must be either 'mysql' or 'sqlite'.")

    def execute(self, sql, params=()):
        cur = self.conn.cursor()
        if self.engine == "mysql":
            sql = sql.replace("?", "%s")
        cur.execute(sql, tuple(params))
        return cur

    def frame(self, sql, params=()):
        cur = self.execute(sql, params)
        columns = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=columns)

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def has_table(self, name):
        if self.engine == "mysql":
            return self.execute(
                """SELECT 1 FROM information_schema.tables
                   WHERE table_schema = DATABASE() AND table_name = ?""",
                (name,),
            ).fetchone() is not None
        return self.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (
                name,)
        ).fetchone() is not None

    def columns(self, table):
        """Return column names for a fixed, internally supplied table name."""
        if self.engine == "mysql":
            rows = self.execute(
                """SELECT column_name FROM information_schema.columns
                   WHERE table_schema = DATABASE() AND table_name = ?""",
                (table,),
            ).fetchall()
            return {r[0] for r in rows}
        rows = self.execute(f"PRAGMA table_info({table})").fetchall()
        return {r[1] for r in rows}

    def ensure_schema(self):
        """Validate existing tables and add only the optional density column.

        The project's existing tables are retained. This does not run a SQLite
        migration and does not modify users, incidents, routes, or predictions.
        """
        if self.engine != "mysql":
            raise RuntimeError(
                "This corrected loader targets the existing MySQL/MariaDB ai_system "
                "database. Set DB_ENGINE=mysql in your .env file."
            )

        required = {
            "road_segments": {
                "segment_id", "road_name", "start_lat", "start_lng",
                "end_lat", "end_lng", "length_km",
            },
            "traffic_records": {
                "traffic_id", "segment_id", "recorded_at", "traffic_level",
                "vehicle_count", "average_speed", "data_source",
            },
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
                    f"Table '{table}' is missing expected columns: "
                    + ", ".join(sorted(missing))
                )

        traffic_cols = self.columns("traffic_records")
        if "density" not in traffic_cols:
            self.execute(
                "ALTER TABLE traffic_records ADD COLUMN density DECIMAL(10,3) NULL"
            )
            self.commit()


def inventory(db):
    """Return row counts for the ML-related existing tables."""
    result = {}
    for table in ("road_segments", "traffic_records", "traffic_predictions", "incidents"):
        result[table] = (
            db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if db.has_table(table) else None
        )
    return result


def load_segments(db):
    """Return the legacy ML field names while reading the existing schema."""
    db.ensure_schema()
    return db.frame(
        """
        SELECT
            segment_id AS id,
            road_name AS name,
            start_lat AS latitude,
            start_lng AS longitude,
            length_km,
            NULL AS free_flow_speed_kmh,
            NULL AS speed_limit_kmh,
            NULL AS geometry_json,
            end_lat,
            end_lng
        FROM road_segments
        ORDER BY segment_id
        """
    )


def load_observations(db, segment_ids=None, start=None, end=None):
    """Load historical observations using aliases expected by the ML pipeline."""
    db.ensure_schema()
    sql = """
        SELECT segment_id, recorded_at,
               average_speed AS speed_kmh,
               vehicle_count, density
        FROM traffic_records
        WHERE 1=1
    """
    params = []
    if segment_ids is not None:
        ids = [int(value) for value in segment_ids]
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
    """Load incidents, mapping the existing reported_at field to created_at."""
    if not db.has_table("incidents"):
        return pd.DataFrame(columns=["latitude", "longitude", "created_at"])
    sql = """
        SELECT latitude, longitude, reported_at AS created_at
        FROM incidents WHERE 1=1
    """
    params = []
    if start is not None:
        sql += " AND reported_at >= ?"
        params.append(fmt_ts(start))
    if end is not None:
        sql += " AND reported_at <= ?"
        params.append(fmt_ts(end))
    return db.frame(sql + " ORDER BY reported_at", params)


def _upsert_segments(db, segments, report):
    """Insert or update road rows using the actual ai_system column names."""
    segments = segments.copy()
    segments.columns = [str(c).strip().lower() for c in segments.columns]

    required = {"name", "start_lat", "start_lng", "end_lat", "end_lng"}
    missing = required - set(segments.columns)
    if missing:
        raise ValueError(
            "roads.csv is missing required columns: " +
            ", ".join(sorted(missing))
        )

    for _, row in segments.iterrows():
        name = str(row["name"]).strip()
        if not name or name.lower() == "nan":
            report["rows_rejected"]["road_missing_name"] = (
                report["rows_rejected"].get("road_missing_name", 0) + 1
            )
            continue

        try:
            start_lat = float(row["start_lat"])
            start_lng = float(row["start_lng"])
            end_lat = float(row["end_lat"])
            end_lng = float(row["end_lng"])
        except (TypeError, ValueError):
            report["rows_rejected"]["road_invalid_coordinates"] = (
                report["rows_rejected"].get("road_invalid_coordinates", 0) + 1
            )
            continue

        length = row.get("length_km")
        length = None if pd.isna(length) else float(length)

        existing = db.execute(
            "SELECT segment_id FROM road_segments WHERE road_name = ? LIMIT 1",
            (name,),
        ).fetchone()

        if existing is None:
            db.execute(
                """INSERT INTO road_segments
                   (road_name, start_lat, start_lng, end_lat, end_lng, length_km)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (name, start_lat, start_lng, end_lat, end_lng, length),
            )
            report["segments_created"] += 1
        else:
            db.execute(
                """UPDATE road_segments
                   SET start_lat=?, start_lng=?, end_lat=?, end_lng=?,
                       length_km=COALESCE(?, length_km)
                   WHERE segment_id=?""",
                (start_lat, start_lng, end_lat, end_lng, length, existing[0]),
            )
    db.commit()


def import_traffic_csv(db, csv_path, source, utc_offset_hours=0, segments_csv=None):
    """Import measured CSV observations into the existing traffic_records table.

    traffic.csv required columns:
      recorded_at, segment_name, speed_kmh
    Optional:
      vehicle_count, density

    Naive timestamps are interpreted in the supplied local UTC offset and stored
    as UTC. For Nairobi local time, use utc_offset_hours=3.
    Duplicate definition: same segment_id + recorded_at + data_source.
    """
    db.ensure_schema()
    report = {
        "rows_read": 0,
        "rows_rejected": {},
        "rows_inserted": 0,
        "duplicates_skipped": 0,
        "segments_created": 0,
    }

    try:
        if segments_csv:
            roads = pd.read_csv(segments_csv)
            _upsert_segments(db, roads, report)

        df = pd.read_csv(csv_path, dtype=str)
        df.columns = [str(c).strip().lower() for c in df.columns]
        report["rows_read"] = len(df)

        for required_col in ("recorded_at", "speed_kmh"):
            if required_col not in df.columns:
                raise ValueError(
                    f"traffic.csv is missing required column '{required_col}'.")
        if "segment_name" not in df.columns:
            raise ValueError(
                "traffic.csv must contain a 'segment_name' column.")

        def reject(mask, reason):
            count = int(mask.sum())
            if count:
                report["rows_rejected"][reason] = (
                    report["rows_rejected"].get(reason, 0) + count
                )
            return df.loc[~mask].copy()

        timestamps = pd.to_datetime(
            df["recorded_at"], errors="coerce", format="mixed"
        )
        # Mixed timezone-aware and naive CSVs are not silently combined.
        if getattr(timestamps.dt, "tz", None) is not None:
            timestamps = timestamps.dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            timestamps = timestamps - \
                pd.Timedelta(hours=float(utc_offset_hours))
        df["tmp_ts"] = timestamps
        df = reject(df["tmp_ts"].isna(), "invalid_timestamp")

        df["tmp_speed"] = pd.to_numeric(df["speed_kmh"], errors="coerce")
        df = reject(df["tmp_speed"].isna() | (
            df["tmp_speed"] < 0), "invalid_speed")

        for column in ("vehicle_count", "density"):
            if column in df.columns:
                df["tmp_" +
                    column] = pd.to_numeric(df[column], errors="coerce")
            else:
                df["tmp_" + column] = float("nan")

        df["tmp_name"] = df["segment_name"].fillna("").astype(str).str.strip()
        df = reject(df["tmp_name"].eq(""), "missing_segment_name")

        names = df["tmp_name"].drop_duplicates().tolist()
        segment_map = {}
        for name in names:
            found = db.execute(
                "SELECT segment_id FROM road_segments WHERE road_name = ? LIMIT 1",
                (name,),
            ).fetchone()
            if found is not None:
                segment_map[name] = int(found[0])

        df["tmp_sid"] = df["tmp_name"].map(segment_map)
        df = reject(df["tmp_sid"].isna(), "unknown_segment")

        # Insert row-by-row so duplicates can be detected without adding a
        # potentially disruptive unique index to a populated production table.
        for row in df.itertuples(index=False):
            segment_id = int(row.tmp_sid)
            recorded_at = row.tmp_ts.strftime(TS_FMT)
            speed = float(row.tmp_speed)
            vehicle_count = (
                None if pd.isna(row.tmp_vehicle_count) else int(
                    float(row.tmp_vehicle_count))
            )
            density = None if pd.isna(
                row.tmp_density) else float(row.tmp_density)

            duplicate = db.execute(
                """SELECT traffic_id FROM traffic_records
                   WHERE segment_id=? AND recorded_at=? AND data_source=?
                   LIMIT 1""",
                (segment_id, recorded_at, source),
            ).fetchone()
            if duplicate is not None:
                report["duplicates_skipped"] += 1
                continue

            db.execute(
                """INSERT INTO traffic_records
                   (segment_id, recorded_at, traffic_level, vehicle_count,
                    average_speed, data_source, density)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    segment_id,
                    recorded_at,
                    classify_traffic(speed),
                    vehicle_count,
                    speed,
                    source,
                    density,
                ),
            )
            report["rows_inserted"] += 1

        db.commit()
        return report
    except Exception:
        db.rollback()
        raise

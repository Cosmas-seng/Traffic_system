
import os
import sqlite3
import secrets
import pymysql
from pymysql.constants import CLIENT
from pymysql.cursors import DictCursor
import sys
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps
from getpass import getpass

from flask import (Flask, abort, flash, g, jsonify, redirect, render_template,
                   request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from machine_learning import data_loader as ml_data
from machine_learning import model_evaluation as ml_evaluation
from machine_learning.prediction_service import PredictionService

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}
MAGIC = {  # file signatures, checked in addition to the extension
    b"\x89PNG\r\n\x1a\n": "png",
    b"\xff\xd8\xff": "jpg",
    b"RIFF": "webp",
}
# Must match the incidents.incident_type ENUM in the database.
INCIDENT_TYPES = {"accident": "Accident", "road_closure": "Road closure",
                  "hazard": "Road hazard", "breakdown": "Vehicle breakdown",
                  "construction": "Construction", "other": "Other"}
# The API/templates say "admin"; the users.role ENUM says "administrator".
ROLE_TO_DB = {"admin": "administrator",
              "driver": "driver", "commuter": "commuter"}
ROLE_FROM_DB = {v: k for k, v in ROLE_TO_DB.items()}

app = Flask(__name__)
PREDICTOR = PredictionService()
ROUTING_ENGINE = None
# SECRET_KEY must come from the environment in production. A random per-process
# key is used for local development only (sessions reset on restart).
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY") or secrets.token_hex(32),
    MAX_CONTENT_LENGTH=5 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
)

DB_CFG = {
    "host": os.environ.get("DB_HOST", "127.0.0.1"),
    "port": int(os.environ.get("DB_PORT", "3306")),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", ""),
    "database": os.environ.get("DB_NAME", "ai_system"),
    "charset": "utf8mb4",
    "cursorclass": DictCursor,
    "connect_timeout": 5,
    # rowcount = rows matched (not only changed), so re-marking a notification
    # as read is not reported as "not found".
    "client_flag": CLIENT.FOUND_ROWS,
}

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ------------------------------------------------------------------ database


class MySQLDatabase:
    """Small adapter that supports the app's db.execute(...).fetchone() pattern."""

    def __init__(self, connection):
        self.connection = connection

    def execute(self, sql, params=()):
        # Convert SQLite-style placeholders left in this file to PyMySQL placeholders.
        cursor = self.connection.cursor()
        if isinstance(self.connection, sqlite3.Connection):
            cursor.execute(sql, params)
        else:
            cursor.execute(sql.replace("?", "%s"), params)
        return cursor

    def commit(self):
        self.connection.commit()

    def rollback(self):
        self.connection.rollback()

    def close(self):
        self.connection.close()


def get_db():
    if "db" not in g:
        test_path = app.config.get("DATABASE") or os.environ.get("DATABASE_PATH")
        if test_path:
            conn = sqlite3.connect(test_path)
            conn.row_factory = sqlite3.Row
            g.db = MySQLDatabase(conn)
        else:
            g.db = MySQLDatabase(pymysql.connect(**DB_CFG))
    return g.db


def ml_db():
    """Open the traffic database through the ML adapter, honoring test/local SQLite config."""
    path = app.config.get("DATABASE") or os.environ.get("DATABASE_PATH")
    if path:
        return ml_data.Database(sqlite3.connect(path), engine="sqlite")
    return ml_data.Database.from_env()


def init_db():
    """Initialize the app-owned tables in the configured local/test SQLite DB."""
    path = app.config.get("DATABASE") or os.environ.get("DATABASE_PATH")
    if not path:
        raise RuntimeError("DATABASE/DATABASE_PATH must point to a SQLite file")
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT, full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL,
                role TEXT NOT NULL, created_at TEXT NOT NULL, is_active INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS incidents (
                incident_id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER,
                incident_type TEXT, location_text TEXT, latitude REAL, longitude REAL,
                description TEXT, image_file TEXT, reported_at TEXT, created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS notifications (
                notification_id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
                incident_id INTEGER, message TEXT NOT NULL, is_read INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
        """)


@app.teardown_appcontext
def close_db(_exc):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def current_user():
    uid = session.get("uid")
    if not uid:
        return None
    # Always re-read from the DB: never trust role data stored client-side.
    row = get_db().execute("SELECT * FROM users WHERE id=? ", (uid,)).fetchone()
    if row is None:
        session.clear()
        return None
    return norm_user(row)


def norm_user(row):
    """Translate the DB role value ('administrator') to the app's ('admin')."""
    row = dict(row)
    row["role"] = ROLE_FROM_DB.get(row["role"], row["role"])
    return row


# ------------------------------------------------------------ auth & security
def login_required(view):
    @wraps(view)
    def wrapped(*a, **kw):
        if current_user() is None:
            if request.path.startswith("/api/"):
                return jsonify(error="Authentication required."), 401
            return redirect(url_for("login", next=request.path))
        return view(*a, **kw)
    return wrapped


def admin_required(view):
    @wraps(view)
    @login_required
    def wrapped(*a, **kw):
        if current_user()["role"] != "admin":
            if request.path.startswith("/api/"):
                return jsonify(error="Administrator access required."), 403
            abort(403)
        return view(*a, **kw)
    return wrapped


def csrf_token():
    if "csrf" not in session:
        session["csrf"] = secrets.token_hex(16)
    return session["csrf"]


@app.before_request
def csrf_protect():
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        sent = request.headers.get(
            "X-CSRF-Token") or request.form.get("csrf_token")
        if not sent or not secrets.compare_digest(sent, session.get("csrf", "")):
            if request.path.startswith("/api/"):
                return jsonify(error="Invalid or missing CSRF token."), 400
            abort(400, "Invalid or missing CSRF token.")


@app.after_request
def security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    # Must send the origin to cross-site requests: OpenStreetMap's tile servers
    # reject requests that carry no Referer ("same-origin" broke the map).
    resp.headers.setdefault(
        "Referrer-Policy", "strict-origin-when-cross-origin")
    return resp


@app.context_processor
def inject_globals():
    return {"csrf_token": csrf_token, "current_user": current_user(),
            "incident_types": INCIDENT_TYPES}


def valid_email(v):
    return bool(v) and len(v) <= 254 and "@" in v and "." in v.split("@")[-1] and " " not in v


def password_problem(pw):
    if len(pw) < 8:
        return "Password must be at least 8 characters."
    if pw.isalpha() or pw.isdigit():
        return "Password must mix letters and numbers."
    return None


# --------------------------------------------------------------------- pages
@app.route("/")
def index():
    return redirect(url_for("dashboard" if current_user() else "login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        pw = request.form.get("password", "")
        row = get_db().execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if row and check_password_hash(row["password_hash"], pw):
            session.clear()
            session.permanent = True
            session["uid"] = row["id"]
            nxt = request.args.get("next", "")
            # Only allow same-site relative redirects (prevents open redirect).
            if nxt.startswith("/") and not nxt.startswith("//"):
                return redirect(nxt)
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "danger")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        pw = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        error = None
        if not (2 <= len(name) <= 100):
            error = "Full name must be 2-100 characters."
        elif not valid_email(email):
            error = "Enter a valid email address."
        elif password_problem(pw):
            error = password_problem(pw)
        elif pw != confirm:
            error = "Passwords do not match."
        if error:
            flash(error, "danger")
            return render_template("register.html", form=request.form), 400
        try:
            # Role is NEVER read from the form: public sign-ups are always 'commuter'.
            get_db().execute(
                "INSERT INTO users(full_name,email,password_hash,role,created_at) VALUES(?,?,?,?,?)",
                (name, email, generate_password_hash(pw), "commuter", now_iso()))
            get_db().commit()
        except pymysql.err.IntegrityError:
            flash("An account with that email already exists.", "danger")
            return render_template("register.html", form=request.form), 400
        flash("Account created. You can log in now.", "success")
        return redirect(url_for("login"))
    return render_template("register.html", form={})


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


def page(template, active, title, **ctx):
    return render_template(template, active=active, page_title=title, **ctx)


@app.route("/dashboard")
@login_required
def dashboard():
    return page("dashboard.html", "dashboard", "Live Traffic Map")


@app.route("/route-planner")
@login_required
def route_planner():
    return page("route_planner.html", "route_planner", "Route Planner")


@app.route("/report-incident")
@login_required
def report_incident():
    return page("report_incident.html", "report_incident", "Report an Incident")


@app.route("/predictions")
@login_required
def predictions():
    return page("predictions.html", "predictions", "Traffic Predictions")


@app.route("/notifications")
@login_required
def notifications():
    return page("notifications.html", "notifications", "Notifications")


@app.route("/reports")
@login_required
def reports():
    return page("reports.html", "reports", "Traffic & System Reports")


@app.route("/profile")
@login_required
def profile():
    return page("profile.html", "profile", "My Profile")


@app.route("/admin")
@admin_required
def admin():
    return page("admin.html", "admin", "Users")


@app.errorhandler(403)
def forbidden(_e):
    return render_template("login.html", forbidden=True), 403


# MySQL error codes that mean "cannot connect / not allowed in" (a real outage
# or misconfiguration), as opposed to a bug in a query.
DB_CONNECTION_ERRORS = {1040, 1044, 1045, 1049, 2002, 2003, 2005, 2006, 2013}


@app.errorhandler(pymysql.err.MySQLError)
def db_error(e):
    """The real reason (code + driver message) goes to the server log only,
    never to the browser."""
    code = e.args[0] if e.args else None
    app.logger.error("Database error %s (host=%s port=%s user=%s db=%s)",
                     e.args, DB_CFG["host"], DB_CFG["port"],
                     DB_CFG["user"], DB_CFG["database"])
    if code in DB_CONNECTION_ERRORS:
        msg, status = "Database unavailable. Please try again shortly.", 503
    else:
        msg, status = "A database error occurred. Please contact the administrator.", 500
    if request.path.startswith("/api/"):
        return jsonify(error=msg), status
    return msg, status


# ----------------------------------------------------------------- JSON API
def iso(v):
    """DATETIME columns come back as datetime objects; send 'YYYY-MM-DD HH:MM:SS'
    (UTC) as documented, instead of Flask's default RFC 822 format."""
    return v.strftime("%Y-%m-%d %H:%M:%S") if hasattr(v, "strftime") else v


def num(v):
    """DECIMAL columns come back as Decimal (serialized as strings); Leaflet
    needs real numbers."""
    return float(v) if v is not None else None


def row_user(r):
    return {"id": r["id"], "full_name": r["full_name"], "email": r["email"],
            "role": r["role"], "created_at": iso(r["created_at"])}


def row_incident(r):
    t = r["incident_type"]
    return {"id": r["incident_id"], "type": t, "type_label": INCIDENT_TYPES.get(t, t),
            "location_text": r["location_text"], "latitude": num(r["latitude"]),
            "longitude": num(r["longitude"]), "description": r["description"],
            "has_image": bool(r["image_file"]), "status": r["status"],
            "created_at": iso(r["reported_at"])}


@app.get("/api/traffic")
@login_required
def api_traffic():
    """Traffic levels need a live data provider (not connected). We still report
    real incident counts from the database."""
    db = get_db()
    since = (datetime.now(timezone.utc) - timedelta(hours=24)
             ).strftime("%Y-%m-%d %H:%M:%S")
    n = db.execute(
        "SELECT COUNT(*) c FROM incidents WHERE reported_at>=?", (since,)).fetchone()["c"]
    return jsonify(available=False, provider=None,
                   message="No live traffic provider is connected. Congestion levels, delay "
                           "and segments will appear here once one is integrated.",
                   incidents_24h=n, segments=[], summary=None, updated_at=now_iso() + " UTC")


@app.get("/api/incidents")
@login_required
def api_incidents():
    rows = get_db().execute(
        "SELECT * FROM incidents ORDER BY incident_id DESC LIMIT 200").fetchall()
    return jsonify(incidents=[row_incident(r) for r in rows])


def sniff_image(head):
    for sig, kind in MAGIC.items():
        if head.startswith(sig):
            if kind == "webp" and head[8:12] != b"WEBP":
                continue
            return kind
    return None


@app.post("/api/incidents")
@login_required
def api_incidents_post():
    f = request.form
    itype = f.get("type", "")
    loc = f.get("location_text", "").strip()
    desc = f.get("description", "").strip()
    errors = {}
    if itype not in INCIDENT_TYPES:
        errors["type"] = "Choose a valid incident type."
    if not (3 <= len(loc) <= 200):
        errors["location_text"] = "Location must be 3-200 characters."
    if len(desc) > 1000:
        errors["description"] = "Description is limited to 1000 characters."
    try:
        lat, lng = float(f.get("latitude", "")), float(f.get("longitude", ""))
        # Rough bounding box around Nairobi and surroundings.
        if not (-1.5 <= lat <= -1.0 and 36.6 <= lng <= 37.2):
            errors["location"] = "Pick a location inside the Nairobi area."
    except ValueError:
        errors["location"] = "Click the map to choose the incident location."
    saved = None
    img = request.files.get("image")
    if img and img.filename:
        ext = secure_filename(img.filename).rsplit(".", 1)[-1].lower()
        kind = sniff_image(img.stream.read(16))
        img.stream.seek(0)
        if ext not in ALLOWED_EXT or kind is None:
            errors["image"] = "Upload a PNG, JPG or WebP image."
        else:
            saved = f"{uuid.uuid4().hex}.{'jpg' if kind == 'jpg' else kind}"
    if errors:
        return jsonify(error="Please correct the highlighted fields.", fields=errors), 422
    if saved:
        img.save(os.path.join(UPLOAD_DIR, saved))
    db = get_db()
    cur = db.execute(
        "INSERT INTO incidents(user_id,incident_type,location_text,latitude,longitude,description,image_file,reported_at)"
        " VALUES(?,?,?,?,?,?,?,?)",
        (current_user()["id"], itype, loc, lat, lng, desc or None, saved, now_iso()))
    incident_id = cur.lastrowid
    # notifications.user_id is NOT NULL (no broadcast rows), so alert every user
    # with one row each. Same transaction as the incident: both or neither.
    db.execute("INSERT INTO notifications(user_id,incident_id,message,created_at) "
               "SELECT id,?,?,? FROM users",
               (incident_id, f"{INCIDENT_TYPES[itype]} reported: {loc}", now_iso()))
    db.commit()
    row = db.execute("SELECT * FROM incidents WHERE incident_id=?",
                     (incident_id,)).fetchone()
    return jsonify(incident=row_incident(row)), 201


@app.post("/api/routes")
@login_required
def api_routes():
    data = request.get_json(silent=True) or {}
    if not str(data.get("origin", "")).strip() or not str(data.get("destination", "")).strip():
        return jsonify(error="Both a start and a destination are required."), 422
    if ROUTING_ENGINE is None:
        return jsonify(available=False, routes=[],
                       error="Route optimisation is not connected yet. Connect a routing engine "
                             "to POST /api/routes to see results."), 501
    try:
        db = ml_db()
        candidates = ROUTING_ENGINE(data["origin"], data["destination"])
        from services import routing_service
        catalogue = PREDICTOR.catalogue(db)
        speeds = {}
        for seg in catalogue:
            pred = PREDICTOR.predict(db, seg["id"], PREDICTOR.horizons()[0]) if PREDICTOR.horizons() else {}
            if pred.get("available"):
                speeds[seg["id"]] = pred["predicted_speed_kmh"]
        ranked = routing_service.rank_routes(candidates, routing_service.speed_lookup([], speeds))
        return jsonify(available=True, routes=ranked)
    except Exception:
        app.logger.exception("Route planning failed")
        return jsonify(error="Route planning failed."), 500


@app.get("/api/predictions")
@login_required
def api_predictions():
    try:
        db = ml_db()
        location = request.args.get("location")
        horizon = request.args.get("horizon")
        if location is None and horizon is None:
            status = PREDICTOR.status(db)
            locations = PREDICTOR.catalogue(db) if status["horizons"] else []
            return jsonify(available=bool(status["horizons"]), locations=locations,
                           message="No trained model is deployed. Run train_model to create one." if not status["horizons"] else None, **status)
        if location is None or horizon is None or not location.isdecimal() or not horizon.isdecimal() or int(location) < 1 or int(horizon) < 1:
            return jsonify(error="location and horizon must be positive integers."), 422
        return jsonify(PREDICTOR.predict(db, int(location), int(horizon)))
    except Exception:
        app.logger.exception("Prediction request failed")
        return jsonify(error="Prediction request failed."), 500
    finally:
        if "db" in locals(): db.close()


@app.get("/api/predictions/<int:segment_id>")
@login_required
def api_prediction_one(segment_id):
    horizon = request.args.get("horizon")
    if horizon is None or not horizon.isdecimal() or int(horizon) < 1:
        return jsonify(error="horizon must be a positive integer."), 422
    try:
        db = ml_db()
        return jsonify(PREDICTOR.predict(db, segment_id, int(horizon)))
    except Exception:
        app.logger.exception("Prediction request failed")
        return jsonify(error="Prediction request failed."), 500
    finally:
        if "db" in locals(): db.close()


@app.get("/api/predictions/<int:segment_id>/history")
@login_required
def api_prediction_history(segment_id):
    try:
        db = ml_db()
        return jsonify(kind="model_predictions", predictions=PREDICTOR.history(db, segment_id))
    except Exception:
        app.logger.exception("Prediction history request failed")
        return jsonify(error="Prediction history request failed."), 500
    finally:
        if "db" in locals(): db.close()


@app.get("/api/models/status")
@login_required
def api_models_status():
    try:
        db = ml_db()
        return jsonify(PREDICTOR.status(db))
    except Exception:
        app.logger.exception("Model status request failed")
        return jsonify(error="Model status request failed."), 500
    finally:
        if "db" in locals(): db.close()


@app.post("/api/predictions/generate")
@admin_required
def api_predictions_generate():
    data = request.get_json(silent=True) or {}
    if not data:
        try:
            db = ml_db()
            return jsonify(PREDICTOR.generate_all(db))
        except Exception:
            app.logger.exception("Prediction generation failed")
            return jsonify(error="Prediction generation failed."), 500
        finally:
            if "db" in locals(): db.close()
    road_id, horizon = data.get("road_id"), data.get("horizon")
    if (isinstance(road_id, bool) or not isinstance(road_id, int) or road_id < 1 or
            isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1):
        return jsonify(error="road_id and horizon must be positive integers."), 422
    try:
        db = ml_db()
        return jsonify(PREDICTOR.predict(db, road_id, horizon, persist=True))
    except Exception:
        app.logger.exception("Prediction generation failed")
        return jsonify(error="Prediction generation failed."), 500
    finally:
        if "db" in locals(): db.close()


@app.get("/api/models/metrics")
@admin_required
def api_models_metrics():
    try:
        db = ml_db()
        realized = ml_evaluation.realized_error(db)
        from machine_learning import model_registry as registry
        from machine_learning import model_config
        import json
        from pathlib import Path
        reports = {}
        for horizon in PREDICTOR.horizons():
            version = registry.current_version(horizon, PREDICTOR.models_dir)
            report_path = (Path(PREDICTOR.models_dir or model_config.MODELS_DIR) /
                           f"h{horizon}" / version / "metrics.json")
            report = json.loads(report_path.read_text())
            reports[str(horizon)] = {"version": version,
                                     "test": report.get("test", [])}
        return jsonify(horizons=reports, realized=realized)
    except Exception:
        app.logger.exception("Model metrics request failed")
        return jsonify(error="Model metrics request failed."), 500
    finally:
        if "db" in locals(): db.close()


@app.get("/api/notifications")
@login_required
def api_notifications():
    rows = get_db().execute(
        "SELECT n.notification_id, n.incident_id, n.message, n.is_read, n.created_at, "
        "i.incident_type FROM notifications n "
        "LEFT JOIN incidents i ON i.incident_id=n.incident_id "
        "WHERE n.user_id=? ORDER BY n.notification_id DESC LIMIT 100",
        (current_user()["id"],)).fetchall()
    # The table has no category/title columns, so derive them for the frontend.
    return jsonify(notifications=[{
        "id": r["notification_id"],
        "category": "incident" if r["incident_id"] else "system",
        "title": (f"{INCIDENT_TYPES.get(r['incident_type'], r['incident_type'])} reported"
                  if r["incident_type"] else "Notification"),
        "message": r["message"], "is_read": bool(r["is_read"]), "created_at": iso(r["created_at"])} for r in rows])


@app.post("/api/notifications/<int:nid>/read")
@login_required
def api_notification_read(nid):
    db = get_db()
    cur = db.execute("UPDATE notifications SET is_read=1 WHERE notification_id=? AND user_id=?",
                     (nid, current_user()["id"]))
    db.commit()
    if cur.rowcount == 0:
        return jsonify(error="Notification not found."), 404
    return jsonify(ok=True)


@app.get("/api/profile")
@login_required
def api_profile():
    return jsonify(profile=row_user(current_user()))


@app.put("/api/profile")
@login_required
def api_profile_put():
    data = request.get_json(silent=True) or {}
    name = str(data.get("full_name", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    if not (2 <= len(name) <= 100):
        return jsonify(error="Full name must be 2-100 characters."), 422
    if not valid_email(email):
        return jsonify(error="Enter a valid email address."), 422
    try:  # only name/email are editable; role/status are not accepted here
        get_db().execute("UPDATE users SET full_name=?, email=? WHERE id=?",
                         (name, email, current_user()["id"]))
        get_db().commit()
    except pymysql.err.IntegrityError:
        return jsonify(error="That email is already in use."), 409
    return jsonify(profile=row_user(current_user()))


@app.post("/api/profile/password")
@login_required
def api_profile_password():
    data = request.get_json(silent=True) or {}
    user = current_user()
    if not check_password_hash(user["password_hash"], str(data.get("current_password", ""))):
        return jsonify(error="Current password is incorrect."), 403
    new = str(data.get("new_password", ""))
    if password_problem(new):
        return jsonify(error=password_problem(new)), 422
    get_db().execute("UPDATE users SET password_hash=? WHERE id=?",
                     (generate_password_hash(new), user["id"]))
    get_db().commit()
    return jsonify(ok=True)


# -- admin: user management (every route enforced server-side by admin_required)
@app.get("/api/admin/users")
@admin_required
def api_admin_users():
    q = request.args.get("q", "").strip()
    role = request.args.get("role", "")
    page_n = max(request.args.get("page", 1, type=int), 1)
    per = 10
    where, args = ["1=1"], []
    if q:
        where.append("(full_name LIKE ? OR email LIKE ?)")
        args += [f"%{q}%", f"%{q}%"]
    if role in ("admin", "driver", "commuter"):
        where.append("role=?")
        args.append(ROLE_TO_DB[role])
    db = get_db()
    total = db.execute(
        f"SELECT COUNT(*) c FROM users WHERE {' AND '.join(where)}", args).fetchone()["c"]
    rows = db.execute(f"SELECT * FROM users WHERE {' AND '.join(where)} ORDER BY id LIMIT ? OFFSET ?",
                      args + [per, (page_n - 1) * per]).fetchall()
    return jsonify(users=[row_user(r) for r in rows], total=total, page=page_n, per_page=per)


@app.post("/api/admin/users")
@admin_required
def api_admin_users_post():
    d = request.get_json(silent=True) or {}
    name, email = str(d.get("full_name", "")).strip(), str(
        d.get("email", "")).strip().lower()
    role, pw = d.get("role", "commuter"), str(d.get("password", ""))
    if not (2 <= len(name) <= 100) or not valid_email(email) or role not in ("admin", "driver", "commuter"):
        return jsonify(error="Provide a valid name, email and role."), 422
    if password_problem(pw):
        return jsonify(error=password_problem(pw)), 422
    try:
        get_db().execute("INSERT INTO users(full_name,email,password_hash,role,created_at) VALUES(?,?,?,?,?)",
                         (name, email, generate_password_hash(pw), ROLE_TO_DB[role], now_iso()))
        get_db().commit()
    except pymysql.err.IntegrityError:
        return jsonify(error="That email is already in use."), 409
    return jsonify(ok=True), 201


@app.put("/api/admin/users/<int:uid>")
@admin_required
def api_admin_users_put(uid):
    d = request.get_json(silent=True) or {}
    db = get_db()
    target = db.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    if target is None:
        return jsonify(error="User not found."), 404
    norm_user(target)
    name = str(d.get("full_name", target["full_name"])).strip()
    role = d.get("role", target["role"])
    if not (2 <= len(name) <= 100) or role not in ("admin", "driver", "commuter"):
        return jsonify(error="Invalid name or role."), 422
    if uid == current_user()["id"] and (role != "admin"):
        return jsonify(error="You cannot demote or deactivate your own account."), 400
    db.execute("UPDATE users SET full_name=?, role=? WHERE id=?",
               (name, ROLE_TO_DB[role], uid))
    db.commit()
    return jsonify(ok=True)


@app.get("/api/reports")
@login_required
def api_reports():
    """Real incident statistics from the database for the chosen date range."""
    start = request.args.get("from", "")
    end = request.args.get("to", "")
    try:
        d0 = datetime.strptime(start, "%Y-%m-%d")
        d1 = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)
    except ValueError:
        return jsonify(error="Provide valid 'from' and 'to' dates (YYYY-MM-DD)."), 422
    if d1 <= d0:
        return jsonify(error="The end date must not be before the start date."), 422
    db = get_db()
    a, b = d0.strftime("%Y-%m-%d"), d1.strftime("%Y-%m-%d")
    total = db.execute(
        "SELECT COUNT(*) c FROM incidents WHERE reported_at>=? AND reported_at<?", (a, b)).fetchone()["c"]
    by_type = db.execute("SELECT incident_type AS type, COUNT(*) c FROM incidents "
                         "WHERE reported_at>=? AND reported_at<? GROUP BY incident_type",
                         (a, b)).fetchall()
    by_day = db.execute("SELECT DATE(reported_at) d, COUNT(*) c FROM incidents "
                        "WHERE reported_at>=? AND reported_at<? GROUP BY d ORDER BY d", (a, b)).fetchall()
    reporters = db.execute("SELECT COUNT(DISTINCT user_id) c FROM incidents WHERE reported_at>=? AND reported_at<?",
                           (a, b)).fetchone()["c"]
    hot = db.execute("SELECT location_text, COUNT(*) c FROM incidents WHERE reported_at>=? AND reported_at<? "
                     "AND location_text IS NOT NULL GROUP BY location_text ORDER BY c DESC LIMIT 1",
                     (a, b)).fetchone()
    return jsonify(
        total_incidents=total, active_reporters=reporters,
        most_reported_location=hot["location_text"] if hot else None,
        by_type=[{"type": r["type"], "label": INCIDENT_TYPES.get(
            r["type"], r["type"]), "count": r["c"]} for r in by_type],
        by_day=[{"date": str(r["d"]), "count": r["c"]} for r in by_day],
        traffic_trend={"available": False,
                       "message": "Congestion trends and travel-time averages need a live traffic data provider."})


# --------------------------------------------------------------------- CLI
def create_admin():
    """python app.py create-admin  -> prompts for details, password never echoed."""
    name = input("Full name: ").strip()
    email = input("Email: ").strip().lower()
    pw = getpass("Password: ")
    if not valid_email(email) or password_problem(pw):
        sys.exit(password_problem(pw) or "Invalid email.")
    try:
        with app.app_context():
            db = get_db()
            db.execute(
                "INSERT INTO users(full_name,email,password_hash,role,created_at) "
                "VALUES(?,?,?,?,?)",
                (name, email, generate_password_hash(pw), "administrator", now_iso()))
            db.commit()
    except pymysql.err.IntegrityError:
        sys.exit("An account with that email already exists.")
    print("Administrator created.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "create-admin":
        create_admin()
    else:
        app.run(debug=os.environ.get("FLASK_DEBUG") == "1")

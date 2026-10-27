"""
Nairobi Traffic System - Flask application.

No existing backend was found in the workspace, so this file provides a
minimal, working reference backend for the frontend. It uses SQLite so it runs
anywhere. All SQL is isolated in the `db_*` helpers and the `get_db()` function,
so you can point them at your existing MySQL/MariaDB database (PyMySQL) without
touching templates or JavaScript.

Honest-by-design: live traffic, route optimisation and AI predictions need
external providers/models. Those endpoints respond with {"available": false}
(or HTTP 501) until you connect them; nothing is faked.
"""
import os
import secrets
import sqlite3
import sys
import uuid
from datetime import datetime, timedelta
from functools import wraps
from getpass import getpass

from flask import (Flask, abort, flash, g, jsonify, redirect, render_template,
                   request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}
MAGIC = {  # file signatures, checked in addition to the extension
    b"\x89PNG\r\n\x1a\n": "png",
    b"\xff\xd8\xff": "jpg",
    b"RIFF": "webp",
}
INCIDENT_TYPES = {"accident": "Accident", "road_closure": "Road closure",
                  "hazard": "Road hazard", "heavy_congestion": "Heavy congestion"}

app = Flask(__name__)
# SECRET_KEY must come from the environment in production. A random per-process
# key is used for local development only (sessions reset on restart).
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY") or secrets.token_hex(32),
    DATABASE=os.environ.get(
        "DATABASE_PATH", os.path.join(BASE_DIR, "traffic.db")),
    MAX_CONTENT_LENGTH=5 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
)
os.makedirs(UPLOAD_DIR, exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  full_name TEXT NOT NULL,
  email TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'commuter',
  is_active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS incidents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL REFERENCES users(id),
  type TEXT NOT NULL,
  location_text TEXT NOT NULL,
  latitude REAL NOT NULL,
  longitude REAL NOT NULL,
  description TEXT,
  image_file TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notifications (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER,            -- NULL = broadcast to everyone
  category TEXT NOT NULL,     -- congestion | incident | route
  title TEXT NOT NULL,
  message TEXT,
  is_read INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);
"""

# ------------------------------------------------------------------ database


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    with sqlite3.connect(app.config["DATABASE"]) as con:
        con.executescript(SCHEMA)


def now_iso():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def current_user():
    uid = session.get("uid")
    if not uid:
        return None
    # Always re-read from the DB: never trust role data stored client-side.
    row = get_db().execute("SELECT * FROM users WHERE id=? AND is_active=1", (uid,)).fetchone()
    if row is None:
        session.clear()
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
        if row and row["is_active"] and check_password_hash(row["password_hash"], pw):
            session.clear()
            session.permanent = True
            session["uid"] = row["id"]
            nxt = request.args.get("next", "")
            # Only allow same-site relative redirects (prevents open redirect).
            if nxt.startswith("/") and not nxt.startswith("//"):
                return redirect(nxt)
            return redirect(url_for("dashboard"))
        flash("Invalid email or password, or the account is deactivated.", "danger")
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
        except sqlite3.IntegrityError:
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


# ----------------------------------------------------------------- JSON API
def row_user(r):
    return {"id": r["id"], "full_name": r["full_name"], "email": r["email"],
            "role": r["role"], "is_active": bool(r["is_active"]), "created_at": r["created_at"]}


def row_incident(r):
    return {"id": r["id"], "type": r["type"], "type_label": INCIDENT_TYPES.get(r["type"], r["type"]),
            "location_text": r["location_text"], "latitude": r["latitude"],
            "longitude": r["longitude"], "description": r["description"],
            "has_image": bool(r["image_file"]), "created_at": r["created_at"]}


@app.get("/api/traffic")
@login_required
def api_traffic():
    """Traffic levels need a live data provider (not connected). We still report
    real incident counts from the database."""
    db = get_db()
    since = (datetime.utcnow() - timedelta(hours=24)
             ).strftime("%Y-%m-%d %H:%M:%S")
    n = db.execute(
        "SELECT COUNT(*) c FROM incidents WHERE created_at>=?", (since,)).fetchone()["c"]
    return jsonify(available=False, provider=None,
                   message="No live traffic provider is connected. Congestion levels, delay "
                           "and segments will appear here once one is integrated.",
                   incidents_24h=n, segments=[], summary=None, updated_at=now_iso() + " UTC")


@app.get("/api/incidents")
@login_required
def api_incidents():
    rows = get_db().execute("SELECT * FROM incidents ORDER BY id DESC LIMIT 200").fetchall()
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
        "INSERT INTO incidents(user_id,type,location_text,latitude,longitude,description,image_file,created_at)"
        " VALUES(?,?,?,?,?,?,?,?)",
        (current_user()["id"], itype, loc, lat, lng, desc or None, saved, now_iso()))
    db.execute("INSERT INTO notifications(user_id,category,title,message,created_at) VALUES(NULL,'incident',?,?,?)",
               (f"{INCIDENT_TYPES[itype]} reported", loc, now_iso()))
    db.commit()
    row = db.execute("SELECT * FROM incidents WHERE id=?",
                     (cur.lastrowid,)).fetchone()
    return jsonify(incident=row_incident(row)), 201


@app.post("/api/routes")
@login_required
def api_routes():
    data = request.get_json(silent=True) or {}
    if not str(data.get("origin", "")).strip() or not str(data.get("destination", "")).strip():
        return jsonify(error="Both a start and a destination are required."), 422
    # Needs a routing engine (OSRM, GraphHopper, Valhalla...) plus traffic data.
    return jsonify(available=False, routes=[],
                   error="Route optimisation is not connected yet. Connect a routing engine "
                         "to POST /api/routes to see results."), 501


@app.get("/api/predictions")
@login_required
def api_predictions():
    # Needs a trained model. Contract the frontend expects when connected:
    # {available:true, road, horizon_minutes, current_level, predicted_level,
    #  estimated_delay_min, generated_at, model:{name,version}, history:[{t,value}], forecast:[{t,value}]}
    return jsonify(available=False, model_status="not_connected", locations=[],
                   message="No traffic prediction model is connected.")


@app.get("/api/notifications")
@login_required
def api_notifications():
    rows = get_db().execute(
        "SELECT * FROM notifications WHERE user_id IS NULL OR user_id=? ORDER BY id DESC LIMIT 100",
        (current_user()["id"],)).fetchall()
    return jsonify(notifications=[{
        "id": r["id"], "category": r["category"], "title": r["title"],
        "message": r["message"], "is_read": bool(r["is_read"]), "created_at": r["created_at"]} for r in rows])


@app.post("/api/notifications/<int:nid>/read")
@login_required
def api_notification_read(nid):
    db = get_db()
    cur = db.execute("UPDATE notifications SET is_read=1 WHERE id=? AND (user_id IS NULL OR user_id=?)",
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
    except sqlite3.IntegrityError:
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
        args.append(role)
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
                         (name, email, generate_password_hash(pw), role, now_iso()))
        get_db().commit()
    except sqlite3.IntegrityError:
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
    name = str(d.get("full_name", target["full_name"])).strip()
    role = d.get("role", target["role"])
    active = 1 if d.get("is_active", bool(target["is_active"])) else 0
    if not (2 <= len(name) <= 100) or role not in ("admin", "driver", "commuter"):
        return jsonify(error="Invalid name or role."), 422
    if uid == current_user()["id"] and (role != "admin" or not active):
        return jsonify(error="You cannot demote or deactivate your own account."), 400
    db.execute("UPDATE users SET full_name=?, role=?, is_active=? WHERE id=?",
               (name, role, active, uid))
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
        "SELECT COUNT(*) c FROM incidents WHERE created_at>=? AND created_at<?", (a, b)).fetchone()["c"]
    by_type = db.execute("SELECT type, COUNT(*) c FROM incidents WHERE created_at>=? AND created_at<? GROUP BY type",
                         (a, b)).fetchall()
    by_day = db.execute("SELECT substr(created_at,1,10) d, COUNT(*) c FROM incidents "
                        "WHERE created_at>=? AND created_at<? GROUP BY d ORDER BY d", (a, b)).fetchall()
    reporters = db.execute("SELECT COUNT(DISTINCT user_id) c FROM incidents WHERE created_at>=? AND created_at<?",
                           (a, b)).fetchone()["c"]
    hot = db.execute("SELECT location_text, COUNT(*) c FROM incidents WHERE created_at>=? AND created_at<? "
                     "GROUP BY location_text ORDER BY c DESC LIMIT 1", (a, b)).fetchone()
    return jsonify(
        total_incidents=total, active_reporters=reporters,
        most_reported_location=hot["location_text"] if hot else None,
        by_type=[{"type": r["type"], "label": INCIDENT_TYPES.get(
            r["type"], r["type"]), "count": r["c"]} for r in by_type],
        by_day=[{"date": r["d"], "count": r["c"]} for r in by_day],
        traffic_trend={"available": False,
                       "message": "Congestion trends and travel-time averages need a live traffic data provider."})


# --------------------------------------------------------------------- CLI
def create_admin():
    """python app.py create-admin  -> prompts for details, password never echoed."""
    init_db()
    name = input("Full name: ").strip()
    email = input("Email: ").strip().lower()
    pw = getpass("Password: ")
    if not valid_email(email) or password_problem(pw):
        sys.exit(password_problem(pw) or "Invalid email.")
    with sqlite3.connect(app.config["DATABASE"]) as con:
        con.execute("INSERT INTO users(full_name,email,password_hash,role,created_at) VALUES(?,?,?,?,?)",
                    (name, email, generate_password_hash(pw), "admin", now_iso()))
    print("Administrator created.")


init_db()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "create-admin":
        create_admin()
    else:
        app.run(debug=os.environ.get("FLASK_DEBUG") == "1")

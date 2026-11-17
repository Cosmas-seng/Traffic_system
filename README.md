# Nairobi Traffic System - Frontend + reference backend

Flask/Jinja2 + Bootstrap 5 + Leaflet/OpenStreetMap + Chart.js. No React/Vue.

> **Important:** no existing backend was found, so `app.py` is a small *reference* backend (SQLite) that
> makes every page work. Live traffic, route optimisation and AI predictions are **not** faked: those
> endpoints answer "not connected" until you plug in a provider (see section 5).

## 1. Run locally in VS Code

```bash
cd traffic_system
python -m venv .venv
# Windows: .venv\Scripts\activate        macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py create-admin          # prompts for name, email, password (not echoed)
python app.py                       # http://127.0.0.1:5000
```

Optional environment variables: `SECRET_KEY` (set a long random value outside local dev),
`DATABASE_PATH`, `FLASK_DEBUG=1`. Needs internet in the browser to load Bootstrap, Leaflet, Chart.js
and map tiles from CDNs.

## 2. Structure

```
app.py                      routes, JSON API, auth, CSRF, admin checks
templates/  base.html + 10 pages + partials/ (sidebar, topbar, flash_messages, stat_card, states)
static/css/ style.css (design system), responsive.css
static/js/  main.js (api(), safe DOM helpers, confirm dialog)   map.js (reusable Leaflet module)
            dashboard.js routes.js incidents.js charts.js notifications.js
            auth.js profile.js admin.js reports.js predictions.js
static/images/ logo.png, nairobi-skyline.jpg (generated illustration - replace with a real photo)
```

## 3. API (frontend <-> Flask)

| Endpoint | Status |
|---|---|
| `GET /api/incidents`, `POST /api/incidents` (multipart) | **Working** (DB-backed, upload signature check) |
| `GET /api/notifications`, `POST /api/notifications/<id>/read` | **Working** |
| `GET/PUT /api/profile`, `POST /api/profile/password` | **Working** |
| `GET/POST /api/admin/users`, `PUT /api/admin/users/<id>` | **Working**, admin only (server-enforced) |
| `GET /api/reports?from=&to=` | **Working** for incident statistics; traffic trend reports `available:false` |
| `GET /api/traffic` | **Needs provider.** Returns `available:false` + real 24h incident count |
| `POST /api/routes` | **Needs routing engine.** Returns HTTP 501 |
| `GET /api/predictions` | **Needs AI model.** Returns `available:false` |

Contracts the frontend already understands once connected are documented at the top of
`static/js/routes.js` and `static/js/predictions.js`, and in `/api/traffic`:
`{available:true, summary:{heavy,moderate,low,avg_delay_min}, segments:[{name,level,coordinates:[[lat,lng],..]}], updated_at}`.

## 4. Test steps per page

1. **Login** - wrong password shows an error; empty/invalid email shows inline validation; eye icon toggles the password.
2. **Register** - weak password, mismatch and bad email are blocked; success redirects to login. There is no role field.
3. **Dashboard** - map centred on Nairobi CBD; zoom/pan works; legend + Quick Info visible; notice says live traffic is not connected; incidents appear as markers and in the table; searching a place (press Enter) lists matches.
4. **Route planner** - submit with empty fields (validation); submit with places -> "Route service not connected" message (expected). With an engine connected, cards + route line appear.
5. **Incidents** - click the map (marker + coordinates), pick a type, optionally add a PNG/JPG/WebP <= 5 MB, submit; success message and marker preview. Try a `.txt` renamed `.png` -> rejected.
6. **Users (admin)** - log in as the admin you created: search, filter by role, add, edit, deactivate (confirm dialog). A normal user opening `/admin` gets 403.
7. **Reports** - Generate for a date range; charts show real incident data; Traffic Summary tab explains what is missing.
8. **Profile** - edit name/email; change password (wrong current password is rejected).
9. **Predictions** - shows "model not connected" placeholder; no numbers are invented.
10. **Notifications** - report an incident first, then mark one / all as read; the top-bar badge updates.
11. **Mobile** - narrow the window below 992 px: hamburger opens the menu, cards stack, tables scroll inside their box, no horizontal page scroll.

## 5. Needs a separate provider / model

- **Live traffic** (Google Roads/Routes, TomTom, HERE, Mapbox, or your own sensor/probe data) -> `/api/traffic`.
- **Routing engine** (OSRM, GraphHopper, Valhalla, or a commercial API) combined with traffic -> `/api/routes`.
- **Prediction model** (your trained model) -> `/api/predictions`.
- OpenStreetMap tiles are only a base map. The public tile server and Nominatim have usage policies
  (no heavy use, user-triggered search only). For production use a permitted provider: change `window.TS_MAP_CONFIG`
  (`tileUrl`, `attribution`, `geocoderUrl`) before `map.js` loads.

## 6. Moving to your MySQL/MariaDB

All SQL lives in `app.py`. Install `PyMySQL`, replace `get_db()`/`init_db()`, change `?` placeholders to
`%s`, `AUTOINCREMENT` to `AUTO_INCREMENT`, and `substr(created_at,1,10)` to `DATE(created_at)`.
Keep the column names so the JSON field names stay consistent with the frontend. Do not change your schema
unless needed; adapt the queries instead.

## 7. Security notes

Passwords hashed (Werkzeug scrypt); CSRF token on every state-changing request; roles read from the DB on each
request; self-registration always creates the lowest role; open-redirect protection on `next`; uploads limited to 5 MB,
extension + file-signature checked, stored under random names outside `static/`; dynamic content is rendered with
`textContent`, never `innerHTML`. Not included: login rate limiting, HTTPS/`SESSION_COOKIE_SECURE`, email verification,
password reset, and a route to view uploaded photos. Add these before production.

(function () {
  "use strict";

  // Central, swappable configuration. Another permitted tile provider can be
  // dropped in by overriding window.TS_MAP_CONFIG before this script runs.
  const DEFAULTS = {
    center: [-1.2864, 36.8172], // Nairobi CBD
    zoom: 13,
    tileUrl: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19,
    geocoderUrl: "https://nominatim.openstreetmap.org/search",
  };
  const CFG = Object.assign({}, DEFAULTS, window.TS_MAP_CONFIG || {});
  const LEVEL_COLOR = { heavy: "#E53935", moderate: "#F59E0B", low: "#16A34A" };
  const TYPE_COLOR = {
    accident: "#E53935",
    road_closure: "#102C4E",
    hazard: "#F59E0B",
    heavy_congestion: "#E53935",
  };
  const TYPE_ICON = {
    accident: "bi-car-front-fill",
    road_closure: "bi-sign-stop-fill",
    hazard: "bi-exclamation-triangle-fill",
    heavy_congestion: "bi-cone-striped",
  };
  const registry = {};

  function incidentIcon(type) {
    const color = TYPE_COLOR[type] || "#0866E8",
      icon = TYPE_ICON[type] || "bi-geo-alt-fill";
    // Both values come from fixed lookup tables above, never from API text.
    return L.divIcon({
      className: "",
      iconSize: [28, 28],
      iconAnchor: [14, 14],
      popupAnchor: [0, -14],
      html:
        '<div style="width:28px;height:28px;border-radius:50%;background:' +
        color +
        ';border:2px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,.4);display:flex;align-items:center;justify-content:center;color:#fff;font-size:13px">' +
        '<i class="bi ' +
        icon +
        '"></i></div>',
    });
  }

  function popupFor(inc) {
    const el = TS.el;
    return el("div", {}, [
      el("strong", { text: inc.type_label || inc.type }),
      el("div", { text: inc.location_text }),
      inc.description
        ? el("div", { class: "mt-1", text: inc.description })
        : null,
      el("div", { class: "text-muted mt-1", text: TS.fmtDate(inc.created_at) }),
    ]);
  }

  function create(elementId, opts) {
    opts = opts || {};
    if (registry[elementId]) return registry[elementId];
    const node = document.getElementById(elementId);
    if (!node) return null;
    if (typeof L === "undefined") {
      showOverlay(
        opts.overlayId,
        "The map library could not be loaded. Check your internet connection.",
      );
      return null;
    }

    const map = L.map(node, {
      zoomControl: true,
      center: opts.center || CFG.center,
      zoom: opts.zoom || CFG.zoom,
    });
    const tiles = L.tileLayer(CFG.tileUrl, {
      maxZoom: CFG.maxZoom,
      attribution: CFG.attribution,
      referrerPolicy: "strict-origin-when-cross-origin",
    }).addTo(map);
    const incidentLayer = L.layerGroup().addTo(map);
    const trafficLayer = L.layerGroup().addTo(map);
    const routeLayer = L.layerGroup().addTo(map);
    let pickMarker = null,
      tileErrors = 0;

    showOverlay(opts.overlayId, "Loading map...", true);
    tiles.on("load", () => {
      tileErrors = 0;
      hideOverlay(opts.overlayId);
    });
    tiles.on("tileerror", () => {
      if (++tileErrors > 3)
        showOverlay(
          opts.overlayId,
          "Map tiles could not be loaded. Check your connection; zoom and markers may be unavailable.",
        );
    });

    const ctl = {
      map,
      /** Replace incident markers. */
      setIncidents(list) {
        incidentLayer.clearLayers();
        (list || []).forEach((inc) => {
          if (
            typeof inc.latitude !== "number" ||
            typeof inc.longitude !== "number"
          )
            return;
          L.marker([inc.latitude, inc.longitude], {
            icon: incidentIcon(inc.type),
            title: inc.type_label || inc.type,
            keyboard: true,
          })
            .bindPopup(popupFor(inc))
            .addTo(incidentLayer);
        });
      },
      /** Draw coloured road segments. segments: [{name, level, coordinates:[[lat,lng],...]}]. Only real geometry is drawn. */
      setTraffic(segments) {
        trafficLayer.clearLayers();
        (segments || []).forEach((s) => {
          if (!Array.isArray(s.coordinates) || s.coordinates.length < 2) return;
          L.polyline(s.coordinates, {
            color: LEVEL_COLOR[s.level] || "#64748B",
            weight: 5,
            opacity: 0.85,
          })
            .bindPopup(
              TS.el("div", {}, [
                TS.el("strong", { text: s.name || "Road segment" }),
                TS.el("div", { text: "Congestion: " + (s.level || "unknown") }),
              ]),
            )
            .addTo(trafficLayer);
        });
      },
      /** Draw a route polyline from real geometry and fit the view. */
      showRoute(coordinates, level) {
        routeLayer.clearLayers();
        if (!Array.isArray(coordinates) || coordinates.length < 2) return false;
        const line = L.polyline(coordinates, {
          color: LEVEL_COLOR[level] || "#0866E8",
          weight: 6,
          opacity: 0.9,
        }).addTo(routeLayer);
        map.fitBounds(line.getBounds(), { padding: [30, 30] });
        return true;
      },
      clearRoute() {
        routeLayer.clearLayers();
      },
      /** Let the user click the map to choose a point; callback receives (lat, lng). */
      enablePicker(onPick) {
        map.getContainer().style.cursor = "crosshair";
        map.on("click", (e) => {
          ctl.setPick(e.latlng.lat, e.latlng.lng, false);
          onPick && onPick(e.latlng.lat, e.latlng.lng);
        });
      },
      setPick(lat, lng, pan) {
        if (pickMarker) pickMarker.setLatLng([lat, lng]);
        else
          pickMarker = L.marker([lat, lng], {
            title: "Selected location",
          }).addTo(map);
        if (pan) map.setView([lat, lng], Math.max(map.getZoom(), 15));
      },
      clearPick() {
        if (pickMarker) {
          map.removeLayer(pickMarker);
          pickMarker = null;
        }
      },
      /** Search a place name (Nairobi area). User-triggered only, per Nominatim usage policy. */
      async geocode(query) {
        const url =
          CFG.geocoderUrl +
          "?format=jsonv2&limit=5&countrycodes=ke&viewbox=36.65,-1.15,37.05,-1.45&bounded=0&q=" +
          encodeURIComponent(query + ", Nairobi");
        const resp = await fetch(url, {
          headers: { Accept: "application/json" },
        });
        if (!resp.ok)
          throw new Error(
            "Place search is unavailable right now (HTTP " + resp.status + ").",
          );
        const rows = await resp.json();
        return (rows || [])
          .map((r) => ({
            name: r.display_name,
            lat: parseFloat(r.lat),
            lng: parseFloat(r.lon),
          }))
          .filter((r) => isFinite(r.lat) && isFinite(r.lng));
      },
      invalidate() {
        map.invalidateSize();
      },
    };
    registry[elementId] = ctl;
    setTimeout(() => map.invalidateSize(), 150);
    return ctl;
  }

  function showOverlay(id, text, quiet) {
    const o = id && document.getElementById(id);
    if (!o) return;
    o.textContent = text;
    o.classList.add("show");
    if (quiet)
      setTimeout(() => {
        if (o.textContent === text) o.classList.remove("show");
      }, 4000); // never block the map forever
  }
  function hideOverlay(id) {
    const o = id && document.getElementById(id);
    if (o) o.classList.remove("show");
  }

  window.TSMap = { create, CFG, LEVEL_COLOR };
})();

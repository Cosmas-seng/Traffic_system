(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const STATES = [
    "routesIdle",
    "routesLoading",
    "routesEmpty",
    "routesUnavailable",
    "routesError",
    "routesList",
  ];
  const ctl = TSMap.create("routeMap", { overlayId: "routeMapOverlay" });
  let routes = [],
    selectedId = null;

  TS.showState("routesIdle", STATES);

  function num(v, digits) {
    return typeof v === "number" && isFinite(v) ? v.toFixed(digits) : null;
  }

  function selectRoute(id) {
    selectedId = id;
    const r = routes.find((x) => x.id === id);
    document
      .querySelectorAll(".route-card")
      .forEach((c) =>
        c.classList.toggle("selected", c.dataset.id === String(id)),
      );
    if (!ctl || !r) return;
    if (!ctl.showRoute(r.geometry, r.level)) {
      ctl.clearRoute();
      TS.alertIn(
        "routeNote",
        "info",
        "This route has no map geometry from the routing service, so it cannot be drawn.",
      );
    } else {
      const n = $("routeNote");
      if (n) n.replaceChildren();
    }
  }

  function card(r, index) {
    const level = ["low", "moderate", "heavy"].includes(r.level) ? r.level : "";
    const mins = num(r.duration_min, 0),
      km = num(r.distance_km, 1);
    const meta = TS.el("div", { class: "route-meta" }, [
      TS.el("span", { text: mins ? mins + " min" : "Time unavailable" }),
      document.createTextNode(" · "),
      TS.el("span", { text: km ? km + " km" : "Distance unavailable" }),
    ]);
    const viewBtn = TS.el("button", {
      class: "btn btn-sm btn-outline-primary",
      text: "View on Map",
      attrs: { type: "button" },
    });
    viewBtn.addEventListener("click", () => selectRoute(r.id));
    const card = TS.el(
      "div",
      { class: "route-card " + level, attrs: { "data-id": String(r.id) } },
      [
        TS.el("div", { class: "flex-grow-1" }, [
          TS.el("p", {
            class: "route-name",
            text:
              (r.name || "Route " + (index + 1)) +
              (r.recommended ? " (recommended)" : ""),
          }),
          meta,
        ]),
        TS.severityBadge(level || null),
        viewBtn,
      ],
    );
    return card;
  }

  $("routeForm").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const o = $("origin"),
      d = $("destination"),
      btn = $("routeBtn");
    const ok = [o, d].map((i) => {
      const v = i.value.trim().length > 0;
      i.classList.toggle("is-invalid", !v);
      return v;
    });
    if (ok.includes(false)) return (ok[0] ? d : o).focus();

    TS.showState("routesLoading", STATES);
    TS.setBusy(btn, true, "Searching...");
    try {
      const data = await TS.api("/api/routes", {
        method: "POST",
        json: { origin: o.value.trim(), destination: d.value.trim() },
      });
      routes = (data.routes || []).filter((r) => r && r.id != null);
      if (ctl) ctl.clearRoute();
      if (!routes.length) return TS.showState("routesEmpty", STATES);
      const list = $("routesList");
      list.replaceChildren(
        TS.el("div", { attrs: { id: "routeNote" } }),
        ...routes.map(card),
      );
      TS.showState("routesList", STATES);
      selectRoute((routes.find((r) => r.recommended) || routes[0]).id);
    } catch (e) {
      if (e.status === 501) {
        // routing engine not connected: say so plainly
        TS.setError("routesUnavailable", e.message);
        const n = $("routesUnavailable").querySelector(".state-text");
        n.textContent = e.message;
        TS.showState("routesUnavailable", STATES);
      } else {
        TS.setError("routesError", e.message);
        TS.showState("routesError", STATES);
      }
    } finally {
      TS.setBusy(btn, false);
    }
  });
})();

/* Dashboard: map, traffic summary, incident list, place search. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const ctl = TSMap.create("trafficMap", { overlayId: "mapOverlay" });
  const INC_STATES = ["incLoading", "incEmpty", "incError", "incTableWrap"];

  /* ---- traffic summary (/api/traffic) ---- */
  async function loadTraffic() {
    try {
      const d = await TS.api("/api/traffic");
      $("lastUpdated").textContent = d.updated_at || "--";
      const note = $("trafficNotice");
      if (!d.available) {
        // Be explicit: nothing is simulated when no provider is connected.
        note.querySelector("span").textContent =
          d.message || "Live traffic data is not connected.";
        note.classList.remove("is-hidden");
        return;
      }
      note.classList.add("is-hidden");
      const s = d.summary || {};
      $("statHeavy").textContent = s.heavy ?? "--";
      $("statModerate").textContent = s.moderate ?? "--";
      $("statLow").textContent = s.low ?? "--";
      $("statDelay").textContent =
        s.avg_delay_min != null ? "+" + s.avg_delay_min + " min" : "--";
      if (ctl) ctl.setTraffic(d.segments);
    } catch (e) {
      const note = $("trafficNotice");
      note.className = "alert alert-warning py-2";
      note.querySelector("span").textContent = e.message;
    }
  }

  /* ---- incidents (/api/incidents) ---- */
  async function loadIncidents() {
    TS.showState("incLoading", INC_STATES);
    try {
      const list = (await TS.api("/api/incidents")).incidents || [];
      $("statIncidents").textContent = list.length;
      if (ctl) ctl.setIncidents(list);
      if (!list.length) return TS.showState("incEmpty", INC_STATES);
      const body = $("incBody");
      body.replaceChildren();
      list
        .slice(0, 8)
        .forEach((i) =>
          body.appendChild(
            TS.el("tr", {}, [
              TS.el("td", { text: i.type_label }),
              TS.el("td", { text: i.location_text }),
              TS.el("td", { text: TS.fmtDate(i.created_at) }),
            ]),
          ),
        );
      TS.showState("incTableWrap", INC_STATES);
    } catch (e) {
      TS.setError("incError", e.message);
      TS.showState("incError", INC_STATES);
    }
  }

  /* ---- place search (user-triggered; no autocomplete, per OSM Nominatim policy) ---- */
  $("searchForm").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const q = $("searchInput").value.trim(),
      box = $("searchResults");
    if (q.length < 3 || !ctl) return;
    box.replaceChildren(
      TS.el("div", {
        class: "list-group-item text-muted",
        text: "Searching...",
      }),
    );
    box.classList.remove("is-hidden");
    try {
      const rows = await ctl.geocode(q);
      if (!rows.length) {
        box.replaceChildren(
          TS.el("div", {
            class: "list-group-item text-muted",
            text: "No matching places found in the Nairobi area.",
          }),
        );
        return;
      }
      box.replaceChildren(
        ...rows.map((r) => {
          const b = TS.el("button", {
            class: "list-group-item list-group-item-action",
            text: r.name,
            attrs: { type: "button" },
          });
          b.addEventListener("click", () => {
            ctl.map.setView([r.lat, r.lng], 16);
            ctl.setPick(r.lat, r.lng, false);
            box.classList.add("is-hidden");
          });
          return b;
        }),
      );
    } catch (e) {
      box.replaceChildren(
        TS.el("div", { class: "list-group-item text-danger", text: e.message }),
      );
    }
  });

  loadTraffic();
  loadIncidents();
  setInterval(loadTraffic, 60000); // refresh traffic summary every minute
})();

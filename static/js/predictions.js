(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const STATES = ["predLoading", "predError", "predUnavailable", "predBody"];
  const LEVELS = { low: "Low", moderate: "Moderate", heavy: "Heavy", severe: "Severe" };

  function setModel(status) {
    const b = $("modelBadge"),
      ready = status === "ready";
    const demo = status === "demo";
    b.textContent = demo ? "Presentation demo — not trained ML" :
      "Model: " + (ready ? "ready" : status === "not_connected" ? "not connected" : status || "unknown");
    b.classList.toggle("off", !ready && !demo);
    b.classList.toggle("demo-mode", demo);
  }
  function unavailable(msg) {
    $("predUnavailable").querySelector(".state-text").textContent = msg;
    TS.showState("predUnavailable", STATES);
  }

  async function fetchPrediction() {
    const loc = $("location").value;
    TS.showState("predLoading", STATES);
    TS.setBusy($("predBtn"), true, "Predicting...");
    try {
      const d = await TS.api(
        "/api/predictions?" +
          new URLSearchParams({ location: loc, horizon: $("horizon").value }),
      );
      setModel(d.model_status);
      if (!d.available)
        return unavailable(
          d.message || "No prediction exists for this selection.",
        );
      $("pCurrent").textContent = LEVELS[d.current_level] || "Unknown";
      $("pLevel").textContent = LEVELS[d.predicted_level] || "Unknown";
      $("pDelay").textContent =
        d.estimated_delay_min != null
          ? d.estimated_delay_min + " min"
          : "Unknown";
      $("pTime").textContent = d.generated_at
        ? TS.fmtDate(d.generated_at)
        : "Unknown";
      $("modelInfo").textContent = d.is_demo
        ? "PRESENTATION DEMO ONLY — " + (d.message || "Heuristic estimate, not a trained or validated ML forecast.") +
          (d.data_warning ? " " + d.data_warning : "")
        : d.model
          ? "Predicted speed: " + d.predicted_speed_kmh + " km/h; model " +
            d.model.name + " (v" + d.model.version + ")."
          : "Model-generated traffic estimate.";
      TS.showState("predBody", STATES); // visible before drawing so the chart can size itself
      const hist = d.history || [],
        fc = d.forecast || [];
      const labels = hist.concat(fc).map((p) => p.t);
      const hData = hist.map((p) => p.value).concat(fc.map(() => null));
      const fData = hist
        .map((p, i) => (i === hist.length - 1 ? p.value : null))
        .concat(fc.map((p) => p.value)); // joins the two lines
      TSCharts.line(
        "predChart",
        labels,
        [
          {
            label: "Historical (recorded)",
            data: hData,
            borderColor: TSCharts.COLORS.navy,
          },
          {
            label: d.is_demo ? "Demonstration estimate (not trained ML)" : "Predicted (model)",
            data: fData,
            borderColor: TSCharts.COLORS.primary,
            borderDash: [6, 4],
          },
        ],
        {
          scales: {
            y: {
              min: 0,
              max: 140,
              title: { display: true, text: "Average speed (km/h)" },
            },
          },
        },
      );
    } catch (e) {
      TS.setError("predError", e.message);
      TS.showState("predError", STATES);
    } finally {
      TS.setBusy($("predBtn"), false);
      $("predBtn").disabled = $("location").disabled;
    }
  }

  $("predForm").addEventListener("submit", (ev) => {
    ev.preventDefault();
    fetchPrediction();
  });

  (async function init() {
    TS.showState("predLoading", STATES);
    try {
      const d = await TS.api("/api/predictions");
      setModel(d.model_status);
      const sel = $("location"),
        locs = d.locations || [];
      if (!locs.length) {
        sel.replaceChildren(
          TS.el("option", { text: "No locations available" }),
        );
        return unavailable(d.message || "No prediction model is connected.");
      }
      sel.replaceChildren(
        ...locs.map((l) =>
          TS.el("option", { text: l.name, attrs: { value: String(l.id) } }),
        ),
      );
      sel.disabled = false;
      $("predBtn").disabled = false;
      fetchPrediction();
    } catch (e) {
      setModel("error");
      TS.setError("predError", e.message);
      TS.showState("predError", STATES);
    }
  })();
})();

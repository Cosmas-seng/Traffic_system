/* Reports: GET /api/reports?from=YYYY-MM-DD&to=YYYY-MM-DD. All figures shown are real database results. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const STATES = ["repLoading", "repError", "repIdle", "repBody"];
  const iso = (d) => d.toISOString().slice(0, 10);

  const today = new Date(),
    start = new Date(Date.now() - 29 * 864e5);
  $("dateFrom").value = iso(start);
  $("dateTo").value = iso(today);
  $("dateTo").max = iso(today);
  TS.showState("repIdle", STATES);

  $("reportForm").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const from = $("dateFrom"),
      to = $("dateTo");
    $("reportAlert").replaceChildren();
    if (!from.value || !to.value)
      return TS.alertIn(
        "reportAlert",
        "danger",
        "Choose both a start and an end date.",
      );
    if (to.value < from.value)
      return TS.alertIn(
        "reportAlert",
        "danger",
        "The end date must not be before the start date.",
      );

    TS.showState("repLoading", STATES);
    TS.setBusy($("genBtn"), true, "Generating...");
    try {
      const d = await TS.api(
        "/api/reports?" +
          new URLSearchParams({ from: from.value, to: to.value }),
      );
      $("kTotal").textContent = d.total_incidents;
      $("kHot").textContent = d.most_reported_location || "No data";
      $("kReporters").textContent = d.active_reporters;
      if (d.traffic_trend && d.traffic_trend.message)
        $("trafficTrendMsg").textContent = d.traffic_trend.message;
      $("repEmpty").classList.toggle("is-hidden", d.total_incidents > 0);

      TS.showState("repBody", STATES); // show first so Chart.js can measure the canvases
      TSCharts.line(
        "dayChart",
        d.by_day.map((r) => r.date),
        [
          {
            label: "Incident reports",
            data: d.by_day.map((r) => r.count),
            borderColor: TSCharts.COLORS.primary,
            backgroundColor: "rgba(8,102,232,.12)",
            fill: true,
          },
        ],
      );
      TSCharts.bar(
        "typeChart",
        d.by_type.map((r) => r.label),
        [
          {
            label: "Incident reports",
            data: d.by_type.map((r) => r.count),
            backgroundColor: TSCharts.COLORS.navy,
          },
        ],
      );
    } catch (e) {
      TS.setError("repError", e.message);
      TS.showState("repError", STATES);
    } finally {
      TS.setBusy($("genBtn"), false);
    }
  });
})();

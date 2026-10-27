/* Reusable Chart.js helpers -> window.TSCharts. Colours come from the project palette. */
(function () {
  "use strict";
  const COLORS = {
    primary: "#0866E8",
    heavy: "#E53935",
    moderate: "#F59E0B",
    low: "#16A34A",
    navy: "#102C4E",
    grid: "#E6EDF5",
    muted: "#64748B",
  };
  const live = {}; // canvasId -> Chart, so a canvas is never drawn on twice

  function render(canvasId, config) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === "undefined") return null;
    if (live[canvasId]) live[canvasId].destroy();
    config.options = Object.assign(
      {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: { boxWidth: 12, color: COLORS.navy },
          },
        },
        scales: {
          x: { grid: { display: false }, ticks: { color: COLORS.muted } },
          y: {
            beginAtZero: true,
            grid: { color: COLORS.grid },
            ticks: { color: COLORS.muted, precision: 0 },
          },
        },
      },
      config.options || {},
    );
    live[canvasId] = new Chart(canvas, config);
    return live[canvasId];
  }

  function line(canvasId, labels, datasets, options) {
    return render(canvasId, {
      type: "line",
      data: {
        labels,
        datasets: datasets.map((d) =>
          Object.assign({ tension: 0.3, pointRadius: 3, borderWidth: 2 }, d),
        ),
      },
      options,
    });
  }
  function bar(canvasId, labels, datasets, options) {
    return render(canvasId, {
      type: "bar",
      data: {
        labels,
        datasets: datasets.map((d) => Object.assign({ borderRadius: 4 }, d)),
      },
      options,
    });
  }
  function destroy(canvasId) {
    if (live[canvasId]) {
      live[canvasId].destroy();
      delete live[canvasId];
    }
  }

  window.TSCharts = { COLORS, line, bar, destroy };
})();

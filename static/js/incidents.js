(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const MAX_BYTES = 5 * 1024 * 1024,
    OK_TYPES = ["image/png", "image/jpeg", "image/webp"];
  const ctl = TSMap.create("incidentMap", { overlayId: "incidentMapOverlay" });
  let autoFilled = false; // true while the location text was filled from a map click

  function fieldError(name, message) {
    const box = document.querySelector('[data-for="' + name + '"]');
    if (!box) return;
    if (message) box.textContent = message;
    box.classList.remove("is-hidden");
    box.classList.add("d-block"); // feedback may not be a sibling of the input
    const input = $(name);
    if (input) input.classList.add("is-invalid");
    if (name === "location") {
      const t = $("location_text");
      if (t) t.classList.add("is-invalid");
    }
  }
  function clearErrors() {
    document
      .querySelectorAll(".is-invalid")
      .forEach((n) => n.classList.remove("is-invalid"));
    document.querySelectorAll("[data-for]").forEach((n) => {
      n.classList.remove("d-block");
      if (n.dataset.for === "location") n.classList.add("is-hidden");
    });
  }

  function setPoint(lat, lng, label, pan) {
    $("latitude").value = lat.toFixed(6);
    $("longitude").value = lng.toFixed(6);
    if (ctl) ctl.setPick(lat, lng, pan);
    $("pinInfo").textContent =
      "Selected: " + lat.toFixed(5) + ", " + lng.toFixed(5);
    if (label) {
      $("location_text").value = label;
      autoFilled = true;
    } else if (!$("location_text").value.trim()) {
      $("location_text").value =
        "Map pin (" + lat.toFixed(4) + ", " + lng.toFixed(4) + ")";
      autoFilled = true;
    }
    document.querySelector('[data-for="location"]').classList.add("is-hidden");
  }

  if (ctl)
    ctl.enablePicker((lat, lng) => {
      // Replace automatic labels, but never overwrite text the user typed.
      if (autoFilled) $("location_text").value = "";
      setPoint(lat, lng, null, false);
    });
  $("location_text").addEventListener("input", () => {
    autoFilled = false;
  });

  $("findOnMap").addEventListener("click", async () => {
    const q = $("location_text").value.trim();
    if (q.length < 3 || !ctl) return fieldError("location_text");
    try {
      const rows = await ctl.geocode(q);
      if (!rows.length)
        return TS.alertIn(
          "formAlert",
          "warning",
          "No matching place found. Try another name or click the map.",
        );
      setPoint(rows[0].lat, rows[0].lng, null, true);
      autoFilled = false;
      $("formAlert").replaceChildren();
    } catch (e) {
      TS.alertIn("formAlert", "danger", e.message);
    }
  });

  $("description").addEventListener("input", (e) => {
    $("descCount").textContent = e.target.value.length;
  });

  $("image").addEventListener("change", (e) => {
    const f = e.target.files[0],
      prev = $("imagePreview");
    e.target.classList.remove("is-invalid");
    if (prev.src && prev.src.startsWith("blob:")) URL.revokeObjectURL(prev.src);
    prev.classList.add("is-hidden");
    if (!f) return;
    if (!OK_TYPES.includes(f.type)) {
      e.target.value = "";
      return fieldError("image", "Only PNG, JPG or WebP images are allowed.");
    }
    if (f.size > MAX_BYTES) {
      e.target.value = "";
      return fieldError("image", "The image must be 5 MB or smaller.");
    }
    prev.src = URL.createObjectURL(f);
    prev.classList.remove("is-hidden");
  });

  $("incidentForm").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    clearErrors();
    $("formAlert").replaceChildren();
    const f = ev.target;
    let bad = false;
    if (!f.type.value) {
      fieldError("type");
      bad = true;
    }
    if (f.location_text.value.trim().length < 3) {
      fieldError("location_text");
      bad = true;
    }
    if (!f.latitude.value || !f.longitude.value) {
      fieldError("location");
      bad = true;
    }
    if (bad)
      return TS.alertIn(
        "formAlert",
        "danger",
        "Please correct the highlighted fields.",
      );

    const btn = $("submitBtn");
    TS.setBusy(btn, true, "Submitting...");
    try {
      const data = await TS.api("/api/incidents", {
        method: "POST",
        body: new FormData(f),
      }); // browser sets multipart boundary
      TS.alertIn(
        "formAlert",
        "success",
        "Report submitted. Thank you for helping other road users.",
      );
      if (ctl && data.incident) ctl.setIncidents([data.incident]); // marker preview of what was saved
      f.reset();
      autoFilled = false;
      $("descCount").textContent = "0";
      $("imagePreview").classList.add("is-hidden");
      $("latitude").value = $("longitude").value = "";
      $("pinInfo").textContent =
        "Report saved. Click the map to report another incident.";
    } catch (e) {
      const fields = (e.data && e.data.fields) || {};
      Object.keys(fields).forEach((k) =>
        fieldError(k === "location" ? "location" : k, fields[k]),
      );
      TS.alertIn("formAlert", "danger", e.message);
    } finally {
      TS.setBusy(btn, false);
    }
  });
})();

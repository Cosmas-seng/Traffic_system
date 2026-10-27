(function () {
  "use strict";

  /** Create an element safely. Text is set with textContent, never innerHTML. */
  function el(tag, opts, children) {
    const node = document.createElement(tag);
    opts = opts || {};
    if (opts.class) node.className = opts.class;
    if (opts.text != null) node.textContent = String(opts.text);
    if (opts.attrs)
      Object.keys(opts.attrs).forEach((k) =>
        node.setAttribute(k, opts.attrs[k]),
      );
    (children || []).forEach((c) => c && node.appendChild(c));
    return node;
  }

  /**
   * fetch wrapper: adds the CSRF header, checks the status, parses JSON safely.
   * Resolves with the parsed body. Rejects with an Error that has .status and .data
   * and a user-friendly .message.
   */
  async function api(url, options) {
    options = Object.assign(
      { method: "GET", headers: {}, credentials: "same-origin" },
      options || {},
    );
    options.headers = Object.assign(
      { Accept: "application/json" },
      options.headers,
    );
    if (options.method !== "GET")
      options.headers["X-CSRF-Token"] = document.body.dataset.csrf || "";
    if (options.json !== undefined) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(options.json);
      delete options.json;
    }
    let resp;
    try {
      resp = await fetch(url, options);
    } catch (e) {
      const err = new Error(
        "Could not reach the server. Check your connection and try again.",
      );
      err.status = 0;
      throw err;
    }
    let data = null;
    const text = await resp.text();
    if (text) {
      try {
        data = JSON.parse(text);
      } catch (e) {
        data = null;
      }
    }
    if (resp.status === 401) {
      window.location.href = "/login";
    }
    if (!resp.ok) {
      const err = new Error(
        (data && data.error) || "Request failed (HTTP " + resp.status + ").",
      );
      err.status = resp.status;
      err.data = data;
      throw err;
    }
    return data || {};
  }

  /** Show exactly one of several state blocks (ids) and hide the rest. */
  function showState(active, all) {
    all.forEach((id) => {
      const n = document.getElementById(id);
      if (n) n.classList.toggle("is-hidden", id !== active);
    });
  }
  function setError(id, message) {
    const n = document.getElementById(id);
    if (!n) return;
    const t = n.querySelector(".state-text");
    if (t) t.textContent = message;
  }

  /** Inline alert in a container (id). type: success | danger | info | warning */
  function alertIn(containerId, type, message) {
    const box = document.getElementById(containerId);
    if (!box) return;
    box.replaceChildren(
      el("div", {
        class: "alert alert-" + type + " py-2 mb-3",
        attrs: { role: type === "danger" ? "alert" : "status" },
        text: message,
      }),
    );
  }

  function setBusy(button, busy, label) {
    if (!button) return;
    if (busy) {
      button.dataset.label = button.textContent;
      button.disabled = true;
      button.replaceChildren(
        el("span", {
          class: "spinner-border spinner-border-sm me-2",
          attrs: { "aria-hidden": "true" },
        }),
        document.createTextNode(label || "Please wait..."),
      );
    } else {
      button.disabled = false;
      button.textContent = button.dataset.label || button.textContent;
    }
  }

  /** Bootstrap confirmation modal -> Promise<boolean> */
  function confirmDialog(message, okLabel) {
    return new Promise((resolve) => {
      const modalEl = document.getElementById("confirmModal");
      const modal = bootstrap.Modal.getOrCreateInstance(modalEl);
      document.getElementById("confirmBody").textContent = message;
      const ok = document.getElementById("confirmOk");
      ok.textContent = okLabel || "Confirm";
      let result = false;
      const onOk = () => {
        result = true;
        modal.hide();
      };
      const onHide = () => {
        ok.removeEventListener("click", onOk);
        modalEl.removeEventListener("hidden.bs.modal", onHide);
        resolve(result);
      };
      ok.addEventListener("click", onOk);
      modalEl.addEventListener("hidden.bs.modal", onHide);
      modal.show();
    });
  }

  function fmtDate(s) {
    if (!s) return "";
    const d = new Date(String(s).replace(" ", "T") + "Z"); // server stores UTC
    return isNaN(d)
      ? String(s)
      : d.toLocaleString("en-KE", {
          dateStyle: "medium",
          timeStyle: "short",
          timeZone: "Africa/Nairobi",
        });
  }

  const SEVERITY = { heavy: "Heavy", moderate: "Moderate", low: "Low" };
  function severityBadge(level) {
    const key = SEVERITY[level] ? level : "neutral";
    return el("span", {
      class: "badge-sev " + key,
      text: SEVERITY[level] || "Unknown",
    });
  }

  /* ---------- page chrome ---------- */
  document.addEventListener("DOMContentLoaded", function () {
    // Mobile sidebar
    const sb = document.getElementById("appSidebar"),
      btn = document.getElementById("menuToggle"),
      back = document.getElementById("sidebarBackdrop");
    function toggle(open) {
      if (!sb || !btn) return;
      sb.classList.toggle("open", open);
      back && back.classList.toggle("show", open);
      btn.setAttribute("aria-expanded", String(open));
    }
    if (btn)
      btn.addEventListener("click", () =>
        toggle(!sb.classList.contains("open")),
      );
    if (back) back.addEventListener("click", () => toggle(false));
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") toggle(false);
    });

    // Password visibility toggles: <button class="toggle-pw" data-target="#id">
    document.querySelectorAll(".toggle-pw").forEach((b) =>
      b.addEventListener("click", () => {
        const input = document.querySelector(b.dataset.target);
        if (!input) return;
        const show = input.type === "password";
        input.type = show ? "text" : "password";
        b.setAttribute("aria-label", show ? "Hide password" : "Show password");
        b.setAttribute("aria-pressed", String(show));
        const i = b.querySelector("i");
        if (i) i.className = "bi " + (show ? "bi-eye-slash" : "bi-eye");
      }),
    );

    // Unread notification count in the top bar (only on app pages)
    const badge = document.getElementById("unreadBadge");
    if (badge) {
      api("/api/notifications")
        .then((d) => {
          const n = (d.notifications || []).filter((x) => !x.is_read).length;
          if (n) {
            badge.textContent = n > 99 ? "99+" : String(n);
            badge.classList.remove("is-hidden");
          }
        })
        .catch(() => {
          /* badge is optional */
        });
    }
  });

  window.TS = {
    el,
    api,
    showState,
    setError,
    alertIn,
    setBusy,
    confirm: confirmDialog,
    fmtDate,
    severityBadge,
  };
})();

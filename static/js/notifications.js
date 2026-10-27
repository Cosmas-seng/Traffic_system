/* Notifications page: list from GET /api/notifications, mark-as-read via POST /api/notifications/<id>/read. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const STATES = ["notifLoading", "notifEmpty", "notifError", "notifList"];
  const CATEGORY = {
    // fixed lookup: icons never come from API text
    congestion: { icon: "bi-cone-striped", label: "Congestion alert" },
    incident: { icon: "bi-exclamation-triangle", label: "Incident alert" },
    route: { icon: "bi-signpost-split", label: "Alternative route" },
  };
  let items = [];
  function row(n) {
    const cat = CATEGORY[n.category] || {
      icon: "bi-bell",
      label: "Notification",
    };
    const body = TS.el("div", { class: "flex-grow-1" }, [
      TS.el("div", { class: "fw-semibold", text: n.title }),
      n.message ? TS.el("div", { text: n.message }) : null,
      TS.el("div", {
        class: "stat-sub",
        text: cat.label + " · " + TS.fmtDate(n.created_at),
      }),
    ]);
    const actions = TS.el("div", { class: "d-flex align-items-center gap-2" });
    if (!n.is_read) {
      actions.appendChild(
        TS.el("span", {
          class: "unread-dot",
          attrs: { title: "Unread", "aria-label": "Unread" },
        }),
      );
      const b = TS.el("button", {
        class: "btn btn-sm btn-outline-primary",
        text: "Mark as read",
        attrs: { type: "button" },
      });
      b.addEventListener("click", () => markRead([n.id], b));
      actions.appendChild(b);
    } else
      actions.appendChild(TS.el("span", { class: "stat-sub", text: "Read" }));
    return TS.el(
      "div",
      { class: "notif-item" + (n.is_read ? "" : " unread") },
      [
        TS.el(
          "div",
          { class: "notif-icon", attrs: { "aria-hidden": "true" } },
          [TS.el("i", { class: "bi " + cat.icon })],
        ),
        body,
        actions,
      ],
    );
  }

  function draw() {
    if (!items.length) return TS.showState("notifEmpty", STATES);
    $("notifList").replaceChildren(...items.map(row));
    $("markAllBtn").disabled = !items.some((n) => !n.is_read);
    TS.showState("notifList", STATES);
  }

  async function markRead(ids, btn) {
    if (btn) btn.disabled = true;
    try {
      await Promise.all(
        ids.map((id) =>
          TS.api("/api/notifications/" + id + "/read", { method: "POST" }),
        ),
      );
      items.forEach((n) => {
        if (ids.includes(n.id)) n.is_read = true;
      });
      $("notifAlert").replaceChildren();
      draw();
      const left = items.filter((n) => !n.is_read).length,
        badge = $("unreadBadge");
      if (badge) {
        badge.textContent = left;
        badge.classList.toggle("is-hidden", left === 0);
      }
    } catch (e) {
      if (btn) btn.disabled = false;
      TS.alertIn("notifAlert", "danger", e.message);
    }
  }

  $("markAllBtn").addEventListener("click", () =>
    markRead(
      items.filter((n) => !n.is_read).map((n) => n.id),
      $("markAllBtn"),
    ),
  );

  (async function load() {
    TS.showState("notifLoading", STATES);
    try {
      items = (await TS.api("/api/notifications")).notifications || [];
      draw();
    } catch (e) {
      TS.setError("notifError", e.message);
      TS.showState("notifError", STATES);
    }
  })();
})();

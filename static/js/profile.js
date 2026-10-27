/* Profile: edit name/email (PUT /api/profile) and change password (POST /api/profile/password). */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  const flag = (input, ok) => {
    input.classList.toggle("is-invalid", !ok);
    return ok;
  };

  const since = $("memberSince");
  if (since)
    since.textContent = new Date(
      since.dataset.utc.replace(" ", "T") + "Z",
    ).toLocaleDateString("en-KE", {
      month: "long",
      year: "numeric",
      day: "numeric",
      timeZone: "Africa/Nairobi",
    });

  function initials(name) {
    const p = name.trim().split(/\s+/);
    return ((p[0] || "?")[0] + (p[1] ? p[1][0] : "")).toUpperCase();
  }

  $("editForm").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const name = $("editName"),
      email = $("editEmail");
    const ok = [
      flag(
        name,
        name.value.trim().length >= 2 && name.value.trim().length <= 100,
      ),
      flag(email, EMAIL.test(email.value.trim())),
    ];
    if (ok.includes(false)) return;
    TS.setBusy($("editBtn"), true, "Saving...");
    try {
      const p = (
        await TS.api("/api/profile", {
          method: "PUT",
          json: { full_name: name.value.trim(), email: email.value.trim() },
        })
      ).profile;
      ["pName", "infoName"].forEach((i) => ($(i).textContent = p.full_name));
      ["pEmail", "infoEmail"].forEach((i) => ($(i).textContent = p.email));
      $("avatar").textContent = initials(p.full_name);
      document
        .querySelectorAll(".app-topbar .user-name")
        .forEach((n) => (n.textContent = p.full_name));
      document
        .querySelectorAll(".app-topbar .avatar")
        .forEach((n) => (n.textContent = initials(p.full_name)));
      $("editAlert").replaceChildren();
      bootstrap.Modal.getInstance($("editModal")).hide();
      TS.alertIn("profileAlert", "success", "Profile updated.");
    } catch (e) {
      TS.alertIn("editAlert", "danger", e.message);
    } finally {
      TS.setBusy($("editBtn"), false);
    }
  });

  $("pwForm").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const cur = $("curPw"),
      n1 = $("newPw"),
      n2 = $("newPw2");
    const strong =
      n1.value.length >= 8 && /[A-Za-z]/.test(n1.value) && /\d/.test(n1.value);
    const ok = [
      flag(cur, cur.value !== ""),
      flag(n1, strong),
      flag(n2, n2.value !== "" && n2.value === n1.value),
    ];
    if (ok.includes(false)) return;
    TS.setBusy($("pwBtn"), true, "Updating...");
    try {
      await TS.api("/api/profile/password", {
        method: "POST",
        json: { current_password: cur.value, new_password: n1.value },
      });
      ev.target.reset();
      $("pwAlert").replaceChildren(); // passwords are never kept in storage or the DOM afterwards
      bootstrap.Modal.getInstance($("pwModal")).hide();
      TS.alertIn("profileAlert", "success", "Password changed.");
    } catch (e) {
      TS.alertIn("pwAlert", "danger", e.message);
    } finally {
      TS.setBusy($("pwBtn"), false);
    }
  });
})();

/* Admin user management. Every call is re-checked by Flask (admin_required); hiding UI is not the protection. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const STATES = ["usersLoading", "usersEmpty", "usersError", "usersWrap"];
  const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  let page = 1,
    pages = 1,
    editing = null;
  const modal = () => bootstrap.Modal.getOrCreateInstance($("userModal"));

  async function load() {
    TS.showState("usersLoading", STATES);
    const qs = new URLSearchParams({
      q: $("q").value.trim(),
      role: $("roleFilter").value,
      page: String(page),
    });
    try {
      const d = await TS.api("/api/admin/users?" + qs);
      pages = Math.max(1, Math.ceil(d.total / d.per_page));
      if (!d.users.length) return TS.showState("usersEmpty", STATES);
      $("usersBody").replaceChildren(...d.users.map(rowFor));
      $("pageInfo").textContent =
        "Page " +
        d.page +
        " of " +
        pages +
        " · " +
        d.total +
        " user" +
        (d.total === 1 ? "" : "s");
      $("prevPage").disabled = d.page <= 1;
      $("nextPage").disabled = d.page >= pages;
      TS.showState("usersWrap", STATES);
    } catch (e) {
      TS.setError(
        "usersError",
        e.status === 403 ? "Administrator access is required." : e.message,
      );
      TS.showState("usersError", STATES);
    }
  }

  function rowFor(u) {
    const edit = TS.el(
      "button",
      {
        class: "icon-btn",
        attrs: {
          type: "button",
          "aria-label": "Edit " + u.full_name,
          title: "Edit",
        },
      },
      [TS.el("i", { class: "bi bi-pencil" })],
    );
    edit.addEventListener("click", () => openModal(u));
    const acts = TS.el("div", { class: "d-flex gap-1" }, [edit]);
    if (u.is_active) {
      const off = TS.el(
        "button",
        {
          class: "icon-btn danger",
          attrs: {
            type: "button",
            "aria-label": "Deactivate " + u.full_name,
            title: "Deactivate",
          },
        },
        [TS.el("i", { class: "bi bi-person-x" })],
      );
      off.addEventListener("click", async () => {
        if (
          !(await TS.confirm(
            "Deactivate " +
              u.full_name +
              "? They will no longer be able to log in.",
            "Deactivate",
          ))
        )
          return;
        try {
          await TS.api("/api/admin/users/" + u.id, {
            method: "PUT",
            json: { is_active: false },
          });
          TS.alertIn(
            "adminAlert",
            "success",
            u.full_name + " was deactivated.",
          );
          load();
        } catch (e) {
          TS.alertIn("adminAlert", "danger", e.message);
        }
      });
      acts.appendChild(off);
    }
    return TS.el("tr", {}, [
      TS.el("td", { text: u.id }),
      TS.el("td", { text: u.full_name }),
      TS.el("td", { text: u.email }),
      TS.el("td", { class: "text-capitalize", text: u.role }),
      TS.el("td", {}, [
        TS.el("span", {
          class: "badge-status" + (u.is_active ? "" : " off"),
          text: u.is_active ? "Active" : "Inactive",
        }),
      ]),
      TS.el("td", {}, [acts]),
    ]);
  }

  function openModal(u) {
    editing = u || null;
    $("userModalTitle").textContent = u ? "Edit User" : "Add User";
    $("uName").value = u ? u.full_name : "";
    $("uEmail").value = u ? u.email : "";
    $("uEmail").disabled = !!u; // email is the login id
    $("uRole").value = u ? u.role : "commuter";
    $("uActive").checked = u ? u.is_active : true;
    $("pwGroup").classList.toggle("is-hidden", !!u);
    $("activeGroup").classList.toggle("is-hidden", !u);
    $("uPw").value = "";
    $("userAlert").replaceChildren();
    document
      .querySelectorAll("#userForm .is-invalid")
      .forEach((n) => n.classList.remove("is-invalid"));
    modal().show();
  }

  $("userForm").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const flag = (i, ok) => {
      i.classList.toggle("is-invalid", !ok);
      return ok;
    };
    const name = $("uName"),
      email = $("uEmail"),
      pw = $("uPw");
    const checks = [flag(name, name.value.trim().length >= 2)];
    if (!editing) {
      checks.push(flag(email, EMAIL.test(email.value.trim())));
      checks.push(
        flag(
          pw,
          pw.value.length >= 8 &&
            /[A-Za-z]/.test(pw.value) &&
            /\d/.test(pw.value),
        ),
      );
    }
    if (checks.includes(false)) return;
    TS.setBusy($("userSaveBtn"), true, "Saving...");
    try {
      if (editing)
        await TS.api("/api/admin/users/" + editing.id, {
          method: "PUT",
          json: {
            full_name: name.value.trim(),
            role: $("uRole").value,
            is_active: $("uActive").checked,
          },
        });
      else
        await TS.api("/api/admin/users", {
          method: "POST",
          json: {
            full_name: name.value.trim(),
            email: email.value.trim(),
            role: $("uRole").value,
            password: pw.value,
          },
        });
      pw.value = "";
      modal().hide();
      TS.alertIn(
        "adminAlert",
        "success",
        editing ? "User updated." : "User created.",
      );
      load();
    } catch (e) {
      TS.alertIn("userAlert", "danger", e.message);
    } finally {
      TS.setBusy($("userSaveBtn"), false);
    }
  });

  $("addUserBtn").addEventListener("click", () => openModal(null));
  $("filterForm").addEventListener("submit", (ev) => {
    ev.preventDefault();
    page = 1;
    load();
  });
  $("prevPage").addEventListener("click", () => {
    if (page > 1) {
      page--;
      load();
    }
  });
  $("nextPage").addEventListener("click", () => {
    if (page < pages) {
      page++;
      load();
    }
  });
  load();
})();

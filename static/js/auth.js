/* Client-side validation for login and registration. The server re-validates everything. */
(function () {
  "use strict";
  const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

  function mark(input, ok) {
    input.classList.toggle("is-invalid", !ok);
    input.classList.toggle("is-valid", ok && input.value !== "");
    return ok;
  }

  function wire(form, rules) {
    if (!form) return;
    form.addEventListener("submit", function (e) {
      const results = Object.keys(rules).map((id) => {
        const input = form.querySelector("#" + id);
        return mark(input, rules[id](input.value, form));
      });
      if (results.includes(false)) {
        e.preventDefault();
        const bad = form.querySelector(".is-invalid");
        if (bad) bad.focus();
      }
    });
    // Clear the error as the user corrects the field
    form.querySelectorAll("input").forEach((i) =>
      i.addEventListener("input", () => {
        if (rules[i.id] && i.classList.contains("is-invalid"))
          mark(i, rules[i.id](i.value, form));
      }),
    );
  }

  const strong = (v) => v.length >= 8 && /[A-Za-z]/.test(v) && /\d/.test(v);

  wire(document.getElementById("loginForm"), {
    email: (v) => EMAIL.test(v.trim()),
    password: (v) => v.length > 0,
  });

  wire(document.getElementById("registerForm"), {
    full_name: (v) => v.trim().length >= 2 && v.trim().length <= 100,
    email: (v) => EMAIL.test(v.trim()),
    password: strong,
    confirm_password: (v, f) =>
      v !== "" && v === f.querySelector("#password").value,
  });
})();

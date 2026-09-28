(() => {
  const log = document.getElementById("log");
  const form = document.getElementById("f");
  const input = document.getElementById("q");

  function add(role, text) {
    const el = document.createElement("div");
    el.className = "msg " + role;
    const r = document.createElement("div");
    r.className = "role";
    r.textContent = role === "user" ? "you" : "lito";
    el.appendChild(r);
    const body = document.createElement("div");
    body.textContent = text;
    el.appendChild(body);
    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
  }

  async function send(text) {
    text = (text || "").trim();
    if (!text) return;
    add("user", text);
    input.value = "";
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      const data = await res.json();
      add("bot", data.reply || data.text || "(empty)");
    } catch (e) {
      add("bot", "error: " + e);
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    send(input.value);
  });
  document.querySelectorAll(".chips [data-q]").forEach((btn) => {
    btn.addEventListener("click", () => send(btn.getAttribute("data-q")));
  });

  add(
    "bot",
    "Hey — I write my own replies as I go (tiny neural stack + tools when facts matter). Ask me anything."
  );
})();

(() => {
  const chat = document.getElementById("chat");
  const form = document.getElementById("composer");
  const input = document.getElementById("input");
  const sendBtn = document.getElementById("send");
  const statusEl = document.getElementById("status");
  const chips = document.getElementById("chips");

  let busy = false;

  function el(tag, cls, html) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  }

  function addBubble(role, html, kind) {
    const wrap = el("div", `msg ${role}${kind === "error" || kind === false ? " error" : ""}`);
    const meta = el("div", "meta", role === "user" ? "You" : "Lito");
    const body = el("div", "body");
    body.innerHTML = html;
    wrap.appendChild(meta);
    wrap.appendChild(body);
    // Remove welcome if present
    const w = chat.querySelector(".welcome");
    if (w) w.remove();
    chat.appendChild(wrap);
    chat.scrollTop = chat.scrollHeight;
    return wrap;
  }

  function showWelcome() {
    chat.innerHTML = "";
    const w = el("div", "welcome");
    w.innerHTML =
      "<h2>Ready when you are</h2>" +
      "<p>Open apps, take notes, run quick tasks — without loading a huge model into RAM.</p>" +
      "<p>Try <kbd>open firefox</kbd>, <kbd>note buy milk</kbd>, or <kbd>help</kbd>.</p>";
    chat.appendChild(w);
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  async function refreshStatus() {
    try {
      const r = await fetch("/api/status");
      const s = await r.json();
      const ram = s.ram_human || "?";
      statusEl.textContent = `${ram} · ${s.apps_known || 0} apps`;
      statusEl.title = JSON.stringify(s);
    } catch {
      statusEl.textContent = "offline";
    }
  }

  async function loadHistory() {
    try {
      const r = await fetch("/api/history");
      const data = await r.json();
      const msgs = data.messages || [];
      if (!msgs.length) {
        showWelcome();
        return;
      }
      chat.innerHTML = "";
      for (const m of msgs) {
        const role = m.role === "user" ? "user" : "bot";
        // history is plain text — light format
        const html = escapeHtml(m.text || "")
          .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
          .replace(/`([^`]+)`/g, "<code>$1</code>")
          .replace(/\n/g, "<br>");
        addBubble(role, html);
      }
    } catch {
      showWelcome();
    }
  }

  async function send(text) {
    text = (text || "").trim();
    if (!text || busy) return;
    busy = true;
    sendBtn.disabled = true;
    addBubble("user", escapeHtml(text));
    input.value = "";
    try {
      const r = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await r.json();
      addBubble("bot", data.html || escapeHtml(data.text || data.error || "…"), data.ok === false ? "error" : data.kind);
    } catch (err) {
      addBubble("bot", escapeHtml("Could not reach Lito: " + err), "error");
    } finally {
      busy = false;
      sendBtn.disabled = false;
      input.focus();
      refreshStatus();
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    send(input.value);
  });

  chips.addEventListener("click", (e) => {
    const t = e.target;
    if (t && t.dataset && t.dataset.q) {
      send(t.dataset.q);
    }
  });

  showWelcome();
  loadHistory();
  refreshStatus();
  setInterval(refreshStatus, 15000);
})();

/* Lito desktop UI - talks to the local API (server.py, proxied by the Go app). */
"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  status: null,
  models: [],
  extensions: [],
  activeModel: "",
  sending: false,
  dlTimer: null,
  trainTimer: null,
  offline: false,
};

/* ------------------------------ API ------------------------------ */

async function api(path, opts) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok && res.status !== 400 && res.status !== 404 && res.status !== 409) {
    throw new Error(`HTTP ${res.status}`);
  }
  return res.json();
}

const post = (path, body) =>
  api(path, { method: "POST", body: JSON.stringify(body || {}) });

/* ------------------------------ Toasts ------------------------------ */

function toast(text, kind) {
  const el = document.createElement("div");
  el.className = "toast" + (kind ? " " + kind : "");
  el.textContent = text;
  $("toast-root").appendChild(el);
  setTimeout(() => {
    el.style.opacity = "0";
    el.style.transition = "opacity 0.3s";
    setTimeout(() => el.remove(), 350);
  }, 3400);
}

/* ------------------------------ Status ------------------------------ */

async function refreshStatus() {
  let s;
  try {
    s = await api("/api/version");
    setOffline(false);
  } catch (e) {
    setOffline(true);
    return;
  }
  state.status = s;
  $("brand-ver").textContent = s.version;
  $("model-chip-text").textContent =
    s.model ? `${s.model} · ${s.loaded ? s.params.toLocaleString() + " params" : "knowledge base"}` : "no model";
  const dot = $("model-dot");
  dot.className = "dot " + (s.loaded ? "on" : s.loading ? "loading" : "");
  $("foot-status").textContent = s.loading
    ? "loading model…"
    : s.loaded
      ? `model ready · ${s.params.toLocaleString()} params`
      : "knowledge base only (train or download a model)";
  $("send").disabled = state.sending || s.loading;
  if (s.search !== undefined) $("search-toggle").checked = s.search;
}

function setOffline(off) {
  if (off === state.offline) return;
  state.offline = off;
  $("offline-banner").classList.toggle("hidden", !off);
  if (!off) toast("Backend reconnected", "ok");
}

/* ------------------------------ Models ------------------------------ */

async function refreshModels() {
  try {
    const data = await api("/api/models");
    state.models = data.models || [];
  } catch (e) {
    state.models = [];
  }
  renderModels();
}

function renderModels() {
  const list = $("model-list");
  list.innerHTML = "";
  if (!state.models.length) {
    const empty = document.createElement("div");
    empty.className = "model-item";
    empty.innerHTML =
      '<span class="m-icon">◇</span><span class="m-body">' +
      '<span class="m-name">No models yet</span>' +
      '<span class="m-sub">download or train one</span></span>';
    list.appendChild(empty);
    return;
  }
  for (const m of state.models) {
    const item = document.createElement("div");
    item.className = "model-item" + (m.name === state.activeModel ? " active" : "");

    const icon = document.createElement("span");
    icon.className = "m-icon";
    icon.textContent = m.source === "trained" ? "★" : "☁";

    const body = document.createElement("span");
    body.className = "m-body";
    const name = document.createElement("div");
    name.className = "m-name";
    name.textContent = m.name;
    const sub = document.createElement("div");
    sub.className = "m-sub";
    sub.textContent =
      (m.source === "trained" ? "trained" : "downloaded") +
      (m.vocab_size ? ` · vocab ${m.vocab_size}` : "") +
      (m.trained_at ? ` · ${m.trained_at.slice(0, 10)}` : "");
    body.append(name, sub);

    item.append(icon, body);

    if (m.name !== "checkpoints") {
      const del = document.createElement("button");
      del.className = "m-del";
      del.title = "Delete model";
      del.textContent = "🗑";
      del.onclick = (e) => {
        e.stopPropagation();
        confirmDelete(m.name);
      };
      item.appendChild(del);
    }

    item.onclick = () => useModel(m.name);
    list.appendChild(item);
  }
}

async function useModel(name) {
  toast(`Loading ${name}…`);
  try {
    const r = await post("/api/models/use", { name });
    if (r.error) return toast(r.error, "error");
    state.activeModel = name;
    await Promise.all([refreshStatus(), refreshModels()]);
    systemMsg(`Switched to model “${name}”.`);
  } catch (e) {
    toast("Could not switch model: " + e.message, "error");
  }
}

function confirmDelete(name) {
  $("confirm-text").textContent =
    `Delete model “${name}”? The files in models/${name}/ will be removed. This cannot be undone.`;
  openModal("modal-confirm");
  $("confirm-ok").onclick = async () => {
    closeModal("modal-confirm");
    const r = await post("/api/models/delete", { name });
    if (r.error) return toast(r.error, "error");
    toast(`Deleted ${name}`, "ok");
    if (state.activeModel === name) state.activeModel = "";
    await Promise.all([refreshModels(), refreshStatus()]);
  };
}

/* ------------------------------ Extensions ------------------------------ */

async function refreshExtensions() {
  try {
    const data = await api("/api/extensions");
    state.extensions = data.extensions || [];
  } catch (e) {
    state.extensions = [];
  }
  renderExtensions();
}

function renderExtensions() {
  const list = $("ext-list");
  list.innerHTML = "";
  for (const ext of state.extensions) {
    const row = document.createElement("div");
    row.className = "ext-item";

    const body = document.createElement("span");
    const name = document.createElement("div");
    name.className = "e-name";
    name.textContent = ext.name;
    const desc = document.createElement("div");
    desc.className = "e-desc";
    desc.textContent = ext.description;
    body.append(name, desc);

    const label = document.createElement("label");
    label.className = "switch";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = ext.enabled;
    input.onchange = async () => {
      const r = await post("/api/extensions", { name: ext.name, enabled: input.checked });
      if (r.error) {
        input.checked = !input.checked;
        toast(r.error, "error");
      } else {
        toast(`${ext.name} ${input.checked ? "enabled" : "disabled"}`);
      }
    };
    const slider = document.createElement("span");
    slider.className = "slider";
    label.append(input, slider);

    row.append(body, label);
    list.appendChild(row);
  }
}

/* ------------------------------ Chat ------------------------------ */

function clearEmptyState() {
  const empty = $("empty-state");
  if (empty) empty.remove();
}

function appendMsg(role, text, meta) {
  clearEmptyState();
  const chat = $("chat");
  const msg = document.createElement("div");
  msg.className = "msg " + role;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "you" : "◆";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  msg.append(avatar, bubble);
  if (meta) {
    const m = document.createElement("div");
    m.className = "meta";
    m.textContent = meta;
    msg.appendChild(m);
  }
  chat.appendChild(msg);
  $("chat-wrap").scrollTop = $("chat-wrap").scrollHeight;
  return bubble;
}

function systemMsg(text) {
  appendMsg("lito", text);
}

function showTyping() {
  clearEmptyState();
  const chat = $("chat");
  const msg = document.createElement("div");
  msg.className = "msg lito";
  msg.id = "typing-row";
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "◆";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const typing = document.createElement("span");
  typing.className = "typing";
  for (let i = 0; i < 3; i++) typing.appendChild(document.createElement("span"));
  bubble.appendChild(typing);
  msg.append(avatar, bubble);
  chat.appendChild(msg);
  $("chat-wrap").scrollTop = $("chat-wrap").scrollHeight;
}

function hideTyping() {
  const row = $("typing-row");
  if (row) row.remove();
}

async function send() {
  const input = $("input");
  const text = input.value.trim();
  if (!text || state.sending) return;

  input.value = "";
  autosize();
  appendMsg("user", text);

  state.sending = true;
  $("send").disabled = true;
  showTyping();
  try {
    const r = await post("/api/chat", { message: text });
    hideTyping();
    appendMsg("lito", r.response || "(no response)");
  } catch (e) {
    hideTyping();
    appendMsg("lito", "(connection error — is the backend still running?)");
  } finally {
    state.sending = false;
    $("send").disabled = state.status && state.status.loading;
    input.focus();
  }
}

function autosize() {
  const input = $("input");
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 180) + "px";
}

/* ------------------------------ Download modal ------------------------------ */

function openModal(id) {
  $(id).classList.remove("hidden");
}
function closeModal(id) {
  $(id).classList.add("hidden");
}

async function openDownloadModal() {
  openModal("modal-download");
  const list = $("release-list");
  list.innerHTML = '<div class="skeleton-row"></div><div class="skeleton-row"></div>';
  const data = await api("/api/releases");
  list.innerHTML = "";
  if (data.error) {
    list.innerHTML = `<div class="release-item"><span class="r-body">` +
      `<span class="r-name">Could not reach GitHub: ${escapeHtml(data.error)}</span>` +
      `<span class="r-meta">Check your internet connection and try again.</span></span></div>`;
    return;
  }
  if (!data.releases.length) {
    list.innerHTML = `<div class="release-item"><span class="r-body">` +
      `<span class="r-tag">No published models yet</span>` +
      `<span class="r-meta">Train your own with the Train button!</span></span></div>`;
    return;
  }
  for (const r of data.releases) {
    const item = document.createElement("div");
    item.className = "release-item";

    const body = document.createElement("span");
    body.className = "r-body";
    const tag = document.createElement("div");
    tag.className = "r-tag";
    tag.textContent = r.tag;
    const name = document.createElement("div");
    name.className = "r-name";
    name.textContent = r.notes || r.name;
    const meta = document.createElement("div");
    meta.className = "r-meta";
    meta.textContent = `${r.date || "—"} · ${r.size_text || "?"}`;
    body.append(tag, name, meta);

    const btn = document.createElement("button");
    btn.className = "btn btn-primary btn-sm";
    btn.textContent = "⬇ Get";
    btn.onclick = () => startDownload(r.tag, btn);

    item.append(body, btn);
    list.appendChild(item);
  }
}

function startDownload(tag, btn) {
  btn.disabled = true;
  btn.textContent = "…";
  post("/api/download", { tag }).then((r) => {
    if (r.error) {
      btn.disabled = false;
      btn.textContent = "⬇ Get";
      toast(r.error, "error");
      return;
    }
    $("dl-progress").classList.remove("hidden");
    $("dl-tag").textContent = `Downloading ${tag}…`;
    pollDownload(tag);
  });
}

function pollDownload(tag) {
  if (state.dlTimer) clearInterval(state.dlTimer);
  state.dlTimer = setInterval(async () => {
    const s = await api("/api/download/status");
    const pct = s.total ? Math.round((s.done / s.total) * 100) : 0;
    $("dl-bar").style.width = pct + "%";
    $("dl-pct").textContent = s.total
      ? `${fmtBytes(s.done)} / ${fmtBytes(s.total)} (${pct}%)`
      : fmtBytes(s.done);
    if (!s.active) {
      clearInterval(state.dlTimer);
      state.dlTimer = null;
      $("dl-progress").classList.add("hidden");
      $("dl-bar").style.width = "0%";
      if (s.error) {
        toast(`Download failed: ${s.error}`, "error");
      } else {
        toast(`Downloaded ${s.model} ✓`, "ok");
        state.activeModel = s.model;
        await Promise.all([refreshModels(), refreshStatus()]);
        systemMsg(`Downloaded model “${s.model}” — it is now active.`);
      }
      closeModal("modal-download");
    }
  }, 500);
}

function fmtBytes(n) {
  if (!n) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return n.toFixed(n < 10 && i > 0 ? 1 : 0) + " " + units[i];
}

/* ------------------------------ Train modal ------------------------------ */

function openTrainModal() {
  openModal("modal-train");
  // Show any existing log; only poll while a run is actually active.
  api(`/api/train/status?since=0`).then((s) => {
    trainSeen = 0;
    const log = $("train-log");
    if (s.lines && s.lines.length) {
      log.classList.remove("hidden");
      log.textContent = s.lines.join("\n") + "\n";
      trainSeen = s.total;
    }
    if (s.running) {
      pollTrain();
    } else {
      $("btn-start-train").disabled = false;
    }
  });
}

async function startTraining() {
  const params = {};
  document.querySelectorAll("#train-form [data-k]").forEach((el) => {
    if (el.type === "checkbox") {
      if (el.checked) params[el.dataset.k] = true;
    } else if (el.value.trim()) {
      params[el.dataset.k] = el.value.trim();
    }
  });
  const r = await post("/api/train", params);
  if (r.error) return toast(r.error, "error");
  toast("Training started", "ok");
  $("train-log").classList.remove("hidden");
  $("train-log").textContent = "$ " + (r.command || "") + "\n";
  $("btn-start-train").disabled = true;
  pollTrain();
}

let trainSeen = 0;
function pollTrain() {
  if (state.trainTimer) return;
  state.trainTimer = setInterval(async () => {
    try {
      const s = await api(`/api/train/status?since=${trainSeen}`);
      const log = $("train-log");
      if (s.lines && s.lines.length) {
        trainSeen = s.total;
        log.classList.remove("hidden");
        log.textContent += s.lines.join("\n") + "\n";
        log.scrollTop = log.scrollHeight;
      }
      if (!s.running) {
        if (s.returncode !== null && state.trainTimer) {
          if (s.returncode === 0) {
            toast("Training finished ✓", "ok");
            state.activeModel = "checkpoints";
            await Promise.all([refreshModels(), refreshStatus()]);
            systemMsg("Training finished — the new model in “checkpoints” is now active.");
          } else {
            toast(`Training exited with code ${s.returncode}`, "error");
          }
          $("btn-start-train").disabled = false;
        }
        clearInterval(state.trainTimer);
        state.trainTimer = null;
      }
    } catch (e) { /* transient */ }
  }, 900);
}

/* ------------------------------ Misc ------------------------------ */

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

/* ------------------------------ Wire up ------------------------------ */

window.addEventListener("DOMContentLoaded", () => {
  $("send").onclick = send;
  $("input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  $("input").addEventListener("input", autosize);

  $("btn-download").onclick = openDownloadModal;
  $("btn-refresh").onclick = async () => {
    await Promise.all([refreshModels(), refreshStatus()]);
    toast("Refreshed");
  };
  $("btn-train").onclick = openTrainModal;
  $("btn-start-train").onclick = startTraining;
  $("btn-new-chat").onclick = async () => {
    await post("/api/reset");
    const chat = $("chat");
    chat.innerHTML = "";
    toast("Conversation cleared", "ok");
    $("input").focus();
  };

  $("search-toggle").onchange = async (e) => {
    const r = await post("/api/search", { enabled: e.target.checked });
    toast(`Web search ${r.search ? "on" : "off"}`);
  };

  document.querySelectorAll(".modal-close, [data-close]").forEach((el) => {
    el.onclick = () => closeModal(el.dataset.close);
  });
  document.querySelectorAll(".modal-backdrop").forEach((bd) => {
    bd.addEventListener("mousedown", (e) => {
      if (e.target === bd) bd.classList.add("hidden");
    });
  });
  $("confirm-cancel").onclick = () => closeModal("modal-confirm");

  // initial load + polling
  (async () => {
    await Promise.all([refreshStatus(), refreshModels(), refreshExtensions()]);
    // wait for the startup model, then mark it active
    const waitForModel = setInterval(async () => {
      await refreshStatus();
      if (state.status && !state.status.loading) {
        clearInterval(waitForModel);
        state.activeModel = state.status.model || "";
        refreshModels();
        $("input").focus();
      }
    }, 700);
  })();

  setInterval(refreshStatus, 5000);
});

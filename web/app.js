// web/app.js
// Vanilla JS, no build step -- this is a zero-config static site for
// Vercel. BACKEND_URL comes from config.js (the one file you edit).

const SLOT_COLORS = {
  onepiece: "#e0b34d",
  topwear: "#e8e1d2",
  bottomwear: "#7fa8c9",
  footwear: "#c98a4a",
  layer: "#8fae7a",
  accessory: "#d4708a",
};
const FEATURE_LABELS = {
  embedding_cos: "Visual + text",
  color_score: "Color harmony",
  graph_score: "Co-occurrence",
  same_occasion: "Same occasion",
  wear_match: "Wear-type match",
};

let sessionId = null;
let graphLoaded = false;

function showError(msg) {
  const el = document.getElementById("error-banner");
  el.textContent = msg;
  el.style.display = "block";
}
function clearError() {
  document.getElementById("error-banner").style.display = "none";
}

function switchTab(tab) {
  document.querySelectorAll("nav.tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === `panel-${tab}`));
  if (tab === "atlas" && !graphLoaded) loadGraph();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderBreakdown(breakdown) {
  if (!breakdown) return "";
  const rows = Object.entries(FEATURE_LABELS)
    .map(([key, label]) => {
      const raw = breakdown[key];
      if (raw === null || raw === undefined) return "";
      const pct = Math.max(0, Math.min(1, raw)) * 100;
      return `<div class="row">
        <span class="label">${label}</span>
        <span class="bar-track"><span class="bar-fill" style="width:${pct.toFixed(0)}%"></span></span>
        <span class="val">${raw.toFixed(2)}</span>
      </div>`;
    })
    .join("");
  return `<div class="breakdown">${rows}</div>`;
}

function renderOutfitCard(outfit) {
  const items = outfit.items
    .map(
      (it) => `<div class="outfit-item">
        <img src="images/${it.image.replace(/^images\//, "")}" alt="${escapeHtml(it.name)}"
             onerror="this.style.opacity=0.15">
        <div class="name">${escapeHtml(it.name)}</div>
        ${it.price_inr != null ? `<div class="price">\u20B9${it.price_inr}</div>` : ""}
      </div>`
    )
    .join("");

  const ref = outfit.style_reference;
  const refHtml = ref && ref.stylist_rationale
    ? `<div class="style-ref">Style reference (${escapeHtml(ref.theme || "")}): ${escapeHtml(ref.stylist_rationale)}</div>`
    : "";

  return `<div class="outfit-card">
    <div class="meta">compat ${outfit.compat_score?.toFixed(2) ?? "-"} &middot; relevance ${outfit.relevance_score?.toFixed(2) ?? "-"}</div>
    <div class="outfit-items">${items}</div>
    ${renderBreakdown(outfit.feature_breakdown)}
    ${refHtml}
  </div>`;
}

function renderOutfits(container, outfits) {
  if (!outfits || outfits.length === 0) {
    container.innerHTML = `<p style="color:var(--ink-soft)">No outfits returned -- try loosening the occasion or style description.</p>`;
    return;
  }
  container.innerHTML = outfits.map(renderOutfitCard).join("");
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------
function appendChatMessage(role, html) {
  const log = document.getElementById("chat-log");
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.innerHTML = html;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

async function sendChat() {
  const input = document.getElementById("chat-input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  clearError();

  if (!sessionId) sessionId = crypto.randomUUID();
  appendChatMessage("user", escapeHtml(text));
  const thinkingId = "thinking-" + Date.now();
  appendChatMessage("assistant", `<span id="${thinkingId}">Thinking…</span>`);

  try {
    const res = await fetch(`${BACKEND_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: text }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    const node = document.getElementById(thinkingId).parentElement;
    let html = escapeHtml(data.reply);
    if (data.outfits && data.outfits.length > 0) {
      html += `<div style="margin-top:10px">${data.outfits.map(renderOutfitCard).join("")}</div>`;
    }
    node.innerHTML = html;
  } catch (e) {
    document.getElementById(thinkingId).parentElement.innerHTML = `Request failed: ${escapeHtml(e.message)}`;
    showError(`Chat request failed: ${e.message}. Is the backend running and GROQ_API_KEY set?`);
  }
}

// ---------------------------------------------------------------------------
// Direct search (bypasses the LLM entirely)
// ---------------------------------------------------------------------------
async function runSearch() {
  clearError();
  const gender = document.getElementById("search-gender").value;
  const occasion = document.getElementById("search-occasion").value;
  const styleText = document.getElementById("search-style").value || "smart casual outfit";
  const anchorItem = document.getElementById("search-anchor").value || null;
  const resultsEl = document.getElementById("search-results");
  resultsEl.innerHTML = `<p style="color:var(--ink-soft)">Searching…</p>`;

  try {
    const res = await fetch(`${BACKEND_URL}/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ gender, occasion, style_text: styleText, anchor_item: anchorItem }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderOutfits(resultsEl, data.outfits);
  } catch (e) {
    resultsEl.innerHTML = "";
    showError(`Search failed: ${e.message}. Is the backend running at ${BACKEND_URL}?`);
  }
}

// ---------------------------------------------------------------------------
// Compatibility Atlas (real co-occurrence graph from data_pipeline.py)
// ---------------------------------------------------------------------------
async function loadGraph() {
  clearError();
  try {
    const res = await fetch(`${BACKEND_URL}/graph`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    graphLoaded = true;

    const nodes = new vis.DataSet(
      data.nodes.map((n) => ({
        id: n.id,
        label: n.label.length > 22 ? n.label.slice(0, 20) + "…" : n.label,
        color: { background: SLOT_COLORS[n.slot] || "#888", border: "#0008" },
        font: { color: "#1f2d3d", size: 11 },
        shape: "dot",
        size: 10,
      }))
    );
    const edges = new vis.DataSet(
      data.edges.map((e) => ({
        from: e.source,
        to: e.target,
        width: 1 + e.weight,
        color: { color: "#e3d9c466" },
      }))
    );

    new vis.Network(document.getElementById("graph-canvas"), { nodes, edges }, {
      physics: { stabilization: true, barnesHut: { gravitationalConstant: -3000 } },
      interaction: { hover: true },
    });
  } catch (e) {
    showError(`Could not load compatibility graph: ${e.message}. Is the backend running at ${BACKEND_URL}?`);
  }
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("nav.tabs button").forEach((b) =>
    b.addEventListener("click", () => switchTab(b.dataset.tab))
  );
  document.getElementById("chat-send").addEventListener("click", sendChat);
  document.getElementById("chat-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendChat();
  });
  document.getElementById("search-run").addEventListener("click", runSearch);
});

"""The single page of the dry-run web app (served by web.py). Plain HTML, CSS and JavaScript, no CDN."""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>little-trading dry run</title>
<link rel="manifest" href="/manifest.webmanifest">
<meta name="theme-color" content="#2563eb">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="Trading">
<link rel="icon" href="/icon-192.png" type="image/png">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
<style>
:root {
  --bg: #f6f7f9; --panel: #ffffff; --text: #1d2330; --muted: #667085; --line: #e3e6eb;
  --accent: #2563eb; --up: #15803d; --down: #b91c1c; --warn: #b45309; --chip: #eef2f7;
  --s1: #2563eb; --s2: #9333ea; --s3: #0d9488; --s4: #ea580c; --s5: #db2777; --s6: #65a30d;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #0f1115; --panel: #171a21; --text: #e6e8ec; --muted: #9aa3b2; --line: #2a2f3a;
          --accent: #60a5fa; --up: #4ade80; --down: #f87171; --warn: #fbbf24; --chip: #232833;
          --s1: #60a5fa; --s2: #c084fc; --s3: #2dd4bf; --s4: #fb923c; --s5: #f472b6; --s6: #a3e635; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { position: sticky; top: 0; z-index: 2; background: var(--panel); border-bottom: 1px solid var(--line);
         padding: 10px 16px; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
header h1 { font-size: 16px; margin: 0 12px 0 0; }
.pill { background: var(--chip); border-radius: 999px; padding: 2px 10px; font-size: 12px; color: var(--muted); }
.pill.ok { color: var(--up); } .pill.bad { color: var(--down); } .pill.warn { color: var(--warn); }
nav { display: flex; flex-wrap: wrap; gap: 4px; padding: 8px 16px; background: var(--panel); border-bottom: 1px solid var(--line); }
nav button { border: 0; background: none; color: var(--muted); padding: 6px 10px; border-radius: 6px; cursor: pointer; font: inherit; }
nav button.on { background: var(--chip); color: var(--text); font-weight: 600; }
main { padding: 16px; max-width: 1400px; margin: 0 auto; }
.grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); margin-bottom: 12px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; margin-bottom: 12px; overflow-x: auto; }
.card h2 { font-size: 14px; margin: 0 0 8px; }
.kpi .v { font-size: 22px; font-weight: 650; font-variant-numeric: tabular-nums; }
.kpi .l { color: var(--muted); font-size: 12px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 500; font-size: 12px; }
tr.click { cursor: pointer; } tr.click:hover { background: var(--chip); }
.up { color: var(--up); } .down { color: var(--down); } .muted { color: var(--muted); }
.chip { display: inline-block; background: var(--chip); border-radius: 4px; padding: 0 6px; margin: 1px; font-size: 12px; }
.bar { display: inline-block; height: 8px; border-radius: 2px; vertical-align: middle; }
pre { white-space: pre-wrap; word-break: break-word; background: var(--chip); padding: 10px; border-radius: 8px; font-size: 12px; }
select, input, .btn { font: inherit; background: var(--panel); color: var(--text); border: 1px solid var(--line); border-radius: 6px; padding: 4px 8px; }
.btn { cursor: pointer; }
svg text { fill: var(--muted); font-size: 11px; }
.legend span { margin-right: 12px; font-size: 12px; }
.two { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); }
@media (max-width: 560px) { .two { grid-template-columns: 1fr; } }
.hidden { display: none; }
</style>
</head>
<body>
<header>
  <h1>little-trading · dry run</h1>
  <span class="pill" id="p-version">…</span>
  <span class="pill" id="p-market">market …</span>
  <span class="pill" id="p-round">…</span>
  <span class="pill" id="p-models">…</span>
  <span class="pill" id="p-errors">…</span>
  <span class="pill muted">paper trading · fake money · real prices and news</span>
</header>
<nav id="tabs"></nav>
<main id="main"></main>
<script>
if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});  // installable as a phone app
const TABS = ["Overview", "Universe", "Decisions", "News", "Models", "Lessons", "Live evaluation", "Logs", "Software", "Lab", "Report for Claude"];
let tab = localStorageGet("tab") || "Overview", symbol = null;
const COLORS = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6"].map(v => getComputedStyle(document.documentElement).getPropertyValue(v).trim());

function localStorageGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
function localStorageSet(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const money = x => x == null ? "–" : Number(x).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
const pct = (x, d = 2) => x == null ? "–" : (x >= 0 ? "+" : "") + (x * 100).toFixed(d) + "%";
const cls = x => x > 0 ? "up" : x < 0 ? "down" : "";
const when = ts => ts ? new Date(ts).toLocaleString() : "–";
async function get(path) { const r = await fetch(path); if (!r.ok) throw new Error(path + " " + r.status); return path.includes("report") ? r.text() : r.json(); }

function line(series, opts = {}) {
  // series: [{name, points: [[x, y], ...]}]; x as ms epoch
  const W = opts.w || 900, H = opts.h || 220, P = 36;
  const pts = series.flatMap(s => s.points);
  if (!pts.length) return '<p class="muted">No data yet.</p>';
  let [x0, x1] = [Math.min(...pts.map(p => p[0])), Math.max(...pts.map(p => p[0]))];
  let [y0, y1] = [Math.min(...pts.map(p => p[1])), Math.max(...pts.map(p => p[1]))];
  if (opts.zero) { y0 = Math.min(y0, 0); y1 = Math.max(y1, 0); }
  if (x1 === x0) x1 = x0 + 1; if (y1 === y0) { y1 += 1; y0 -= 1; }
  const X = x => P + (x - x0) / (x1 - x0) * (W - P - 8), Y = y => H - 20 - (y - y0) / (y1 - y0) * (H - 30);
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img">`;
  for (let k = 0; k <= 4; k++) {
    const y = y0 + (y1 - y0) * k / 4;
    svg += `<line x1="${P}" x2="${W - 8}" y1="${Y(y)}" y2="${Y(y)}" stroke="currentColor" opacity=".08"/><text x="2" y="${Y(y) + 4}">${opts.fmt ? opts.fmt(y) : y.toFixed(2)}</text>`;
  }
  if (opts.zero) svg += `<line x1="${P}" x2="${W - 8}" y1="${Y(0)}" y2="${Y(0)}" stroke="currentColor" opacity=".3"/>`;
  svg += `<text x="${P}" y="${H - 4}">${new Date(x0).toLocaleDateString()}</text><text x="${W - 90}" y="${H - 4}">${new Date(x1).toLocaleDateString()}</text>`;
  series.forEach((s, i) => {
    const c = s.color || COLORS[i % COLORS.length];
    const d = s.points.map((p, j) => (j ? "L" : "M") + X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1)).join("");
    svg += `<path d="${d}" fill="none" stroke="${c}" stroke-width="2" ${s.dash ? 'stroke-dasharray="5 4"' : ""}/>`;
    (s.marks || []).forEach(m => svg += `<circle cx="${X(m[0])}" cy="${Y(m[1])}" r="4" fill="${m[2]}"><title>${esc(m[3] || "")}</title></circle>`);
  });
  svg += "</svg>";
  const legend = series.length > 1 ? '<div class="legend">' + series.map((s, i) => `<span style="color:${s.color || COLORS[i % COLORS.length]}">● ${esc(s.name)}</span>`).join("") + "</div>" : "";
  return legend + svg;
}
const scoreBar = s => `<span class="bar" style="width:${Math.min(60, Math.abs(s) * 60)}px;background:var(${s >= 0 ? "--up" : "--down"})"></span> <span class="${cls(s)}">${s >= 0 ? "+" : ""}${Number(s).toFixed(3)}</span>`;

async function header() {
  try {
    const s = await get("/api/status");
    document.getElementById("p-version").textContent = "v" + s.software + " · " + s.horizon + " horizon";
    const m = document.getElementById("p-market");
    m.textContent = !s.ready ? (s.phase || "catching up…") : s.market && s.market.open ? "market open" : "market closed";
    m.className = "pill " + (!s.ready ? "warn" : s.market && s.market.open ? "ok" : "");
    document.getElementById("p-round").textContent = `round ${s.portfolio.rounds || 0} · ${s.counts.news} headlines`;
    document.getElementById("p-models").textContent = s.models_trained_at ? "models " + when(s.models_trained_at) : "models not trained";
    const errs = s.loops.filter(l => l.last_error).length;
    const e = document.getElementById("p-errors");
    e.textContent = errs ? errs + " loop errors" : "loops ok"; e.className = "pill " + (errs ? "bad" : "ok");
    return s;
  } catch (err) { document.getElementById("p-market").textContent = "server unreachable"; }
}

const views = {
  async "Overview"() {
    const [s, eq, pos, tr] = await Promise.all([get("/api/status"), get("/api/equity"), get("/api/positions"), get("/api/trades?limit=15")]);
    const last = s.last || {}, start = s.portfolio.starting || 0, cur = s.portfolio.currency, st = tr.stats;
    const growth = last.equity ? last.equity / start - 1 : null, bench = last.benchmark ? last.benchmark / start - 1 : null;
    const kpi = (l, v, c = "") => `<div class="card kpi"><div class="l">${l}</div><div class="v ${c}">${v}</div></div>`;
    const year = new Date().getUTCFullYear(), tax = (st.tax_by_year || {})[year];
    const brokers = Object.entries(st.fees_at_each_broker || {});
    return `<div class="grid">${kpi(`Equity (${esc(cur)})`, money(last.equity))}${kpi("Total growth", pct(growth), cls(growth))}
      ${kpi("SPY held", pct(bench), cls(bench))}${kpi("vs SPY", growth == null || bench == null ? "–" : pct(growth - bench), cls(growth - bench))}
      ${kpi("Cash", money(last.cash))}${kpi("Holding value", money(last.invested))}${kpi("Starting", money(start))}${kpi("Rounds", s.portfolio.rounds || 0)}
      ${kpi("Fees paid", money(st.fees))}${cur === "SEK" ? kpi(`Tax ${year} (est.)`, money(tax ? tax.tax : 0)) : ""}</div>
      <div class="card"><h2>Costs at ${esc(s.portfolio.broker)}</h2><p class="muted">Amounts in ${esc(cur)}${s.portfolio.usdsek ? `; USD/SEK ${Number(s.portfolio.usdsek).toFixed(4)} at the last round` : ""}. Prices per share are in USD.
      ${cur === "SEK" ? `Courtage ${money(st.courtage)}, currency conversion ${money(st.fx_fees)}. Tax is 30% of each year's net realised gain (aktie- och fonddepå), paid with the next year's tax return, not taken from the cash.` : "The USD account: a flat 0.1% fee, no conversion."}</p>
      ${brokers.length && st.fees ? `<table><tr><th>The same fills at</th><th>Fees (SEK)</th></tr>${brokers.map(([b, f]) => `<tr><td>${esc(b)}</td><td>${money(f)}</td></tr>`).join("")}</table>` : ""}
      ${cur === "SEK" && Object.keys(st.tax_by_year || {}).length ? `<table><tr><th>Year</th><th>Gains</th><th>Losses</th><th>Net</th><th>Tax 30%</th></tr>${Object.entries(st.tax_by_year).map(([y, t]) => `<tr><td>${y}</td><td>${money(t.gains)}</td><td>${money(t.losses)}</td><td class="${cls(t.net)}">${money(t.net)}</td><td>${money(t.tax)}</td></tr>`).join("")}</table>` : ""}</div>
      <div class="card"><h2>Equity vs SPY bought and held</h2><p class="muted">A point per trading round (every 15 minutes while the US market is open), after each catch-up and after the close. ${eq.length} points, the latest ${when(eq.length ? eq[eq.length - 1].ts : null)}${s.market && s.market.open ? "" : ": the market is closed, so it moves again at the next open"}.</p>${line([
        {name: "dry-run equity", points: eq.map(r => [Date.parse(r.ts), r.equity])},
        {name: "SPY bought and held", dash: true, points: eq.filter(r => r.benchmark).map(r => [Date.parse(r.ts), r.benchmark])}], {fmt: money})}</div>
      <div class="two"><div class="card"><h2>Positions</h2><table><tr><th>Symbol</th><th>Qty</th><th>Entry (USD)</th><th>Price (USD)</th><th>Value (${esc(cur)})</th><th>P/L</th><th>Opened</th></tr>
      ${pos.map(p => `<tr><td>${esc(p.symbol)}</td><td>${p.qty}</td><td>${money(p.avg_price)}</td><td>${money(p.price)}</td><td>${money(p.value)}</td><td class="${cls(p.unrealised)}">${pct(p.unrealised)}</td><td>${esc(p.opened_day)}</td></tr>`).join("") || '<tr><td colspan="7" class="muted">No open positions.</td></tr>'}</table></div>
      <div class="card"><h2>Recent trades</h2><table><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Price (USD)</th><th>Fees (${esc(cur)})</th><th>P/L (${esc(cur)})</th><th>Reason</th></tr>
      ${tr.trades.map(t => `<tr><td>${when(t.ts)}</td><td>${esc(t.symbol)}</td><td>${t.side}</td><td>${t.qty}</td><td>${money(t.price)}</td><td>${money(t.fee)}</td><td class="${cls(t.pnl)}">${t.side === "SELL" ? money(t.pnl) : ""}</td><td>${esc(t.reason)}</td></tr>`).join("") || '<tr><td colspan="8" class="muted">No trades yet.</td></tr>'}</table></div></div>`;
  },

  async "Universe"() {
    const u = await get("/api/universe");
    const sel = u.selection || {fixed: [], dynamic: [], dropped: [], candidates: 0};
    const q = u.quality;
    return `<div class="grid"><div class="card kpi"><div class="l">Watched</div><div class="v">${u.watched.length}</div></div>
      <div class="card kpi"><div class="l">Fixed</div><div class="v">${sel.fixed.length}</div></div>
      <div class="card kpi"><div class="l">Dynamic picks</div><div class="v">${sel.dynamic.length}</div></div>
      <div class="card kpi"><div class="l">Candidates ranked</div><div class="v">${u.ranking.length} / ${sel.candidates}</div></div>
      <div class="card kpi"><div class="l">Picks vs left out</div><div class="v ${cls((q.picked_mean ?? 0) - (q.left_out_mean ?? 0))}">${q.days ? pct(q.picked_mean) + " / " + pct(q.left_out_mean) : "–"}</div></div></div>
      <div class="card"><h2>Fixed</h2>${sel.fixed.map(s => `<span class="chip">${esc(s)}</span>`).join("")}
      ${u.held.filter(s => !u.watched.slice(0, sel.fixed.length + sel.dynamic.length).includes(s)).length ? `<p class="muted">Also watched because held: ${u.held.map(esc).join(" ")}</p>` : ""}</div>
      <div class="two"><div class="card"><h2>Dynamic picks (selected ${when(sel.at)})</h2><p class="muted">Ranked by the decision score, then by dollar volume (while no model has skill every score is 0). A pick stays while it ranks in the top 45.</p>
      <table><tr><th>Rank</th><th>Symbol</th><th>Score</th><th>Why</th><th>Picked since</th></tr>${sel.dynamic.map(d => `<tr class="click" data-sym="${esc(d.symbol)}"><td>${d.rank}</td><td><b>${esc(d.symbol)}</b></td><td>${scoreBar(d.score)}</td><td class="muted">${esc(d.reason)}</td><td>${when(d.since)}</td></tr>`).join("") || '<tr><td colspan="5" class="muted">No selection yet.</td></tr>'}</table></div>
      <div class="card"><h2>Changes</h2><table>${u.changes.map(c => `<tr><td class="muted" style="white-space:nowrap">${when(c.ts)}</td><td>${c.added.map(s => `<span class="chip up">+${esc(s)}</span>`).join("")}${c.removed.map(s => `<span class="chip down">−${esc(s)}</span>`).join("")}${!c.added.length && !c.removed.length ? '<span class="muted">no change</span>' : ""}</td></tr>`).join("")}</table>
      <h2>Do picks beat the candidates left out?</h2><p class="muted">Close on the selection day to the close one holding period later; ${q.days} matured selection days (${q.picked_n} picked, ${q.left_out_n} left out).</p>
      ${line([{name: "picked", points: q.per_day.map(d => [Date.parse(d.day), d.picked])}, {name: "left out", dash: true, points: q.per_day.map(d => [Date.parse(d.day), d.left_out])}], {zero: true, h: 160, fmt: y => (y * 100).toFixed(1) + "%"})}</div></div>
      <div class="card"><h2>Full ranking of the latest selection</h2><table><tr><th>Rank</th><th>Symbol</th><th>Score</th><th>Picked</th></tr>${u.ranking.map(r => `<tr><td>${r.rank}</td><td>${esc(r.symbol)}</td><td>${scoreBar(r.score)}</td><td>${r.picked ? "yes" : ""}</td></tr>`).join("")}</table>
      <details><summary class="muted">Dropped candidates (${sel.dropped.length})</summary><pre>${esc(sel.dropped.join("\n"))}</pre></details></div>`;
  },

  async "Decisions"() {
    const rows = await get("/api/latest");
    let html = `<div class="card"><h2>Latest decision per symbol</h2><p class="muted">score = conviction × skill-weighted votes of the models with measured skill (the technical rule is one of them); no skill means score 0 and waiting. A BUY or SELL signal (≥ +0.15 / ≤ −0.15) must repeat 3 rounds in a row before it is acted on; buys are paced (1 per round, 2 per day). Stop loss and max hold act at once. Click a row for its history and inputs.</p>
      <table><tr><th>Symbol</th><th>Action</th><th>Score</th><th>Technical (raw)</th><th>Models</th><th>Votes (weight)</th><th>Headlines</th><th>Outcome</th><th>At</th></tr>
      ${rows.map(d => `<tr class="click" data-sym="${esc(d.symbol)}"><td><b>${esc(d.symbol)}</b></td><td>${d.action}</td><td>${scoreBar(d.score)}</td>
        <td class="${cls(d.components.technical)}">${d.components.technical.toFixed(3)}</td><td class="${cls(d.components.models)}">${d.components.models.toFixed(3)} <span class="muted">× ${d.components.conviction ?? d.components.model_share}</span></td>
        <td>${Object.entries(d.components.votes).map(([k, v]) => `<span class="chip ${cls(v)}">${esc(k)} ${v.toFixed(2)} (${d.components.weights[k]})</span>`).join("")}</td>
        <td>${d.inputs.news_count}</td><td>${esc(d.outcome)} <span class="muted">${esc(d.detail)}</span></td><td>${when(d.ts)}</td></tr>`).join("") || '<tr><td colspan="9" class="muted">No decision yet: rounds run while the US market is open.</td></tr>'}</table></div>`;
    if (symbol) html += await symbolDetail(symbol);
    return html;
  },

  async "News"() {
    const sym = localStorageGet("newsSym") || "";
    const s = await get("/api/status");
    const rows = await get("/api/news?limit=300" + (sym ? "&symbol=" + encodeURIComponent(sym) : ""));
    return `<div class="card"><h2>Headlines</h2><p>Symbol <select id="news-sym"><option value="">all</option>${s.symbols.map(x => `<option ${x === sym ? "selected" : ""}>${esc(x)}</option>`).join("")}</select>
      <span class="muted">lexicon: fixed word list in [−1, 1]; model: learned P(stock beats the market over the horizon)</span></p>
      <table><tr><th>Published</th><th>Symbol</th><th>Headline</th><th>Source</th><th>Lexicon</th><th>Model</th></tr>
      ${rows.map(n => `<tr><td>${when(n.published)}${n.day_only ? ' <span class="muted">(date)</span>' : ""}</td><td>${esc(n.symbol)}</td><td>${/^https?:\/\//.test(n.link || "") ? `<a href="${esc(n.link)}" target="_blank" rel="noopener noreferrer">${esc(n.title)}</a>` : esc(n.title)}</td>
        <td class="muted">${esc(n.source)}</td><td class="${cls(n.lexicon)}">${Number(n.lexicon).toFixed(2)}</td><td class="${cls((n.model_p ?? .5) - .5)}">${n.model_p == null ? "–" : n.model_p.toFixed(3)}</td></tr>`).join("") || '<tr><td colspan="6" class="muted">No headlines yet.</td></tr>'}</table></div>`;
  },

  async "Models"() {
    const m = await get("/api/models");
    const names = Object.keys(m.history);
    const skill = names.map((n, i) => ({name: n, points: m.history[n].map(h => [Date.parse(h.ts), h.skill || 0])}));
    const auc = names.map((n, i) => ({name: n, points: m.history[n].map(h => [Date.parse(h.ts), h.auc || .5])}));
    let html = `<div class="card"><h2>Leaderboard</h2><p class="muted">Validation is the latest quarter of the history, after an embargo. Skill = Brier skill score against always predicting the base rate (> 0 beats it). Weight = skill when there are enough validation samples. Conviction is <b>${Math.round((m.conviction || 0) * 100)}%</b>: decisions act on that share of the skill-weighted vote (0% means no model has skill and the trader waits). Trained ${when(m.trained_at)}.</p>
      <table><tr><th>Model</th><th>Weight</th><th>Skill</th><th>AUC</th><th>Accuracy</th><th>Majority</th><th>Validation n</th><th>Train n</th><th>Label</th></tr>
      ${m.latest.map(r => `<tr><td><b>${esc(r.model)}</b></td><td>${r.weight}</td><td class="${cls(r.metrics.skill)}">${(r.metrics.skill ?? 0).toFixed(4)}</td><td>${(r.metrics.auc ?? .5).toFixed(3)}</td>
        <td>${pct(r.metrics.accuracy, 1).replace("+", "")}</td><td>${pct(r.metrics.majority, 1).replace("+", "")}</td><td>${r.val_n}</td><td>${r.train_n}</td><td class="muted">${esc(r.details.label || "")}${r.details.continued_from ? `<br>continued from ${when(r.details.continued_from)}, weights moved ${r.details.weight_change}` : ""}</td></tr>`).join("") || '<tr><td colspan="9" class="muted">Not trained yet (the catch-up runs first).</td></tr>'}</table></div>
      <div class="two"><div class="card"><h2>Skill over training runs</h2>${line(skill, {zero: true, fmt: y => y.toFixed(3)})}</div>
      <div class="card"><h2>AUC over training runs</h2>${line(auc, {fmt: y => y.toFixed(2)})}</div></div>`;
    for (const r of m.latest) {
      const d = r.details;
      if (d.weights) {
        const ws = Object.entries(d.weights).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])), mx = Math.max(...ws.map(w => Math.abs(w[1]))) || 1;
        html += `<div class="card"><h2>${esc(r.model)}: learned feature weights</h2><table>${ws.map(([k, v]) => `<tr><td>${esc(k)}</td><td><span class="bar" style="width:${Math.abs(v) / mx * 200}px;background:var(${v >= 0 ? "--up" : "--down"})"></span> ${v.toFixed(4)}</td></tr>`).join("")}</table></div>`;
      }
      if (d.top_tokens) html += `<div class="card"><h2>${esc(r.model)}: tokens the model learned</h2><p><b class="up">Point up:</b> ${d.top_tokens.up.map(t => `<span class="chip">${esc(t[0])} ${t[1]}</span>`).join("")}</p><p><b class="down">Point down:</b> ${d.top_tokens.down.map(t => `<span class="chip">${esc(t[0])} ${t[1]}</span>`).join("")}</p><p class="muted">vocabulary ${d.vocabulary}</p></div>`;
    }
    return html;
  },

  async "Lessons"() {
    const rows = await get("/api/lessons");
    return `<div class="card"><h2>Lessons learned</h2><p class="muted">Written by the server after each training run and each daily evaluation.</p><table>
      ${rows.map(l => `<tr><td class="muted" style="white-space:nowrap">${when(l.ts)}</td><td><span class="chip">${esc(l.category)}</span></td><td>${esc(l.text)}</td></tr>`).join("") || '<tr><td class="muted">No lessons yet.</td></tr>'}</table></div>`;
  },

  async "Live evaluation"() {
    const e = await get("/api/evaluation");
    const t = e.trades;
    return `<div class="card"><h2>Matured decisions</h2><p class="muted">${e.matured} decisions have ${e.horizon_sessions} sessions of outcome. Forward return = close ${e.horizon_sessions} sessions later over the decision price.</p>
      <table><tr><th>Action</th><th>n</th><th>Mean forward return</th><th>Hit rate</th></tr>${Object.entries(e.by_action).map(([a, x]) => `<tr><td>${a}</td><td>${x.n}</td><td class="${cls(x.mean_forward_return)}">${pct(x.mean_forward_return)}</td><td>${x.hit_rate == null ? "–" : Math.round(x.hit_rate * 100) + "%"}</td></tr>`).join("")}</table></div>
      <div class="card"><h2>Does each component separate winners from losers?</h2><table><tr><th>Component</th><th>After up votes</th><th>n</th><th>After down votes</th><th>n</th></tr>
      ${Object.entries(e.by_vote).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="${cls(v.up_mean)}">${pct(v.up_mean)}</td><td>${v.up_n}</td><td class="${cls(v.down_mean)}">${pct(v.down_mean)}</td><td>${v.down_n}</td></tr>`).join("")}</table></div>
      <div class="card"><h2>Round trips</h2><p>${t.round_trips} round trips · win rate ${t.win_rate == null ? "–" : Math.round(t.win_rate * 100) + "%"} · avg win ${money(t.avg_win)} · avg loss ${money(t.avg_loss)} · profit factor ${t.profit_factor ?? "–"} · fees ${money(t.fees)}</p>
      <table><tr><th>Exit</th><th>Count</th><th>P/L</th></tr>${Object.entries(t.by_exit).map(([k, x]) => `<tr><td>${esc(k)}</td><td>${x.count}</td><td class="${cls(x.pnl)}">${money(x.pnl)}</td></tr>`).join("")}</table></div>`;
  },

  async "Logs"() {
    const kind = localStorageGet("logKind") || "", level = localStorageGet("logLevel") || "";
    const rows = await get(`/api/events?limit=300${kind ? "&kind=" + kind : ""}${level ? "&level=" + level : ""}`);
    const kinds = ["", "server", "catch-up", "backfill", "news", "trade", "training", "daily", "prices", "web"];
    return `<div class="card"><h2>Event log</h2><p>Kind <select id="log-kind">${kinds.map(k => `<option ${k === kind ? "selected" : ""} value="${k}">${k || "all"}</option>`).join("")}</select>
      Level <select id="log-level">${["", "info", "warning", "error"].map(k => `<option ${k === level ? "selected" : ""} value="${k}">${k || "all"}</option>`).join("")}</select>
      <span class="muted">Also written as JSON lines to logs/&lt;date&gt;.jsonl in the data directory.</span></p>
      <table>${rows.map(e => `<tr><td class="muted" style="white-space:nowrap">${when(e.ts)}</td><td><span class="chip">${esc(e.kind)}</span></td><td class="${e.level === "error" ? "down" : e.level === "warning" ? "muted" : ""}">${esc(e.message)}
        ${Object.keys(e.data || {}).length ? `<details><summary class="muted">data</summary><pre>${esc(JSON.stringify(e.data, null, 1))}</pre></details>` : ""}</td></tr>`).join("")}</table></div>`;
  },

  async "Software"() {
    const [sw, s] = await Promise.all([get("/api/software"), get("/api/status")]);
    return `<div class="card"><h2>Loops</h2><table><tr><th>Loop</th><th>State</th><th>Runs</th><th>Last ok</th><th>Last result</th><th>Next</th><th>Last error</th></tr>
      ${s.loops.map(l => `<tr><td>${esc(l.name)}</td><td>${esc(l.state)}</td><td>${l.runs}</td><td>${when(l.last_ok)}</td><td>${esc(l.last_result || "")}</td><td>${when(l.next_run)}</td><td class="down">${esc(l.last_error || "")}</td></tr>`).join("")}</table>
      <p class="muted">Data directory: ${esc(s.data_dir)}</p></div>
      <div class="two"><div class="card"><h2>Results by software version</h2><table><tr><th>Version</th><th>Trades</th><th>Realised P/L</th><th>From</th><th>To</th></tr>
      ${sw.trades_by_version.map(v => `<tr><td>${esc(v.software)}</td><td>${v.trades}</td><td class="${cls(v.pnl)}">${money(v.pnl)}</td><td>${when(v.first)}</td><td>${when(v.last)}</td></tr>`).join("") || '<tr><td colspan="5" class="muted">No trades yet.</td></tr>'}</table></div>
      <div class="card"><h2>Downtime</h2><p class="muted">Missed rounds are not replayed: the server catches up on prices, news, training and daily jobs, then acts.</p>
      <table><tr><th>Down from</th><th>To</th><th>Hours</th><th>Sessions missed</th><th>Stop</th></tr>${(s.downtimes || []).slice().reverse().map(d => `<tr><td>${when(d.from)}</td><td>${when(d.to)}</td><td>${d.hours}</td><td>${d.sessions_missed}</td><td class="${d.clean_stop === false ? "down" : ""}">${d.clean_stop === false ? "not clean" : d.clean_stop ? "clean" : "unknown"}</td></tr>`).join("") || '<tr><td colspan="5" class="muted">No restarts yet.</td></tr>'}</table></div>
      ${(s.fresh_starts || []).length ? `<div class="card"><h2>Fresh starts</h2><p class="muted">Each archived the dry run before it (--start-fresh); prices, news and model history were kept.</p><table><tr><th>At</th><th>Previous run from</th><th>Trades / decisions archived</th><th>Archive</th></tr>${s.fresh_starts.slice().reverse().map(f => `<tr><td>${when(f.at)}</td><td>${when(f.previous_started)}</td><td>${f.cleared.trades} / ${f.cleared.decisions}</td><td class="muted">${esc(f.archive)}</td></tr>`).join("")}</table></div>` : ""}
      <div class="card"><h2>Server starts</h2><table><tr><th>Started</th><th>Version</th><th>Symbols</th></tr>${sw.boots.map(b => `<tr><td>${when(b.ts)}</td><td>${esc(b.software)}</td><td class="muted">${esc((b.config.symbols || []).join(" "))}</td></tr>`).join("")}</table></div></div>`;
  },

  async "Lab"() {
    const lab = await get("/api/lab");
    const st = lab.lab || {};
    const sched = s => s.kind === "daily_after_close" ? `weekdays, close + ${s.minutes} min` : s.kind === "every" ? `every ${s.seconds} s` : s.kind === "once" ? `once ${s.at}` : "off";
    return `<div class="card"><h2>Lab</h2><p>The lab is a separate service that tests alternatives and improvements beside the live system. It never trades and never writes to this database. <span class="chip">${esc(st.state || "unknown")}</span>
      <span class="muted">${lab.trials || 0} trials counted in the ledger. Controls are in the command line: <code>little-trading lab --help</code>.</span></p>
      ${lab.error ? `<p class="down">${esc(lab.error)}</p>` : ""}</div>
      <div class="card"><h2>Services</h2><table><tr><th>Service</th><th>Kind</th><th>Switched on</th><th>Schedule</th><th>Environment</th><th>Last run</th></tr>
      ${(lab.services || []).map(v => `<tr><td><b>${esc(v.id)}</b><br><span class="muted">${esc(v.title)}</span></td><td>${esc(v.kind)}</td><td>${v.enabled ? "yes" : "no"}</td><td>${esc(sched(v.schedule))}</td><td>${esc(v.env)}</td>
      <td>${v.last_run ? `<span class="chip">${esc(v.last_run.state)}</span> ${when(v.last_run.started_at)}<br><span class="muted">${esc(v.last_run.message || "")}</span>` : '<span class="muted">never</span>'}</td></tr>`).join("") || '<tr><td colspan="6" class="muted">The lab has not been created (little-trading lab create).</td></tr>'}</table></div>
      <div class="two"><div class="card"><h2>Trial ledger</h2><p class="muted">Every variant tried is counted, so a lucky result is discounted.</p><table><tr><th>When</th><th>Service</th><th>Variants</th><th>State</th></tr>
      ${(lab.ledger || []).map(l => `<tr><td>${when(l.ts)}</td><td>${esc(l.service_id)}</td><td>${l.variants}</td><td>${esc(l.status)}</td></tr>`).join("") || '<tr><td colspan="4" class="muted">No trials yet.</td></tr>'}</table></div>
      <div class="card"><h2>Gaps</h2><p class="muted">Slots the lab missed. They are never run late: forward scoring must not use data the model could not have had.</p><table><tr><th>Service</th><th>Slot</th><th>Why</th></tr>
      ${(lab.gaps || []).map(g => `<tr><td>${esc(g.service_id)}</td><td>${when(g.slot)}</td><td class="muted">${esc(g.reason)}</td></tr>`).join("") || '<tr><td colspan="3" class="muted">None.</td></tr>'}</table></div></div>
      <div class="card"><h2>Lab events</h2><table>${(lab.events || []).map(e => `<tr><td class="muted" style="white-space:nowrap">${when(e.ts)}</td><td><span class="chip">${esc(e.kind)}</span></td><td>${esc(e.service_id || "")}</td><td>${esc(e.message)}</td></tr>`).join("")}</table></div>`;
  },

  async "Report for Claude"() {
    const text = await get("/api/report?days=7");
    return `<div class="card"><h2>Report for Claude</h2><p><button class="btn" id="copy">Copy</button> <a class="btn" href="/api/report?days=7" download="little-trading-report.md">Download .md</a>
      <span class="muted">Paste it into Claude Code in the little-trading repository and ask what to improve.</span></p><pre id="report">${esc(text)}</pre></div>`;
  },
};

async function symbolDetail(sym) {
  const [d, decs] = await Promise.all([get("/api/symbol?symbol=" + encodeURIComponent(sym)), get("/api/decisions?limit=1&symbol=" + encodeURIComponent(sym))]);
  const last = decs[0];
  const tradeMarks = d.trades.map(t => [Date.parse(t.ts), t.price, `var(${t.side === "BUY" ? "--up" : "--down"})`, `${t.side} ${t.qty} @ ${t.price.toFixed(2)}: ${t.reason}`]);
  const cssVar = v => getComputedStyle(document.documentElement).getPropertyValue(v.slice(4, -1)).trim();
  tradeMarks.forEach(m => m[2] = cssVar(m[2]));
  let html = `<div class="card"><h2>${esc(sym)} ${d.name ? "· " + esc(d.name) : ""}</h2>${line([{name: "close", points: d.bars.map(b => [Date.parse(b.day), b.close]), marks: tradeMarks}], {fmt: money})}
    <h2>Decision score over time</h2>${line([{name: "score", points: d.decisions.map(x => [Date.parse(x.ts), x.score])}], {zero: true, h: 160, fmt: y => y.toFixed(2)})}</div>`;
  if (last) {
    const i = last.inputs;
    html += `<div class="two"><div class="card"><h2>Inputs of the latest decision</h2><table>${Object.entries(i.price_features).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="${cls(v)}">${v}</td></tr>`).join("")}</table></div>
      <div class="card"><h2>Why the models voted as they did</h2>${Object.entries(i.explain || {}).map(([k, parts]) => `<p><b>${esc(k)}</b> p=${last.components.probabilities[k]}: ${parts.map(p => `<span class="chip">${esc(p[0])} ${p[1]}</span>`).join("")}</p>`).join("")}
      <h2>Headlines used (${i.news_count})</h2><table>${(i.headlines || []).map(h => `<tr><td class="muted">${when(h.published)}</td><td>${esc(h.title)}</td><td>${Object.entries(h).filter(([k]) => k.startsWith("p_")).map(([k, v]) => `<span class="chip">${esc(k.slice(2))} ${v}</span>`).join("")}</td></tr>`).join("") || '<tr><td class="muted">none in the last 72 hours</td></tr>'}</table></div></div>`;
  }
  return html;
}

async function render() {
  document.getElementById("tabs").innerHTML = TABS.map(t => `<button class="${t === tab ? "on" : ""}" data-tab="${t}">${t}</button>`).join("");
  const main = document.getElementById("main");
  try { main.innerHTML = await views[tab](); } catch (err) { main.innerHTML = `<div class="card down">Could not load: ${esc(err.message)}</div>`; }
}

document.addEventListener("click", async e => {
  const t = e.target.closest("[data-tab]");
  if (t) { tab = t.dataset.tab; localStorageSet("tab", tab); symbol = null; return render(); }
  const r = e.target.closest("tr[data-sym]");
  if (r) { symbol = r.dataset.sym; await render(); document.querySelectorAll(".card")[1]?.scrollIntoView({behavior: "smooth"}); }
  if (e.target.id === "copy") { try { await navigator.clipboard.writeText(document.getElementById("report").textContent); e.target.textContent = "Copied"; } catch (err) { e.target.textContent = "Copy failed"; } }
});
document.addEventListener("change", e => {
  if (e.target.id === "news-sym") { localStorageSet("newsSym", e.target.value); render(); }
  if (e.target.id === "log-kind") { localStorageSet("logKind", e.target.value); render(); }
  if (e.target.id === "log-level") { localStorageSet("logLevel", e.target.value); render(); }
});
header(); render();
setInterval(() => { header(); if (!["Report for Claude", "Logs"].includes(tab) && !document.querySelector("details[open]")) render(); }, 30000);
</script>
</body>
</html>
"""

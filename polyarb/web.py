from __future__ import annotations

import asyncio

import structlog
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from .state import AppState
from .weather_state import WeatherAppState

log = structlog.get_logger("polyarb.web")


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>polyarb dashboard</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
         background: #0d1117; color: #c9d1d9; margin: 0; padding: 24px; }
  h1 { margin: 0 0 4px 0; font-size: 18px; }
  .sub { color: #8b949e; font-size: 12px; margin-bottom: 18px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
          padding: 14px 16px; }
  .card h2 { font-size: 13px; margin: 0 0 10px 0; color: #58a6ff; font-weight: 600;
             text-transform: uppercase; letter-spacing: 0.5px; }
  .row { display: flex; justify-content: space-between; gap: 10px; padding: 4px 0;
         border-bottom: 1px dashed #21262d; font-size: 13px; }
  .row:last-child { border-bottom: 0; }
  .k { color: #8b949e; }
  .v { font-variant-numeric: tabular-nums; }
  .v.warn { color: #f0883e; }
  .v.good { color: #3fb950; }
  .v.bad  { color: #f85149; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 6px; }
  th, td { padding: 5px 8px; border-bottom: 1px solid #21262d; text-align: right;
           font-variant-numeric: tabular-nums; }
  th:first-child, td:first-child { text-align: left; }
  th { color: #8b949e; font-weight: 500; }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 999px;
          background: #21262d; color: #c9d1d9; font-size: 11px; margin-left: 6px; }
  .pill.demo { background: #30363d; color: #f0883e; }
  .pill.live { background: #1f6feb33; color: #58a6ff; }
  .empty { color: #6e7681; font-style: italic; padding: 12px 0; }
</style>
</head>
<body>
  <h1>polyarb<span id="mode" class="pill">…</span></h1>
  <div class="sub" id="sub">connecting…</div>

  <div class="grid">
    <div class="card">
      <h2>Market</h2>
      <div id="market"></div>
    </div>
    <div class="card">
      <h2>Top of book</h2>
      <div id="top"></div>
    </div>
  </div>

  <div class="card" style="margin-top: 16px;">
    <h2>Totals (paper-fills)</h2>
    <div id="totals"></div>
  </div>

  <div class="card" style="margin-top: 16px;">
    <h2>Recent opportunities</h2>
    <div id="opps"></div>
  </div>

<script>
function fmt(x, n=4) {
  if (x === null || x === undefined) return "—";
  return Number(x).toFixed(n);
}
function row(k, v, cls="") {
  return `<div class="row"><span class="k">${k}</span><span class="v ${cls}">${v}</span></div>`;
}
async function tick() {
  let r;
  try { r = await fetch("/state.json", {cache: "no-store"}); }
  catch { document.getElementById("sub").textContent = "disconnected"; return; }
  const s = await r.json();

  const modeEl = document.getElementById("mode");
  modeEl.textContent = s.mode;
  modeEl.className = "pill " + s.mode;

  document.getElementById("sub").textContent =
    `tick #${s.tick} • polling every 1s`;

  const m = s.market;
  document.getElementById("market").innerHTML = m
    ? row("slug", m.slug)
      + row("asset", m.asset)
      + row("yes_token", m.yes_token.slice(0, 16) + "…")
      + row("no_token", m.no_token.slice(0, 16) + "…")
      + row("settles in", m.settle_in_s.toFixed(1) + " s")
      + row("min order size", m.minimum_order_size)
    : `<div class="empty">no active market yet</div>`;

  const top = `
    ${row("YES best ask", s.yes.best_ask
          ? `${fmt(s.yes.best_ask.price,3)} × ${fmt(s.yes.best_ask.size,1)}` : "—")}
    ${row("NO  best ask", s.no.best_ask
          ? `${fmt(s.no.best_ask.price,3)} × ${fmt(s.no.best_ask.size,1)}` : "—")}
    ${row("ask sum", fmt(s.ask_sum, 4),
          s.ask_sum !== null && s.ask_sum < 1.0 ? "good" : "")}
    ${row("YES best bid", s.yes.best_bid
          ? `${fmt(s.yes.best_bid.price,3)} × ${fmt(s.yes.best_bid.size,1)}` : "—")}
    ${row("NO  best bid", s.no.best_bid
          ? `${fmt(s.no.best_bid.price,3)} × ${fmt(s.no.best_bid.size,1)}` : "—")}
    ${row("bid sum", fmt(s.bid_sum, 4),
          s.bid_sum !== null && s.bid_sum > 1.0 ? "warn" : "")}
  `;
  document.getElementById("top").innerHTML = top;

  document.getElementById("totals").innerHTML =
    row("opportunities", s.totals.opps)
    + row("avg edge (bps)", fmt(s.totals.avg_edge_bps, 1),
          s.totals.avg_edge_bps > 0 ? "good" : "")
    + row("paper profit (USD)", fmt(s.totals.profit_usd, 4),
          s.totals.profit_usd > 0 ? "good" : "");

  const opps = s.recent;
  if (!opps.length) {
    document.getElementById("opps").innerHTML =
      `<div class="empty">no opportunities yet — wait for an injection</div>`;
  } else {
    let h = `<table><thead><tr>
        <th>time</th><th>YES@</th><th>NO@</th><th>size</th>
        <th>fees</th><th>edge bps</th><th>profit $</th><th>settle s</th>
      </tr></thead><tbody>`;
    for (const o of opps) {
      const t = new Date(o.ts*1000).toLocaleTimeString();
      h += `<tr>
        <td>${t}</td>
        <td>${fmt(o.yes_price,3)}</td>
        <td>${fmt(o.no_price,3)}</td>
        <td>${fmt(o.size,1)}</td>
        <td>${fmt(o.fees_usd,3)}</td>
        <td class="v good">${fmt(o.edge_bps,1)}</td>
        <td class="v good">${fmt(o.profit_usd,3)}</td>
        <td>${fmt(o.settle_in_s,0)}</td>
      </tr>`;
    }
    h += `</tbody></table>`;
    document.getElementById("opps").innerHTML = h;
  }
}
tick();
setInterval(tick, 1000);
</script>
</body>
</html>
"""


def make_app(state: AppState, weather_state: WeatherAppState | None = None) -> FastAPI:
    app = FastAPI(title="polyarb")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(INDEX_HTML)

    @app.get("/state.json")
    def state_json():
        return JSONResponse(state.snapshot())

    @app.get("/weather/state.json")
    def weather_state_json():
        if weather_state is None:
            return JSONResponse({"enabled": False, "events": []})
        return JSONResponse({"enabled": True, **weather_state.snapshot()})

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "mode": state.mode, "tick": state.tick}

    return app


async def serve(
    state: AppState,
    host: str,
    port: int,
    stop: asyncio.Event,
    *,
    weather_state: WeatherAppState | None = None,
) -> None:
    app = make_app(state, weather_state=weather_state)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve())
    log.info("web_listening", url=f"http://{host}:{port}")
    try:
        await stop.wait()
    finally:
        server.should_exit = True
        await serve_task

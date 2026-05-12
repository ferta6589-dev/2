# polyarb

Paper-trading arbitrage bot for Polymarket 5-minute crypto markets
(`BTC Up or Down 5m` and friends). Detects the classic
`best_ask(YES) + best_ask(NO) < $1` mispricing on the Polymarket CLOB and logs
would-be fills as JSONL — **no on-chain orders are sent in v1**.

## Why this works

Polymarket's 5-minute crypto markets are binary: at settlement either YES or
NO pays $1 and the other pays $0. So `price(YES) + price(NO) = $1` is a hard
arbitrage equality. Whenever the CLOB asks momentarily fall below $1 in sum
(after fees), buying both legs in equal size locks in a risk-free profit
regardless of which way BTC moves.

## Install

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
cp .env.example .env
```

## Run

```sh
# One-shot diagnostic: prints active 5m markets and top of book.
python -m polyarb.main --once

# Paper-trading loop (Ctrl-C to stop):
python -m polyarb.main

# Aggregate logged opportunities:
python -m polyarb.pnl
```

## Configuration (`.env`)

| Var | Meaning |
| --- | --- |
| `ASSETS` | CSV of crypto symbols, e.g. `BTC,ETH` |
| `PROFIT_BPS_MIN` | Minimum edge (in bps) before logging an opportunity |
| `MIN_TIME_TO_SETTLE_S` | Skip opportunities settling sooner than this |
| `FEE_RATE_BPS` | Polymarket nominal taker fee (default 200 = 2.00%); realised fee follows the `min(p, 1-p)` curve |
| `GAMMA_HOST`, `CLOB_HTTP_HOST`, `CLOB_WS_HOST` | Polymarket endpoints |
| `LOG_DIR` | Where `paper_trades_*.jsonl` is written |

## How market discovery works

5m market slugs are deterministic: `window_ts = now - (now % 300)`, then
`slug = f"{asset}-updown-5m-{window_ts}"`. The bot polls Gamma every
`DISCOVERY_INTERVAL_S` seconds, swaps its WebSocket subscription on each window
roll, and resets the local orderbook.

## Tests

```sh
pytest -q
```

## Weather strategy — METAR-driven, latency-edge play

`polyarb` ships a second strategy targeting Polymarket's daily city-temperature
events (e.g. "Highest temperature in Moscow on May 12 2026" with `<11°C`,
`11-13°C`, … `>23°C` buckets). It is gated behind `WEATHER_ENABLED=false` by
default.

The core bet is **speed**: Polymarket weather markets resolve from a single
NOAA-tracked station (e.g. Vnukovo `UUWW` for Moscow, Central Park `KNYC` for
NYC). The fastest free path to that station's observation is NOAA tgftp:

    https://tgftp.nws.noaa.gov/data/observations/metar/stations/{ICAO}.TXT

That tiny plain-text file is overwritten in place each time a new METAR hits
NOAA's GTS ingest — typically 30-90 seconds after the airport issues it. We
poll it every 20 seconds inside the publish window (HH:25-40 / HH:55-10 UTC)
and every ~3 minutes outside, with `If-Modified-Since` to keep requests cheap.

End-to-end:

1. `weather_markets.fetch_weather_events` discovers daily-temperature events
   on Polymarket and maps the city to its resolver ICAO via `CITY_RESOLVERS`.
2. `metar.fetch_metar` pulls the latest report from NOAA tgftp (primary), the
   per-hour cycles file (fallback), and Ogimet (last resort).
3. `metar.DailyMaxTracker` accumulates the day's observations and exposes the
   running max in whole °C (matches resolver precision).
4. `weather_strategy.classify_buckets` labels each bucket from the running
   max: `leader` / `exceeded` / `unreached`.
5. `weather_strategy.decide_entry` buys YES on the **current leader** at best
   ask (capped by `WEATHER_MAX_PRICE`, sized by
   `WEATHER_PER_EVENT_BUDGET_USD`). This is the "buy before the order book
   reprices" play — we act on the fresh METAR before slower traders.
6. `weather_strategy.decide_exits` sells YES on freshly **exceeded** buckets
   at best bid to recover capital before they decay to zero.
7. After `event.end_ts`, the bucket containing the rounded max pays $1.

Run:

```sh
# Synthetic METAR feed (Moscow / UUWW timeline, no network)
python -m polyarb.main --strategy weather --demo

# Live discovery against Polymarket + real METAR polling (paper-only)
WEATHER_ENABLED=true python -m polyarb.main --strategy weather

# Both strategies at once with the dashboard
WEATHER_ENABLED=true python -m polyarb.main --strategy both --web

# Aggregate weather paper P&L
python -m polyarb.pnl --prefix weather_trades
```

`--mode live` is a stub that raises `NotImplementedError`. Live execution will
require `py-clob-client`, USDC/CTF allowance, and signing — deferred to v2.

## Out of scope (v1)

- **Live execution** — only `PaperExecutor` / `WeatherPaperExecutor` are
  implemented. Wiring `py-clob-client` requires a Polygon private key,
  USDC + CTF allowance, and a careful retry/cancel policy.
- **Historical backtest** — Polymarket does not freely publish historical
  orderbook snapshots.
- **Sub-30-second METAR feeds** — Synoptic Data's Push Streaming or a direct
  GTS subscription would beat NOAA tgftp by tens of seconds. Both are paid.
- **Pre-position on long-range forecast** — the bot now reacts only to live
  observations. Adding a small pre-event entry on NWS / Open-Meteo forecasts
  is a follow-up.

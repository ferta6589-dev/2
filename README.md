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

## Weather strategy (parallel module)

`polyarb` ships a second strategy targeting Polymarket's daily city-temperature
events (e.g. "Highest temperature in NYC on May 7" with buckets `<60°F`,
`60-65°F`, …, `>75°F`). It is gated behind `WEATHER_ENABLED=false` by default.

End-to-end:

1. `weather_markets.fetch_weather_events` discovers daily-temperature events on
   Polymarket (all cities by default).
2. `forecast.fetch_distribution` blends NWS + Open-Meteo into a Gaussian
   forecast for the target date.
3. `weather_strategy.select_window` chooses the contiguous 3-bucket window
   with the highest combined probability mass.
4. `weather_strategy.decide_entry` buys YES on each bucket at best ask, capped
   by `WEATHER_CENTRAL_MAX_PRICE` / `WEATHER_WING_MAX_PRICE` and a per-event
   budget (`WEATHER_PER_EVENT_BUDGET_USD`).
5. On the day-of, `weather_main.monitoring_loop` polls the resolver station's
   latest observation every `NWS_POLL_INTERVAL_S` (default 30 min) and
   classifies each bucket as winner / competitor / loser
   (`weather_strategy.classify_buckets`).
6. Losers are unwound at best bid; winners are held to resolution.
7. After `event.end_ts`, a `kind:"resolve"` row credits $1 × qty for the
   winning bucket.

Run:

```sh
# Synthetic feed (no network)
python -m polyarb.main --strategy weather --demo

# Live discovery against Polymarket (paper-only)
WEATHER_ENABLED=true python -m polyarb.main --strategy weather

# Both strategies at once with the dashboard
WEATHER_ENABLED=true python -m polyarb.main --strategy both --web

# Aggregate weather paper P&L
python -m polyarb.pnl --prefix weather_trades
```

`--mode live` is a stub that raises `NotImplementedError`. Live execution will
require `py-clob-client`, USDC/CTF allowance, and signing — deferred to v2.

## Out of scope (v1)

- **Live execution** — the `Executor` interface is in place, but only
  `PaperExecutor` / `WeatherPaperExecutor` are implemented. Wiring
  `py-clob-client` for real orders requires a Polygon private key,
  USDC + CTF allowance setup, and a careful retry/cancel policy.
- **Historical backtest** — Polymarket does not freely publish historical
  orderbook snapshots.
- **Multi-level VWAP sizing** — the crypto arb detector only sizes against the
  matched-min of top-of-book asks.
- **Calibrated forecast σ** — the weather strategy uses a hand-coded σ table
  (4/3/2 °F at 5/3/0 days). Historical NWS-vs-realised calibration is a v2
  follow-up.

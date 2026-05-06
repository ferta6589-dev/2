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

## Out of scope (v1)

- **Live execution** — the `Executor` interface is in place, but only
  `PaperExecutor` is implemented. Wiring `py-clob-client` for real orders
  requires a Polygon private key, USDC + CTF allowance setup, and a careful
  retry/cancel policy.
- **Historical backtest** — Polymarket does not freely publish historical
  orderbook snapshots.
- **Multi-level VWAP sizing** — current detector only sizes against the
  matched-min of top-of-book asks.

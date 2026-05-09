# BTC scalping backtest — honest results

Backtest of 9 mechanical scalping strategies + buy-and-hold on Bitfinex BTC/USD
1-minute data (2017–2019, ~315k 5m bars), with realistic Bybit perp costs:

- Taker fee: 0.055% per side (round-trip 0.11%)
- Slippage: 2 bps per side (round-trip 0.04%)
- Total round-trip cost: **0.15%** of notional per trade
- Liquidation modeled when price moves more than `1/leverage − maint_margin` against you
- All signals act on **next bar's open** (no look-ahead)

## TL;DR — none of the standard scalping textbook setups produced money

| Test                              | Best strategy        | Result                       |
|-----------------------------------|----------------------|------------------------------|
| 5m, 2017-2018 in-sample           | BuyAndHold           | +311%, MaxDD -84%            |
| 5m, 2019 out-of-sample            | BuyAndHold           | +88%, MaxDD -53%             |
| 15m, 2017-2019 full               | BuyAndHold           | +650%                        |
| **5m, ZERO fees** (signal sanity) | MACD signal cross    | +261,400% theoretical        |
| **5m, REALISTIC fees**            | All scalpers ruined  | -100% (most were liquidated) |

**Every active strategy lost money under realistic costs. Most lost 100%.**
At 5m bars these strategies fire 3,000–24,000 trades over 3 years; round-trip
costs eat 4.4–36× the starting equity in fees alone. Even at zero fees,
breakout-style strategies (Donchian, Keltner) still lose because BTC chop
shreds them between false signals.

## Leverage stress (top-3 IS picks, OOS 2019)

The "best" backtest picks (BuyAndHold / VWAP-fade / RSI-MR) on OOS:

| Leverage | BuyAndHold     | VWAP fade        | RSI mean-revert  |
|----------|----------------|------------------|------------------|
| 1×       | +88%, dd -53%  | -99%, dd -99%    | -82%, dd -82%    |
| 3×       | +265%, dd -65% | -100%, dd -100%  | -90%, dd -91%    |
| 5×       | +442%, dd -68% | -100%, dd -100%  | -96%, dd -98%    |
| 10×      | -44%, dd -98%  | -100% + liq      | -100%            |
| 20×      | -98%, dd -100% | -100% + 113 liqs | -100% + 25 liqs  |

Note BuyAndHold at 10×/20× was **liquidated** in 2018's bear market despite
being profitable at 5×. Leverage doesn't multiply edge — it multiplies the
left tail.

## Files

| File                            | Purpose                                   |
|---------------------------------|-------------------------------------------|
| `backtest.py`                   | Vectorized engine: signals→trades→equity |
| `strategies.py`                 | 9 scalping strategies + buy-and-hold      |
| `run_compare.py`                | IS/OOS leaderboard + leverage stress      |
| `run_zero_fee.py`               | "Does the signal have any edge?" check    |
| `results_*.csv`                 | Output tables                             |

Data source: <https://github.com/Zombie-3000/Bitfinex-historical-data>
(BTC/USD 1m, 2013–2019, 2.6M bars). Not committed — regenerate locally.

## Honest takeaways

1. **Fees are the boss-fight of scalping.** A 0.15% round-trip cost requires
   each trade to make 15+ basis points just to break even. Most signals on
   crypto 5m bars do not.
2. **The only "strategy" that survived was buy-and-hold**, and that's not a
   scalping strategy. It worked because BTC happened to rally in 2019.
3. **Leverage on a losing strategy accelerates ruin**, and on a winning
   strategy still risks liquidation in normal drawdowns.
4. **Past performance** tested here is from 2017-2019. Microstructure has
   changed (more HFT, tighter spreads, more crowding). Today's fees and slip
   may be slightly better, but the cost-vs-edge inequality is the same.
5. To find a real edge you'd need: a) a non-trivial feature (e.g. funding
   rates, order-book imbalance, cross-exchange basis), b) maker-only execution
   to flip from -0.055% to +0.02% per fill, c) parameter selection that holds
   up out-of-sample with walk-forward, not just one fixed split.

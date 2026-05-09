# Swing/position strategies — honest results

After scalping failed (fees ate every signal), I tested 20 swing/position
strategies on 1h, 4h, and daily BTC bars (2015–2019). Here's what survived.

## Methodology

- **Data**: Bitfinex BTC/USD 1m, resampled to 1h/4h/daily (covers 2015 chop,
  2016 bull, 2017 mania, 2018 bear, 2019 recovery — diverse regimes)
- **Costs**: Bybit perp taker 0.055% + 2bp slippage per side
- **Splits**: IS = 2017–2018, OOS = 2019. Bonus: 2015–2016 as deeper OOS
- **Tests**: IS/OOS leaderboard → year-by-year consistency → parameter grid → leverage stress
- **No look-ahead**: signals at bar close act on next bar's open

## Survivors (positive in BOTH IS and OOS)

Top 10 by OOS Sharpe across all timeframes:

| TF  | Strategy            | IS ret % | IS Sharpe | OOS ret % | OOS Sharpe | OOS DD |
|-----|---------------------|----------|-----------|-----------|------------|--------|
| 4h  | Weekend_Effect      | +121     | 0.92      | +172      | 2.35       | -21    |
| 1D  | Mom_ROC_90          | +944     | 1.87      | +201      | 2.14       | -27    |
| 4h  | EMA200_filter       | +282     | 1.35      | +131      | 1.87       | -34    |
| **4h**  | **DualMom_30_90** | **+664** | **2.19** | **+94**  | **1.72**   | **-24** |
| 4h  | Supertrend_10_3_LO  | +266     | 1.41      | +94       | 1.62       | -33    |
| 4h  | SMA_50_200_LO       | +394     | 1.54      | +90       | 1.51       | -42    |
| 1D  | DualMom_30_90       | +619     | 2.00      | +71       | 1.28       | -44    |
| 4h  | BuyAndHold          | +305     | 1.21      | +89       | 1.30       | -52    |

## Year-by-year consistency (5 years)

| Strategy            | Pos. yrs | Avg yr % | Worst yr % | Avg yearly DD |
|---------------------|---------:|---------:|-----------:|--------------:|
| **DualMom_30_90 4h** | **4/5**  | **+206**  | **-0.2**   | **-24%**       |
| Supertrend_10_3_LO 4h| 4/5      | +128      | -17        | -29%          |
| Turtle_20_10_LO 4h   | 4/5      | +86       | -37        | -33%          |
| Weekend_Effect 4h    | 4/5      | +88       | -37        | -33%          |
| EMA200_filter 4h     | 4/5      | +168      | -46        | -41%          |
| SMA_50_200_LO 4h     | 4/5      | +208      | -44        | -39%          |
| BuyAndHold daily     | 4/5      | +289      | **-72**    | **-47%**       |

The standout is **DualMomentum 30/90 on 4h**: only -0.2% in the brutal 2018
bear market when buy-and-hold lost 72%.

## Why DualMomentum is the cleanest

Logic: long when **both** 30-bar AND 90-bar rate-of-change are positive, flat
otherwise. On 4h that's roughly: long only when last 5 days AND last 15 days
are both up. Result: it sits in cash through downtrends (35% time in market on
average), capturing trends and avoiding the deep drawdowns of buy-and-hold.

### Parameter sensitivity (4h, 2015-2019)

Tested 18 (n_short, n_long) combos. Sharpe ranged 1.20–1.75, all positive:

| n_short | n_long | Sharpe | DD %  |
|---------|--------|--------|-------|
| 30      | 90     | 1.75   | -40   |
| 30      | 50     | 1.72   | -35   |
| 30      | 150    | 1.71   | -50   |
| 20      | 50     | 1.67   | -38   |
| ... (all combos remained positive Sharpe) |

No single magic combo — the strategy's edge is structural, not curve-fit.

## Leverage stress on the *winners* (2015-2019, 5 years)

| Leverage | DualMom 30/90       | EMA200 filter   | BuyAndHold     |
|----------|---------------------|-----------------|----------------|
| 1×       | +3,210% / DD -40%   | +1,769% / -74%  | +2,175% / -84% |
| 2×       | +44,329% / DD -58%  | +7,411% / -90%  | +4,351% / -86% |
| 3×       | +262,117% / DD -73% | +11,454% / -97% | **liquidated** |
| 5×       | +1.48M% / DD -91%   | +3,832% / -100% | **liquidated** |
| 10×      | **liquidated**      | **liquidated**  | **liquidated** |

Even the best swing strategy gets liquidated at 10× because crypto's normal
drawdowns easily breach 1/10 = 10% from peak entry. **Buy-and-hold is liquidated
already at 3×** in the 2018 -84% drawdown.

The "money math" lesson:
- 1× DualMom: ~100% CAGR, -40% DD — sleeping easy, blowup-proof
- 2× DualMom: ~239% CAGR, -58% DD — still no liquidations across 5 years
- 3-5× DualMom: monstrous returns but DD goes 73-91% (you need iron stomach + spare capital)
- ≥10× — inevitable ruin, just a question of time

## Verdict

1. **There IS a real edge in trend/momentum on BTC at 4h-daily timeframes.**
   The signal is robust across years, parameters, and timeframes — not curve-fit.

2. **The best strategy I found is dual-window absolute momentum (30/90 on 4h)**.
   It captures the bull markets (2015, 2016, 2017, 2019) and dodges the bear
   (2018: -0.2% vs buy-and-hold -72%). At 1× it triples buy-and-hold's
   risk-adjusted return.

3. **Leverage destroys even good strategies past 3-5×**. The BTC market has too
   many >10% drawdowns. A "rock-solid" Sharpe 1.75 strategy still hits -91%
   peak DD at 5×; one notch higher liquidates everything.

4. **There's no "huge money on huge leverage"** — the math doesn't allow it.
   What's possible: **2-3× leverage on a real edge → CAGR in 100-400% range
   with deep but survivable drawdowns**.

5. **Caveats**: this is BTC/USD 2015-2019. The 2020-2024 environment has more
   competition, tighter spreads, more flash crashes. The signal probably still
   exists but with smaller magnitude. Always re-test on recent data before
   deploying.

## How to use this practically (NOT financial advice)

If you wanted to actually trade the DualMomentum 30/90 4h strategy on Bybit:

1. Compute 30-bar and 90-bar ROC of close on 4h candles.
2. Long when both > 0; flat otherwise (or short if you want full long-short,
   but the long-only filter helps reduce trades).
3. Bybit linear perp BTCUSDT, leverage 2× max, isolated margin.
4. Position size = (account * 2) / price.
5. Place market orders at next 4h candle open after signal change.
6. **Forward-paper-trade for 3 months minimum** before risking real money.
   The walk-forward test here gives you cautious optimism, not certainty.

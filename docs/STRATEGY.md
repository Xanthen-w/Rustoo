# Strategy: risk-managed BTC/ETH trend core

## What the bot does

Every hour, for each of BTC and ETH:

1. **Trend filter:** compare the price with its 40-day EMA (960 hourly bars), using a ±3% hysteresis band.
   The asset counts as *in trend* once the price closes 3% above the EMA, and *out of trend* once it closes
   3% below. In between, the previous state holds, so the position doesn't flip on noise.
2. **Volatility-targeted size:** each asset gets a weight of `(0.5 / 2) / realized_vol_30d`, i.e. an equal
   share of a 50% annualized volatility budget, so a calmer asset gets a bigger position. Total exposure
   is capped at 100%, with no leverage.
3. **Exposure:** full size while in trend, 15% of it while out of trend (`min_exposure`). The rest is cash.
4. **Execution:** trade only when an asset's weight has drifted at least 5% from target, except at
   **00:00 UTC every day**, when the portfolio is rebalanced exactly to target.

Parameters are in `config/strategy.yaml`; the code is `src/strategy/signals.py::trend_vol_target`.

## Why this design

The competition ranks first by raw 14-day return, then by `0.4·Sortino + 0.3·Sharpe + 0.3·Calmar`, and
requires trades on at least 8 days (`docs/COMPETITION_RULES.md`). So the design aims at:

- **A thin bad tail.** Calmar and Sortino punish drawdowns and downside volatility. Exiting assets in
  downtrends and sizing by volatility both cut the worst outcomes.
- **Low cost.** At 0.1% per taker trade, every hourly strategy we tested lost more to fees than its
  signal earned (see below). A 40-day trend changes state rarely, and the 5% band suppresses small trades.
- **Daily activity from the strategy's own logic.** The 15% floor means there is always a position,
  and the daily exact rebalance of that position trades every day.

## Evidence

Data: Binance spot 5-minute klines resampled to 1h (`src/data/binance.py`), 2024-09-26 → 2026-09-26.
Costs: 0.10% fee + 5 bps slippage per trade. Chronological splits (`config/research.yaml`):
train → 2026-01-01, validation → 2026-06-01, holdout test after, **not yet used**.

1. **Hourly alpha strategies fail after costs.** A walk-forward search (`scripts/walk_forward.py`)
   covered 7 families and 142 parameter sets: cross-sectional momentum, EMA trend following, mean reversion,
   vol-filtered momentum, and single-asset momentum. On train, none had a positive stitched out-of-sample
   return. In-sample scores didn't carry forward; trend following picked 8 different parameter sets in 11 folds.
2. **A drawdown state machine hurt.** Cutting exposure after 5/10/20% drawdowns (`src/risk/drawdown.py`)
   sold after drops and bought back after rebounds. It turned buy-and-hold's −16% into −21% out-of-sample,
   so it's implemented but switched off.
3. **Fourteen-day window analysis** (`scripts/window_analysis.py`) starts from cash on every day of a split:

   | split | strategy | mean 14d return | 10th percentile | worst | windows with ≥ 8 trading days |
   |---|---|---|---|---|---|
   | train (338 windows) | BTC buy-and-hold | −0.5% | −9.2% | −18.0% | 0% |
   | train | **this strategy** | **+0.8%** | **−4.3%** | **−12.4%** | **100%** |
   | validation (137) | BTC buy-and-hold | −1.6% | −14.0% | −28.7% | 0% |
   | validation | **this strategy** | **−1.3%** | **−7.1%** | **−11.0%** | **100%** |

   It was selected on train from a 48-point sweep and a 24-point follow-up, then run once on validation.

## Honest limitations

- **The return edge did not survive validation; the risk reduction did.** In a falling market the
  strategy still loses money, just about half as much in bad windows. It can't manufacture return
  when BTC and ETH fall.
- Both evaluation periods (2025, early 2026) were weak for crypto. The windows overlap, so train has only
  about 24 independent 14-day periods.
- Backtests use Binance prices. Roostoo's mock-exchange prices have not yet been compared with
  recorded Roostoo tickers.
- Shorting ("1x long and short" in the rules) could turn downtrends into gains. It's not used until the
  bot confirms that Roostoo accepts shorts in this competition.

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
train → 2026-01-01, validation → 2026-06-01, holdout test after (**used once**, see below).

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

## Holdout evaluation (run once, 2026-09-27)

The holdout split (2026-06-01 → 2026-09-26, 118 days) was evaluated exactly once, after the
strategy and all its parameters were fixed, with nothing changed afterwards. It was a rising
market: BTC went from $73.7k to $84.1k (+14%) and ETH from $2,007 to $2,692 (+34%).
Base costs unless noted.

**Continuous run over the whole holdout**

| | return | max drawdown | Sharpe | Sortino | Calmar | composite | avg exposure | trading days |
|---|---|---|---|---|---|---|---|---|
| **this strategy** | **+32.4%** | **−6.6%** | **3.00** | **4.80** | 21.2 | **9.18** | 66% | 118 / 118 |
| this strategy, pessimistic costs | +32.0% | −6.6% | 2.97 | 4.75 | 20.8 | 9.04 | 66% | 118 / 118 |
| 50/50 BTC+ETH buy-and-hold | +23.5% | −22.7% | 1.66 | 2.45 | 4.1 | 2.71 | 100% | 1 |
| BTC buy-and-hold | +13.7% | −21.2% | 1.18 | 1.75 | 2.3 | 1.75 | 100% | 1 |

**Every 14-day window from cash (104 windows)**

| | mean return | median | P(return > 0) | 10th pct | worst | worst drawdown | windows with ≥ 8 trading days |
|---|---|---|---|---|---|---|---|
| **this strategy** | **+3.9%** | +0.6% | 62.5% | **−1.9%** | **−4.3%** | **−6.6%** | **100%** |
| 50/50 BTC+ETH buy-and-hold | +4.6% | +2.0% | 67.3% | −3.8% | −13.1% | −21.7% | 0% |
| BTC buy-and-hold | +3.2% | +1.0% | 57.7% | −4.5% | −11.6% | −19.7% | 0% |

Reading it honestly:
- It behaved as designed. It kept most of the upside of a rising market (a third of its
  gain over the reference comes from holding ETH, which outperformed), at about a third of
  buy-and-hold's drawdown, and it traded every day.
- Over the continuous 4 months it beat 50/50 BTC+ETH. That comes from sidestepping a
  drawdown, not from higher exposure: it averaged 66% invested.
- In a *typical* 14-day window, simply holding 50/50 BTC+ETH returned a bit more (median +2.0%
  vs +0.6%). The strategy's advantage is the much thinner bad tail (worst −4.3% vs −13.1%),
  which the composite score rewards.
- Composite scores over a 4-month run are inflated by annualization (Calmar 21). Compare
  them across rows, not as absolute numbers.
- **No untouched data remains.** Any further change to the strategy is judged only on
  train and validation, or on live results.

## Which coins? (universe comparison, 2026-09-27)

`scripts/universe_comparison.py` ran the live settings on six coin sets fixed up front, using
14-day windows from cash plus one continuous run, with base costs:

| coins | train: mean 14d | train: worst 14d | train: continuous | train: max DD | windows ≥ 8 trading days | validation: mean 14d | validation: continuous |
|---|---|---|---|---|---|---|---|
| BTC | +0.42% | −12.8% | +31.5% | −22.5% | 68% | −0.69% | −8.9% |
| **BTC+ETH (live)** | **+0.93%** | −12.5% | **+48.6%** | −23.5% | 100% | −1.18% | −13.0% |
| BTC+ETH+TRX | +0.97% | −10.1% | +45.0% | −29.3% | 100% | +0.39% | +2.1% |
| BTC+TRX | +0.64% | −9.5% | +30.8% | −32.2% | 100% | +0.75% | +5.3% |
| Top 5 by liquidity | +0.89% | −11.0% | +46.7% | −21.5% | 100% | −1.51% | −14.0% |
| Top 10 by liquidity | +0.81% | −9.5% | +44.0% | −20.9% | 100% | −0.78% | −5.1% |

- On **train** (the selection data), BTC+ETH, BTC+ETH+TRX and the top-5 set are statistically
  indistinguishable. Under the pre-stated rule (choose on train), the live BTC+ETH set stays.
- BTC alone fails the ≥ 8 trading-days rule in about a third of windows: with one asset, nothing
  drifts between holdings for the daily rebalance to fix.
- On **validation**, the TRX sets did clearly better. But TRX was singled out after looking at a
  table that included validation-period returns, so validation can't confirm it independently.
  Adding TRX is a judgement call (it's a low-correlation diversifier), not a result.

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

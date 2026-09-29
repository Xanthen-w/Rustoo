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
4. **Execution:** trade only when an asset's weight has drifted at least 5% from target, except
   **every 6 hours (00:00, 06:00, 12:00, 18:00 UTC)**, when the portfolio is rebalanced exactly to target.

Parameters are in `config/strategy.yaml`; the code is `src/strategy/signals.py::trend_vol_target`.

## Why this design

The competition ranks first by raw 14-day return, then by `0.4·Sortino + 0.3·Sharpe + 0.3·Calmar`, and
requires trades on at least 8 days (`docs/COMPETITION_RULES.md`). So the design aims at:

- **A thin bad tail.** Calmar and Sortino punish drawdowns and downside volatility. Exiting assets in
  downtrends and sizing by volatility both cut the worst outcomes.
- **Low cost.** At 0.1% per taker trade, every hourly strategy we tested lost more to fees than its
  signal earned (see below). A 40-day trend changes state rarely, and the 5% band suppresses small trades.
- **Daily activity from the strategy's own logic.** The 15% floor means there is always a position,
  and the exact rebalances every 6 hours trade it back to target (about 8 fills a day).

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

## Rebalance schedule (2026-09-28)

The competition requires trades on at least 8 days with "enough trades each day", without a
number. `scripts/rebalance_frequency.py` compared exact rebalances 1×, 2×, 4×, 6× and 24× a day
(everything else at the live settings):

| schedule | fills per day | median fill | train return | validation return | mean 14-day return (train / validation) |
|---|---|---|---|---|---|
| 1× (00:00) | 2.1 | $197–319 | +48.6% | −13.0% | +0.93% / −1.18% |
| **4× (00/06/12/18)** | **8.1** | $64–120 | +47.4% | −13.3% | +0.90% / −1.21% |
| 24× (hourly) | 48 | $19–39 | +45.6% | −13.5% | +0.86% / −1.23% |

Every schedule trades on 100% of days. 4× costs about 0.03% of capital per 14 days against
1×, in exchange for clearly visible daily activity. Hourly was rejected: it adds hundreds of
tiny trades for no strategic reason. The tables above were computed with the 1× schedule;
at 4× those numbers are about 0.03 points lower per 14-day window.

## Shorting (researched 2026-09-28, not adopted)

The organizers confirmed 1x shorts are allowed (losses capped at the collateral, long and short
in the same pair permitted, no native stops). `scripts/short_research.py` added a short leg to the
live strategy: short (at 50% or 100% of the normal size) once a coin closes 3/6/10% below its
40-day EMA, cover back at the EMA; everything else unchanged (6-hourly rebalances, base costs,
shorts charged 0.1% on open and close). Selection on train, one check on validation.

| variant | train: mean 14d | train: continuous | train: max DD | train: composite | validation: mean 14d | validation: continuous |
|---|---|---|---|---|---|---|
| **long-only (live)** | **+0.90%** | **+47.4%** | **−23.8%** | **1.60** | −1.21% | −13.3% |
| short ×0.5 below −6% (best short on train) | +0.68% | +37.5% | −26.7% | 1.24 | −0.30% | −6.2% |
| short ×1.0 below −3% | +0.57% | +26.4% | −37.0% | 0.79 | +0.51% | −2.4% |

By regime (BTC trailing 30-day return), train: in **bear** stretches (15% of bars) shorts turned
−10% into +9% to +49%, but in **sideways** stretches (61% of bars) they deepened losses from −19%
to −39%/−55%, as shorts entered below the trend and were stopped back out at the EMA over and over.
Validation (a falling period) looks better for every short variant, but that's the same bear
effect, and the rule was to choose on train. **Every short variant is worse on train, so the short
leg is not adopted.** It behaves like a bet that the market will fall, not a robust improvement.
The largest move against any short was +14%, far from the +100% that would wipe out collateral.
The engine support (`allow_short`) and `trend_vol_long_short` stay available for research.

## Where the money goes, and the choppy-market filter (2026-09-29)

`scripts/loss_attribution.py` (live settings, 6-hourly rebalances, base costs):

| | train (Nov 2024 – Dec 2025) | validation (Jan – May 2026) |
|---|---|---|
| net P&L | +$47,424 | −$13,303 |
| in trend at full size (BTC + ETH) | +$71,518 | −$6,753 |
| **15% floor while out of trend** | **−$17,450** | **−$6,750** |
| trend episodes shorter than 14 days (whipsaws) | 7 episodes, −$18,739 | 2 episodes, −$3,941 |
| trend episodes of 14 days or more | 7 episodes, +$90,257 | 4 episodes, −$2,812 |
| sideways market stretches (61–63% of the time) | −$25,663 | −$13,885 |
| costs | $4,057 ($2,818 from 42 signal-change fills) | $987 |

(The categories overlap, e.g. a whipsaw is also an in-trend episode, so rows don't add up to the
net P&L.) A handful of long trends make all the money; whipsaws and the floor give a lot back.

**Choppy-market filter** (`scripts/chop_filter_research.py`): only enter a trend if Kaufman's
efficiency ratio over 10/20/30 days is at least 0.2/0.3/0.4. The adoption rule was written into the
script before it ran. **No variant passed.** Requiring a "clean" move delays entry into the good
trends (train bull-regime return fell from +101% to +56–99%) without consistently cutting whipsaw
losses, and every variant scored below live on train. Not adopted.

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

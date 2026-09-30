# Rustoo: a risk-managed BTC/ETH trend bot for the Roostoo hackathon

An autonomous trading bot for the APAC University Quant Trading Hackathon (Roostoo × Susquehanna × AWS),
built by Team125-100aqi (IITR). It trades BTC and ETH on Roostoo's mock exchange with a slow,
volatility-targeted trend-following strategy, and it was designed around the competition's scoring:
first **portfolio return**, then `0.4 × Sortino + 0.3 × Sharpe + 0.3 × Calmar` over a 14-day live period.

**In one paragraph:** the bot holds BTC and ETH while each is in an uptrend (price above its 40-day
average, with a 3% buffer against noise), sizes each position by its recent volatility, and drops a
coin to a small 5% "floor" position when its trend breaks. It rebalances to its targets every 6 hours
and otherwise trades only when a position has drifted meaningfully. In backtests it didn't reliably
beat the market's direction over two weeks, but in every period we tested it **made the worst
14-day loss 34–65% smaller** than holding BTC.

Contents: [Strategy](#1-the-strategy) · [How we arrived at it](#2-how-we-arrived-at-it) ·
[Implementation](#3-implementation) · [Backtesting](#4-backtesting) · [Trading engine](#5-the-trading-engine) ·
[Transaction fees](#6-transaction-fees-maker-and-taker) · [Risk management](#7-risk-management) ·
[Limitations](#8-limitations-and-honest-caveats) · [How we built this](#9-how-we-built-this) ·
[Running it](#10-running-it) · [Repository map](#11-repository-map)

---

## 1. The strategy

Every hour, for each of BTC and ETH:

| Step | Rule | Parameter (`config/strategy.yaml`) |
|---|---|---|
| **Trend filter** | A coin is *in trend* once its hourly close is 3% above its 40-day exponential moving average, and *out of trend* once it closes 3% below. In between, the previous state is kept, so a price hovering around the average doesn't flip the position. | `trend_span: 960` hourly bars, `band: 0.03` |
| **Position size** | Each coin gets an equal share of a 50% annualised volatility budget: weight = (0.5 / 2) ÷ its 30-day realised volatility. Calmer coins get bigger positions. | `target_vol: 0.5`, `vol_lookback: 720` |
| **Out of trend** | Keep 5% of the normal size (the "floor") instead of going fully to cash, so there's always a position to rebalance (the rules require a trade on at least 8 days). | `min_exposure: 0.05` |
| **Caps** | Total exposure ≤ 100% (no leverage), long-only, 0.5% of equity kept in cash for fees and rounding. | |
| **Trading** | Rebalance exactly to target at 00:00, 06:00, 12:00 and 18:00 UTC. At other hours, trade a coin only if its weight is ≥ 5 percentage points away from target. Market orders. | `rebalance_hours_utc`, `rebalance_threshold: 0.05` |

Code: `src/strategy/signals.py::trend_vol_target`, `src/features/trend.py`, `src/features/volatility.py`.

**The hypothesis** (ours, not proven by backtests): crypto majors trend over weeks, so staying out of
coins below their 40-day trend and sizing by volatility should cut the downside — drawdowns and
downside volatility, which the Sortino and Calmar terms punish — by more than it costs in missed upside.
We expected it to struggle in choppy, directionless markets, and our loss attribution confirms that
this is where it loses money (section 4).

## 2. How we arrived at it

We treated this as a search that most ideas would fail, and kept a record of what failed and why.

1. **Fast signals lose to fees.** Our first backtests (hourly bars, 86 coins, 2 years) tried
   cross-sectional momentum, EMA-crossover trend following, mean reversion and volatility-filtered
   momentum. All of them lost money after the 0.1% fee; cross-sectional momentum paid over 3× its
   capital in fees. Only buy-and-hold BTC was positive.
2. **Tuned parameters didn't survive unseen data.** Walk-forward testing (tune on 4 months, trade
   the next month, repeat) showed the best-looking parameters kept changing and lost money out of
   sample: in-sample composite scores of 30–40 turned negative the next month.
3. **Optimise for the real competition horizon.** The competition scores one 14-day run starting
   from cash, so we evaluated every candidate on *every* historical 14-day window, measuring return,
   worst case and trading days, rather than on multi-year Sharpe ratios.
4. **A slow trend with volatility sizing** was the only design that improved the bad tail
   consistently: in the validation period its worst 14-day window was −10.0% against BTC's −28.8%.
5. **Rule-driven adjustments.** After the organizers clarified that an "active day" is one with at
   least one trade, we reduced the out-of-trend floor from 15% to 5% (tested with a selection rule
   written before the test ran). We also moved from one to four scheduled rebalances a day, which
   gives several chances to register each day's trade for about 0.03% of capital per 14 days.

**Ideas we tested and rejected** (all in `docs/STRATEGY.md`, each with the script that tested it):

| Idea | Why rejected |
|---|---|
| Drawdown state machine (cut exposure after 5/10/20% drawdowns) | Sold after drops, bought back after rebounds: turned buy-and-hold's −16% into −21% out of sample |
| Short positions in downtrends | Big gains in falling markets, but deeper losses in choppy ones (61% of the time); worse overall on the selection data |
| Choppy-market entry filter (Kaufman efficiency ratio) | Delayed entry into the few big trends that make the money; no variant beat the live strategy |
| More coins (top 5/10 by liquidity, adding TRX) | No better than BTC+ETH on the selection data; altcoins crash together with BTC |
| Hourly rebalancing | Hundreds of tiny trades, higher costs, no benefit |

## 3. Implementation

Python 3.10+, pandas/numpy, no external trading framework. The same strategy function drives the
backtests and the live bot, so what was tested is what trades.

```
Binance hourly closes ──► strategy (trend_vol_target) ──► target weights
                                                              │
Roostoo balance + prices ──► order planner ──► broker (Roostoo API) ──► fills
                                  │                    │
                                  └──── audit trail (SQLite): every API call, decision, order, fee, equity
```

- **Signal data:** Roostoo has no price-history endpoint, so signals use Binance's public hourly
  klines for the same coins (`COIN/USD` on Roostoo ↔ `COINUSDT` on Binance). We checked Binance
  against our Bloomberg exports: 3–6 bps median price difference, 0.99+ return correlation.
- **Execution:** Roostoo's own prices and wallet, through a signed API client (HMAC-SHA256,
  verified against the worked example in Roostoo's documentation).
- **Separation:** strategy code never touches the network; it's a pure function of prices, which
  makes it testable and backtestable.

## 4. Backtesting

**Data:** 2 years of Binance 5-minute klines for all Roostoo pairs (Sep 2024 – Sep 2026), resampled
to hourly bars. Every bar is labelled by its UTC close time, so a signal only uses prices that
existed at that moment.

**Protocol** (`config/research.yaml`), chronological and never shuffled:

| Split | Period | Used for |
|---|---|---|
| Train | Sep 2024 – Dec 2025 | Developing ideas and choosing parameters |
| Validation | Jan – May 2026 | Checking finalists once |
| Holdout | Jun – Sep 2026 | One final evaluation (done on 2026-09-27, before any later change) |

Before each experiment we wrote down the rule that decides whether a change is adopted, and the
scripts print whether it passed. The code refuses to load holdout data unless explicitly asked to.

**Engine** (`backtest/engine.py`):
- A signal computed at an hourly close fills at the **next bar's open**; the engine refuses zero
  delay, and every strategy passes a test showing its past decisions can't change when future prices
  are altered.
- Fees, slippage, spread and market impact are charged per fill, and the accounting is checked:
  final equity = starting capital + gross P&L − costs, to the cent.
- No leverage; cash can't go negative; missing prices are held rather than valued at zero.

**Results for the current configuration** (base costs: 0.10% taker fee + 5 bps slippage per trade;
$100k starting capital):

| Period | | Return | Max drawdown | Sharpe | Mean 14-day return | Worst 14-day return | 14-day windows with ≥ 8 trading days |
|---|---|---|---|---|---|---|---|
| **Train** (Nov 2024 – Dec 2025) | **Strategy** | **+48.5%** | **−22.3%** | **1.31** | +0.94% | −12.2% | 100% |
| | BTC buy-and-hold | +28.8% | −34.8% | 0.71 | +0.41% | −18.5% | — |
| | 50/50 BTC+ETH buy-and-hold | +26.5% | −47.0% | 0.65 | +0.45% | −21.2% | — |
| **Validation** (Jan – May 2026) | **Strategy** | **−11.7%** | **−17.9%** | −1.12 | −1.02% | **−10.0%** | 100% |
| | BTC buy-and-hold | −15.8% | −35.6% | −0.69 | −1.51% | −28.8% | — |
| | 50/50 BTC+ETH buy-and-hold | −23.3% | −40.5% | −1.03 | −2.32% | −32.3% | — |
| **Holdout** (Jun – Sep 2026)* | **Strategy** | **+34.5%** | **−6.6%** | **3.20** | +3.84% | −4.6% | 100% |
| | BTC buy-and-hold | +14.0% | −21.1% | 1.20 | +3.20% | −11.3% | — |
| | 50/50 BTC+ETH buy-and-hold | +22.0% | −22.2% | 1.61 | +4.31% | −12.5% | — |

\*The holdout was evaluated once, on 2026-09-27, with the configuration at that time (15% floor, one
rebalance a day). The numbers above are recomputed for the current configuration as a report only;
no decision was made from them.

**What the results say:**
- **It loses money when the market falls** (validation), just much less: its worst two weeks were
  about a third as bad as BTC's.
- **Its typical two-week return is close to the market's;** the advantage is in the tail. In the
  holdout, 50/50 buy-and-hold had a higher mean 14-day return, but a worst window almost 3× as bad.
- **Where it loses** (`scripts/loss_attribution.py`, train, computed with the earlier 15% floor):
  seven long trends earned +$84,934, seven short-lived "whipsaw" entries that reversed within two
  weeks lost −$31,942, and costs were $4,057. Losses come from choppy markets, as the hypothesis
  predicted.
- **Robustness** (`scripts/robustness_report.py`, validation, computed with the earlier 15% floor
  and one rebalance a day): tripling fees or adding an hour of delay moves the
  result by about 1 percentage point; timing beat random entry with the same position sizes in 2 of 3
  periods; a Monte Carlo of 14-day outcomes (validation data, 8,000 paths, 4 seeds) gave a 61%
  chance of a loss, a median of −1.3% and a 5th percentile of −8.9% — i.e. results mostly follow
  the market.

The full research record, including every rejected idea and a bug we found in our own
analysis, is in [`docs/STRATEGY.md`](docs/STRATEGY.md).

## 5. The trading engine

`src/main.py` → `src/bot/runner.py`, deployed as a systemd service on AWS EC2 (restarts automatically).

- **Every minute:** read Roostoo prices and the wallet; save an equity snapshot every 15 minutes.
- **Two minutes after each hourly close:** fetch the last 2,000 hourly Binance closes (closed bars
  only), compute target weights with the same function as the backtests, and plan orders.
- **Order planning** (`src/execution/portfolio.py`): sells before buys; quantities rounded down to
  Roostoo's precision; orders under Roostoo's minimum ($1) skipped; buys sized so that cost plus
  fees fits the available cash; targets scaled to keep the 0.5% cash buffer.
- **Execution** (`src/execution/broker.py`): market orders, at least 10 seconds apart. The fill
  price, quantity, fee and maker/taker role are read from Roostoo's response, never assumed.
- **If an order's outcome is unknown** (timeout, server error), the bot does **not** resubmit; it
  looks the order up with `/v3/query_order` (the method the organizers recommend) to avoid duplicates.
- **Daily activity:** besides the four scheduled rebalances, if a UTC day reaches 12:00 with no
  filled order the bot forces one rebalance.
- **Audit trail** (`src/bot/store.py`, SQLite): every API request and whether it succeeded, every
  hourly decision (targets, current weights, reason for trading or not), every order with Roostoo's
  full response, and equity snapshots.

**Live test** (testing account, 2026-09-30): both first orders filled in ~4 ms; fees were exactly
0.10% (taker); the wallet matched the bot's records to the last decimal for BTC and ETH and within
$0.0006 for USD. The second order filled 12 bps worse than priced because it waited out a 60-second
gap between orders, which we then reduced to 10 seconds. Before that, a 5-day paper run caught one bug
we fixed: a fully invested portfolio couldn't rebalance because its cash buffer blocked every buy.

## 6. Transaction fees: maker and taker

Roostoo charges **0.10% for taker orders** (market orders, or limit orders that fill immediately)
and **0.05% for maker orders** (limit orders that rest on the book). The exchange decides the role.

- **The live bot uses market orders only, so every fill is a taker fill (0.10%).** We chose certainty
  of execution over the lower maker fee: a missed rebalance could cost more than the fee saved, and
  maker orders need logic for partial and unfilled orders. Moving to limit orders could save about
  half the fees (≈2% of capital over 14 months on train); it's the first improvement we'd make.
- **Backtests charge fees per fill** (`backtest/costs.py`): fee = maker share × 0.05% + taker share ×
  0.10%, plus slippage. The base case assumes **100% taker plus 5 bps slippage**. We also run an
  optimistic case (50% maker, 2 bps slippage) and a pessimistic one (0.15% taker, 15 bps
  slippage), and stress tests up to 3× fees and 100 bps.
- **The live bot never assumes a fee.** It records Roostoo's `CommissionChargeValue`,
  `CommissionPercent` and `Role` for every fill; in the live test these were exactly 0.10% and TAKER.
- **Fees shaped the design:** the 40-day trend, the 5% no-trade band and the 6-hourly schedule all
  exist because faster trading lost more to fees than it earned. Fees were 4.2% of capital over
  14 months on train (most of it from ~40 large signal changes, not the routine rebalances).
- **Orders are sized with fees included,** so a buy never fails for lack of cash to pay its fee.

## 7. Risk management

**Market risk**
- **Trend filter:** a coin that falls 3% below its 40-day average drops to a 5% position; this is
  what halved the worst 14-day losses in testing.
- **Volatility targeting:** position sizes shrink when a coin gets more volatile.
- **No leverage, long-only:** total exposure ≤ 100%. Shorting was tested and rejected (section 2).
- **Cash buffer:** 0.5% of equity is never invested, so fees and rounding can't overdraw the account.

**Operational risk**
- **Two switches for live trading:** orders are sent only if both `APP_ENV=live` and
  `LIVE_TRADING=true`; otherwise the bot runs against a simulated wallet.
- **Kill switch:** creating a file named `STOP` makes the bot keep deciding and logging, but send no orders.
- **Stale data guard:** if the latest price history is more than 2 hours old, the bot doesn't trade.
- **No duplicate orders:** order calls are never retried blindly; unknown outcomes are reconciled.
- **Exchange rules:** quantities rounded to Roostoo's precision, orders under the $1 minimum skipped.
- **Clock sync:** request timestamps follow Roostoo's server time (it rejects requests >60 s off).
- **Redundant daily trading:** four scheduled rebalances plus a midday fallback, so one failed
  order can't cost an active trading day.
- **Audit trail and logs** for every decision, order and API call; the service restarts automatically.

**Our intervention policy during the live period:** change the bot only to fix bugs, via a commit
and a restart (the organizers allow teams to update and redeploy); never because of a few days of P&L.

## 8. Limitations and honest caveats

- **Over 14 days, results mostly follow the market.** No strategy we tested reliably beat BTC/ETH's
  direction over two weeks; ours reduces losses in falls and keeps part of the gains in rises.
- **Choppy markets are the known weakness** (section 4). Our attempt to filter them failed.
- **Limited history:** two years is only about 50 independent 14-day periods, and both the train
  and validation periods were weak for crypto; the validation period was also looked at several
  times during development, so it's no longer a fully clean test.
- **Signals use Binance prices, trades use Roostoo's.** In our checks they agree closely, but they're
  not identical.
- **Backtest costs are modelled:** spread and slippage are assumptions; the live test agreed with
  them (0.10% fee, 1–12 bps slippage).

## 9. How we built this

We used an AI coding assistant (Anthropic's Claude, through Claude Code) extensively to write code,
run backtests and draft documentation; commits made with it are marked `Co-Authored-By: Claude`. The
team directed the work and made the decisions, including:

- which data to trust (Binance history, with our Bloomberg exports as a cross-check),
- the evaluation method (train/validation/holdout, 14-day windows, rules fixed before each test),
- what to ask the organizers, and how to act on their answers,
- the final choices: BTC+ETH only, long-only, the 5% floor, four rebalances a day, the 10-second
  order gap, and which ideas to reject.

*[Team: expand or rewrite this section in your own words, with who did what.]*

## 10. Running it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q                                   # ~300 tests
.venv/bin/python scripts/download_binance_history.py            # 2 years of market data (~400 MB, once)
```

**Backtests and research** (outputs go to `research/experiments/`, not committed):

```bash
.venv/bin/python scripts/backtest_report.py --split validation   # full interactive HTML report
.venv/bin/python scripts/robustness_report.py --split validation # stress tests, Monte Carlo, regimes
.venv/bin/python scripts/runs.py list                            # history of runs; `reproduce <id>` re-checks one
./run_app.sh                                                     # local web app for all of the above
```

**The bot:**

```bash
.venv/bin/python -m src.main --once     # one cycle
.venv/bin/python -m src.main            # run continuously (simulated wallet unless live trading is enabled)
.venv/bin/python -m src.main --status   # decisions, orders, active trading days
bash deployment/setup_ec2.sh            # on the EC2 instance: packages, clock sync, venv, systemd service
```

Credentials go only in `.env` (see `.env.example`), which is never committed.

More operational detail (research app pages, data import, forward testing, safety internals):
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

## 11. Repository map

```
config/            strategy.yaml (live strategy), config.yaml (API, fees, safety), research.yaml (splits, grids)
src/strategy/      signals.py — all strategies as pure functions of prices; the live one is trend_vol_target
src/features/      trend, volatility, momentum, volume indicators (all use past data only)
src/execution/     client.py (signed Roostoo API), portfolio.py (order planner), broker.py (live and simulated)
src/bot/           runner.py (the loop), store.py (SQLite audit trail), logging
src/data/          Binance history, live price history, Roostoo universe and exchange rules, Excel/CSV importer
backtest/          engine, costs, metrics, analytics, walk-forward, 14-day windows, robustness, reports
scripts/           research and operations scripts (one per experiment, each documents its own method)
app/               local research web app (Streamlit)
deployment/        EC2 setup script and systemd service
docs/              STRATEGY.md (full research record), DEVELOPMENT.md, COMPETITION_RULES.md, API_NOTES.md, BACKTESTER_PLAN.md
tests/             ~300 tests, including look-ahead, accounting-reconciliation and live-bot tests
```

# Roostoo Quant Trading Bot

Autonomous quant trading system for the HK vs AU vs IN Quant Trading
Hackathon (Roostoo × Susquehanna). Optimizes for the competition's actual
scoring function:

```
Composite Score = 0.4 * Sortino + 0.3 * Sharpe + 0.3 * Calmar
```

— not raw return. See `backtest/metrics.py::composite_score`.

**Live strategy:** risk-managed BTC/ETH trend core (40-day EMA trend filter with
hysteresis, volatility-targeted sizing, 15% exposure floor, daily 00:00 UTC
rebalance). Rationale and evidence: [`docs/STRATEGY.md`](docs/STRATEGY.md).
Competition constraints: [`docs/COMPETITION_RULES.md`](docs/COMPETITION_RULES.md).

## Status

Built and tested (185 tests): API client with safety gates, historical data
pipeline, look-ahead-safe backtester, walk-forward and 14-day window
research tooling, the selected live strategy, and the **live bot**:
signal → order planning → execution → reconciliation → SQLite audit trail,
with a paper mode, a kill switch, and EC2/systemd deployment. Remaining work
is listed in the roadmap at the end.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in ROOSTOO_API_KEY / ROOSTOO_API_SECRET once you have them
```

Run the tests:

```bash
.venv/bin/python -m pytest -q
```

Smoke-test the public API client (no credentials needed):

```bash
.venv/bin/python -c "
from src.execution.client import PublicMarketDataClient
c = PublicMarketDataClient()
print(c.get_server_time())
print(list(c.get_exchange_info()['TradePairs'].keys())[:5])
"
```

## Running the bot

```bash
.venv/bin/python -m src.main --once     # one iteration (smoke test)
.venv/bin/python -m src.main            # run continuously
.venv/bin/python -m src.main --status   # audit summary: decisions, orders, active trading days
```

- **Mode:** orders go to Roostoo only when `APP_ENV=live` **and** `LIVE_TRADING=true`
  (and API credentials are set). Otherwise the bot runs the same decisions against a
  simulated `PaperBroker` wallet ($100k, 0.1% taker fee). Paper and live keep separate
  databases (`data/state/bot-<mode>.sqlite3`).
- **Loop:** every 60s it reads Roostoo tickers and the balance and snapshots equity.
  A couple of minutes after each hourly bar closes, it recomputes the strategy's targets on
  Binance hourly closes (the backtested data) and trades toward them (`src/bot/runner.py`).
- **Execution policy:** same as the backtest. Trade an asset only when its weight drifts
  ≥5% from target; exact rebalance daily at 00:00 UTC. If a UTC day reaches 12:00 with no
  filled order, one exact rebalance is forced, since the rules require ≥8 trading days.
  MARKET orders, rounded to exchange precision, with Roostoo's minimum order size respected;
  sells before buys.
- **Audit trail** (`src/bot/store.py`): every API request and its outcome, every decision,
  every order with its real fill/fee/role from Roostoo, and equity snapshots.
  JSON logs go to `logs/bot.jsonl`.
- **Kill switch:** `touch STOP` in the repo root makes the bot keep deciding and logging but
  stop sending orders. `rm STOP` resumes.
- **Deployment:** `bash deployment/setup_ec2.sh` on the EC2 instance (systemd service,
  clock sync, venv, tests).

## Architecture

```
docs/API_NOTES.md        - what's verified about the Roostoo API vs. assumed
config/                  - non-secret config: fees, universe filters, strategy params
src/config/settings.py   - env vars + config.yaml; hard gate on live trading
src/execution/client.py  - PublicMarketDataClient / PrivateTradingClient (signed HMAC)
src/data/market_data.py  - normalized Ticker type + pluggable HistoricalDataSource
src/data/universe.py     - tradable universe + precision/min-notional rules, from exchangeInfo
src/data/roostoo_data.py - live feed wrapper + TickerBarBuilder (self-collected OHLCV)
src/data/binance.py      - Binance public-archive kline downloader (backtest history)
src/data/historical.py   - ParquetDataSource / BloombergExcelSource, resampling, wide panels
scripts/                 - download_binance_history, run_baselines, walk_forward, window_analysis, compare_sources
src/data/live_history.py - recent Binance hourly closes for the live signal (closed bars only)
src/execution/portfolio.py - wallet parsing + pure order planner (band, daily rebalance, precision, MiniOrder)
src/execution/broker.py  - LiveBroker (Roostoo, reconciles unknown outcomes) / PaperBroker (simulated)
src/bot/                 - runner (loop), store (SQLite audit trail), logging
src/risk/drawdown.py     - drawdown state machine (implemented, off: hurt in walk-forward)
src/main.py              - entry point
deployment/              - systemd unit + EC2 setup script
src/features/            - momentum, trend, volatility, volume, cross-sectional (all causal)
src/strategy/signals.py  - baseline strategies -> target weights (long-only, no leverage)
src/strategy/portfolio.py- vol-scaled weighting, constraints, rebalance-threshold hysteresis
backtest/engine.py       - chronological multi-asset sim with an enforced execution lag
backtest/costs.py        - maker/taker fee + slippage model, optimistic/base/pessimistic
backtest/metrics.py      - Sharpe/Sortino/Calmar/drawdown/turnover/composite score
tests/                   - incl. a live signature check against the doc's worked example,
                            and look-ahead-bias regression tests for every baseline strategy
```

### Live-trading safety

`PrivateTradingClient` refuses every state-changing call (place/cancel order,
short open/close) unless it was built with live trading enabled, and
`build_clients_from_settings` only enables it when **both** `APP_ENV=live`
and `LIVE_TRADING=true`. Read-only calls (balance, order queries) always work.
Shorting additionally needs `execution.allow_shorting: true`, and a
cancel-everything call needs `execution.allow_cancel_all_without_filter: true`
plus an explicit per-call flag. Order-creating calls are never retried
automatically (a retry after a timeout could double-submit); see
`docs/API_NOTES.md`.

Strategy code (`src/strategy`, `src/features`) never imports the API client —
it only ever sees normalized `pandas` data, so every strategy is testable and
backtestable without touching the network.

### Look-ahead safety

Two independent guards:

1. Every feature/strategy function is causal by construction (rolling/ewm
   windows only), and `tests/test_lookahead.py` proves it empirically: it
   mutates the *future* tail of a price series and asserts each strategy's
   past output doesn't change.
2. `backtest.BacktestEngine` refuses `execution_lag < 1` — a signal computed
   from data through bar `t` is only ever filled at bar `t + execution_lag`,
   never at bar `t`'s own price.

### Backtest accounting

`BacktestEngine` (`backtest/engine.py`):

- **Execution timing is explicit.** A signal from bar *t* fills at bar *t + execution_lag*, at
  that bar's **close** (default) or **open** (`execution_price="open"`, closest to the live bot,
  which trades minutes after each hourly bar closes).
- **Costs are itemised** (`backtest/costs.py`): fee (maker/taker mix), half of a modeled spread,
  modeled slippage, and modeled market impact (`k · participation^α`, only with volume data,
  never fabricated). Every fill is logged with its costs and participation (`result.fills`).
- **Gross vs net:** `result.gross_pnl` is the mark-to-market P&L of the positions actually held;
  `result.gross_value` is the same trades with no costs deducted. Tests enforce
  `final − initial = Σ gross P&L − Σ costs` to floating-point precision.
- It rejects negative targets and rows above 100% (no shorting, no leverage), never lets cash
  go negative to pay costs (buys are scaled down), and treats a missing price as "can't trade
  this bar": the position is held and marked at its last price.
- Sortino uses the standard downside deviation (`backtest/metrics.py::downside_deviation`).

Planned upgrades and the decisions behind them: [`docs/BACKTESTER_PLAN.md`](docs/BACKTESTER_PLAN.md).

### Historical data

Roostoo's API has no historical-candle endpoint — only a live ticker
snapshot (`/v3/ticker`) — so backtests use **Binance spot klines** for the
same coins (`COIN/USD` on Roostoo ↔ `COINUSDT` on Binance; 86 of Roostoo's 88
pairs exist there, all but OMNI and TON). Binance's bulk archive is public
and free; Bloomberg exports are used only to cross-check it.

```bash
.venv/bin/python scripts/download_binance_history.py   # 2y of 5m bars, all Roostoo pairs -> data/binance/5m/
.venv/bin/python scripts/run_baselines.py              # every baseline x cost scenario, ranked by composite score
.venv/bin/python scripts/compare_sources.py            # Binance vs Bloomberg exports in data/raw/bloomberg/
```

Every loaded bar is indexed by its **close time in UTC** — the moment its
close price is known — whatever the source's own convention (Bloomberg
exports are IST and labelled by bar start; Binance by open time). `data/` is
gitignored: market data, especially licensed Bloomberg data, must never be
committed.

### Research protocol: train / validation / holdout

`config/research.yaml` splits the history chronologically (never shuffled):

| split | window | use |
|---|---|---|
| train | 2024-09-26 → 2026-01-01 | idea development, walk-forward parameter search |
| validation | 2026-01-01 → 2026-06-01 | choosing between finalists |
| test (holdout) | 2026-06-01 → end | **one** final evaluation of the chosen strategy |

`backtest/splits.py::load_split_panel` never loads bars past the requested
split's end, and refuses the holdout unless explicitly allowed (scripts:
`--use-holdout`). Indicators warm up on the data before a split; only the
split's own window is scored.

```bash
.venv/bin/python scripts/run_baselines.py --split train          # fixed-parameter baselines
.venv/bin/python scripts/walk_forward.py --split train           # walk-forward search over config/research.yaml grids
.venv/bin/python scripts/window_analysis.py --split train        # every 14-day window from cash (the competition horizon)
.venv/bin/python scripts/backtest_report.py --split validation   # full HTML report + CSV/JSON exports for the live strategy
.venv/bin/python scripts/robustness_report.py --split validation # costs, stress, Monte Carlo, regimes, parameter landscapes, random entry
.venv/bin/python scripts/runs.py list                            # experiment history (every report run gets a run ID)
.venv/bin/python scripts/runs.py reproduce <run-id>              # re-run it and check every number matches
```

**Importing your own data** (Excel/CSV, e.g. Bloomberg exports):

```bash
.venv/bin/python scripts/import_data.py data/raw/bloomberg/*.xlsx --source bloomberg \
    --tz Asia/Kolkata --labelled-by start --reference data/binance/5m
```

Columns are detected from common aliases (a wrong or ambiguous guess stops with the `--map` to
pass). Every file gets a validation report: duplicates, gaps, OHLC consistency, non-24/7 trading,
extreme moves, identical content across files, and a price cross-check against a reference.
Clean files land in `data/imported/<source>/` in the same format the backtester reads; the
dataset library is `data/imported/library.json`.

**Forward testing:** `scripts/forward_test.py freeze --name <name>` snapshots the live
configuration and the data cut-off into `research/forward/<name>.json` (commit it). Later,
`scripts/forward_test.py evaluate --name <name>` tests that frozen configuration only on data
that arrived after the freeze, with no parameter overrides possible.

`scripts/backtest_report.py` writes `research/experiments/reports/<run>/report.html`: a single
self-contained page (works offline) with gross vs net equity, itemised costs, drawdown episodes,
monthly returns, rolling Sharpe/volatility, exposure, a FIFO trade table, tail risk (VaR/CVaR),
14-day window distributions, benchmarks, a random-entry baseline, statistical context,
limitations and reproducibility info. Next to it: `summary.json`, `config.json`, `equity.csv`,
`fills.csv`, `trades.csv`, `open_positions.csv`, `drawdowns.csv`, `random_entry.csv`.

`scripts/walk_forward.py` picks parameters on each rolling in-sample window
and trades them on the next unseen window; the stitched out-of-sample record
is the honest estimate. `BacktestEngine(rebalance_threshold=...)` skips
trades smaller than the band (exits always execute) and is searched as a
parameter, since trading costs dominate at hourly frequency.

The live bot can also build its own bars from repeated ticker polling
(`src/data/roostoo_data.py::TickerBarBuilder`).

## Roadmap

- **Prep period (Oct 1–3):** deploy to EC2, run in paper mode, then live with the
  competition key. Confirm through the bot (never manually) whether Roostoo accepts
  shorts, and compare Roostoo prices with Binance on recorded tickers.
- Maker (LIMIT) execution to cut fees from 0.10% to 0.05% where fills allow.
- Shorting in downtrends, if the competition allows it: the largest remaining lever,
  given both evaluation periods were falling markets.
- The holdout split (Jun–Sep 2026) has been used, once (results in `docs/STRATEGY.md`);
  further strategy changes are judged on train/validation or live results only.

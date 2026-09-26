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

This repo is being built incrementally, in the order laid out below. Phases
0–9 ("foundation") are done and tested; later phases (portfolio risk state
machine, execution/reconciliation daemon, logging/monitoring, walk-forward
validation, ML, AWS deployment) are not built yet — see the roadmap section.

Verified against the **official** API docs
([`roostoo/Roostoo-API-Documents`](https://github.com/roostoo/Roostoo-API-Documents))
and live-smoke-tested against `https://mock-api.roostoo.com` — see
`docs/API_NOTES.md` for everything that's confirmed vs. still an open
question.

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
scripts/                 - download_binance_history, run_baselines, compare_sources
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

`BacktestEngine` rejects negative target weights and rows summing above
100% (no shorting, no leverage), never lets cash go negative to pay fees
(buys are scaled down instead), and treats a missing price as "can't trade
this bar": the position is held and marked at its last known price rather
than valued at zero. Sortino uses the standard downside deviation
(`backtest/metrics.py::downside_deviation`).

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
```

`scripts/walk_forward.py` picks parameters on each rolling in-sample window
and trades them on the next unseen window; the stitched out-of-sample record
is the honest estimate. `BacktestEngine(rebalance_threshold=...)` skips
trades smaller than the band (exits always execute) and is searched as a
parameter, since trading costs dominate at hourly frequency.

The live bot can also build its own bars from repeated ticker polling
(`src/data/roostoo_data.py::TickerBarBuilder`).

## Roadmap (not yet built)

- Phase 10–13: rule-based market regime model, full risk engine (drawdown
  state machine: NORMAL → CAUTION → DEFENSIVE → EMERGENCY), correlation-aware
  portfolio construction.
- Phase 15–18: order execution engine (submit → confirm fill → reconcile),
  position reconciliation against Roostoo's actual account state, structured
  rotating logs, a persisted performance database (SQLite).
- Phase 20–25: YAML-driven parameter sweeps, walk-forward validation,
  parameter robustness/ablation studies, stress tests, Monte Carlo
  robustness diagnostics.
- Phase 26: ML/regime classification, only after the rule-based baseline
  above is validated.
- `deployment/`: systemd unit + AWS EC2 bring-up, `src/main.py` continuous
  run loop wiring all of the above together.

# Backtester upgrade plan

On 2026-09-27 the team proposed a full research-platform spec (web app, event-driven engine,
Monte Carlo, stress tests, …). Each item was judged against this project's goal: a 14-day
Roostoo competition from Oct 4, trading BTC/ETH hourly with $100k, long-only, no leverage,
ranked by return and then `0.4·Sortino + 0.3·Sharpe + 0.3·Calmar`.

## Accepted, in build order

| Step | Scope | Status |
|---|---|---|
| 1 | Explicit execution timing (next-bar open/close, latency); itemised costs (fee, spread, slippage, market impact); gross vs net P&L with reconciliation tests; fills ledger with participation; engine-level causality test; walk-forward leakage test | **done** |
| 2 | Full metrics (drawdown episodes and recovery, VaR/CVaR, monthly returns, rolling Sharpe/vol, exposure, per-sale P&L stats); benchmarks + random-entry baseline; self-contained HTML report with interactive charts + JSON/CSV export | planned |
| 3 | Cost/slippage sensitivity, stress scenarios and worst case; Monte Carlo block bootstrap with multiple seeds (14-day outcome distributions); regime analysis; parameter heatmaps (full landscape) | planned |
| 4 | Excel/CSV importer with alias-based column detection and a validation report (library + CLI); run registry (run ID, config/dataset hashes, git commit), frozen forward-test mode | planned |

## Rejected or deferred

| Item | Decision |
|---|---|
| React + FastAPI web app, drag-and-drop upload, 13 pages, progress bars | Deferred until after the competition. Static HTML reports + scripts cover the analysis now. |
| Leverage, margin, Kelly sizing | Rejected: the competition forbids leverage. |
| Limit/stop/trailing orders, stop-loss/take-profit, intrabar ambiguity handling | Deferred: the strategy and live bot use market orders. To be built together with maker (limit) execution if that's pursued. |
| Rewrite as an event-driven `on_bar` strategy API | Rejected: the engine is already sequential and proven causal; a rewrite adds risk and slows research. |
| Partial fills, capacity analysis | Rejected for now: $100k is negligible vs BTC/ETH volume. Participation is *reported* per fill, so this is shown rather than assumed. |
| Trade-order permutation Monte Carlo | Rejected: permuting returns of a continuously invested strategy doesn't change its final return. Block bootstrap is the meaningful variant. |
| DuckDB, Polars, PDF export, random/Bayesian search, multiprocessing, tax/brokerage fields | Not needed at this scale; SQLite/parquet and HTML/JSON/CSV exports suffice. |

## Findings from step 1

- **Dust fills:** the no-trade cutoff was an absolute 1e-12, so float residue (~1e-11 at $100k)
  was being logged as trades. It's now relative to equity. Re-running the documented results of
  the selected strategy (train, validation, holdout) gave identical numbers.
- **Daily activity vs. Roostoo's minimum order ($1 for BTC/USD and ETH/USD):** the selected
  strategy still has a fill of at least $10 on 148/151 validation days and 115/118 holdout days.
  Its typical daily rebalance fill is small, though (median ~$170–300 on $100k). Whether judges
  consider that "enough trades each day" is an open question.

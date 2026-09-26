# Backtester upgrade plan

On 2026-09-27 the team proposed a full research-platform spec (web app, event-driven engine,
Monte Carlo, stress tests, …). Each item was judged against this project's goal: a 14-day
Roostoo competition from Oct 4, trading BTC/ETH hourly with $100k, long-only, no leverage,
ranked by return and then `0.4·Sortino + 0.3·Sharpe + 0.3·Calmar`.

## Accepted, in build order

| Step | Scope | Status |
|---|---|---|
| 1 | Explicit execution timing (next-bar open/close, latency); itemised costs (fee, spread, slippage, market impact); gross vs net P&L with reconciliation tests; fills ledger with participation; engine-level causality test; walk-forward leakage test | **done** |
| 2 | Full metrics (drawdown episodes and recovery, VaR/CVaR, monthly returns, rolling Sharpe/vol, exposure, FIFO trade table and stats); benchmarks + random-entry baseline; self-contained HTML report with interactive charts + JSON/CSV export | **done** |
| 3 | Cost/slippage sensitivity, stress scenarios and worst case; Monte Carlo block bootstrap with multiple seeds (14-day outcome distributions); regime analysis; parameter heatmaps (full landscape); random-entry test on every split | **done** |
| 4 | Excel/CSV importer with alias-based column detection and a validation report (library + CLI); run registry (run ID, config/dataset hashes, git commit), reproduce command, frozen forward-test mode | **done** |

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

## Findings from step 2

- **Buy-and-hold basket bug (caught before use):** "never rebalance" had been expressed as a
  99.9% rebalance band, which also blocked the initial 50% entry, so the equal-weight benchmark
  never invested. Benchmarks now follow the passive basket's drifting weights with a 1% band;
  a test checks they buy exactly once and track the passive holdings.
- **Validation split, selected strategy (next-bar-open fills, base costs):** net −13.0%
  (gross −12.0%, costs $923) vs BTC −15.8%, ETH −32.4%, 50/50 basket −24.1%; max drawdown
  −19.3% vs −41.0% for the basket.
- **Random-entry baseline on validation:** with the strategy's own exposure levels and switching
  rate but random timing, the strategy beat only 60% of 20 seeds on return and 50% on Sharpe
  (median random −17.4%). In that period the smaller losses came mainly from *sizing*
  (about 50% average exposure), not from the trend filter's timing.

## Findings from step 3 (`scripts/robustness_report.py --split validation`)

Selected strategy, validation split (Jan–May 2026), next-bar-open fills, base costs.

- **Costs are not what loses money here.** Net −13.0% with $923 of costs; −12.4% even at zero
  fees. Every cost knob at 100 bps only takes it to −16% to −18%. (The first run reported this
  as a "0 bps breakeven", which is misleading; the report now says "negative even at zero cost".)
- **Stress:** the worst plausible combination (2× fees, 2× slippage, +1 bar latency, market impact
  at 10% of real volume) gives −18.8% vs −13.0% base. Latency or close-instead-of-open fills cost
  about 1 point. Participation stays tiny, so impact only matters in the reduced-liquidity case.
- **Monte Carlo (8,000 bootstrapped 14-day paths from validation, 4 seeds):** P(loss) 61%, median
  −1.3%, 5th percentile −8.9%, P(drawdown worse than 10%) 7.8%, P(beating BTC) 49%. The seeds
  agree to within 0.05% on the median. This resamples a falling period, so it inherits that drift.
- **Regimes (by BTC's trailing 30-day return):** in *bear* stretches the strategy lost −2.5% vs
  BTC −15.3% (11% average exposure); in *bull* it made +3.0% vs +5.8%; in *sideways* stretches
  (63% of bars) it lost −13.4% vs BTC −6.0%. That's the whipsaw the hypothesis predicted: entering
  after rises and exiting after drops around the trend line. (Part of the gap is the ETH sleeve;
  ETH fell 32% over the split.)
- **Timing vs random entry (30 seeds each):** train beats 90% of random-timing runs on return;
  validation 60% (43% on Sharpe); holdout 97%. So the trend timing added value in 2 of 3 periods
  and nothing measurable in the choppy validation period.
- **Parameter landscapes:** on train, the best cell for both mean 14-day return and Sharpe is
  trend_span 960 with target_vol 1.0 (live: 960 / 0.5). Neighbours are reasonably close (mean
  0.63% vs best 0.91%), a moderate plateau. On validation every cell is negative and the best
  cell moves elsewhere, so the optimum isn't stable across periods. `min_exposure = 0` scores
  best in both, since the floor costs return, but the floor is there for the ≥8-trading-days rule.

## Findings from step 4

- **Importer vs the Bloomberg exports** (`scripts/import_data.py data/raw/bloomberg/*.xlsx --tz
  Asia/Kolkata --labelled-by start --reference data/binance/5m`): it rejects exactly the two
  known-bad files with no manual input. BNB has almost no weekend bars and sits 9,989 bps from
  Binance's BNB. USDC has prices identical to the XRP file; XRP matches the reference, so XRP is
  kept and USDC rejected. Monero is imported with a warning about its 10-day gap, USDT with a
  "mostly flat" warning, and BTC, ETH, SOL, TRX, ZEC and DOGE cleanly (3–6 bps from Binance).
- **Bug found by the tests:** a column already in datetime format was converted to nanosecond
  integers and treated as epoch timestamps, so the source timezone was silently ignored (a 5.5h
  error for IST data). Excel imports weren't affected (cells arrive as datetime objects). Fixed,
  with a test.
- **Reproducibility check on real data:** a validation-split report registered as
  `backtest-…-fce2bab9` was re-run by `scripts/runs.py reproduce` and every number matched.

"""Robustness analysis (step 3 of the backtester upgrade).

Every analysis re-runs the *same* engine on the *same* bars with one thing
changed, so results are directly comparable with the base run:

- cost / slippage / spread sensitivity sweeps and breakeven levels
- named stress scenarios (costs, latency, fill timing, market impact with
  reduced liquidity, synthetic higher volatility) and the worst case among them
- Monte Carlo: stationary-style block bootstrap of the strategy's own bar
  returns into competition-length (14-day) paths, with optional cost and
  return perturbation, over several seeds; paired with a benchmark
- regime breakdown by benchmark trend and volatility
- 2-parameter landscapes (heatmaps) with a plateau-vs-peak measure
- random-entry baseline across splits

Stressed or bootstrapped results are *modeled* scenarios, not observations,
and are labelled as such in the report.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from backtest import analytics as an
from backtest.costs import CostModel
from backtest.engine import BacktestEngine
from backtest.windows import evaluate_windows, rolling_windows


@dataclass
class RunSpec:
    """Everything needed to re-run one backtest with variations."""

    close: pd.DataFrame  # may include warm-up rows before eval_index
    open: pd.DataFrame
    volume: pd.DataFrame
    eval_index: pd.DatetimeIndex
    strategy_fn: object
    params: dict
    cost_model: CostModel
    engine_kwargs: dict = field(default_factory=dict)  # execution_price, lag, bands, hours
    initial_capital: float = 100_000.0

    @property
    def ppy(self) -> float:
        return an.periods_per_year(self.eval_index)

    def weights(self, params: dict | None = None, close: pd.DataFrame | None = None) -> pd.DataFrame:
        close = self.close if close is None else close
        w = self.strategy_fn(close, **(self.params if params is None else params))
        return w.reindex(index=close.index, columns=close.columns).fillna(0.0)

    def run(self, *, cost_model: CostModel | None = None, params: dict | None = None,
            engine_overrides: dict | None = None, close: pd.DataFrame | None = None,
            open_: pd.DataFrame | None = None, volume: pd.DataFrame | None = None, weights: pd.DataFrame | None = None):
        close = self.close if close is None else close
        open_ = self.open if open_ is None else open_
        volume = self.volume if volume is None else volume
        w = self.weights(params, close) if weights is None else weights
        cm = cost_model or self.cost_model
        kwargs = {**self.engine_kwargs, **(engine_overrides or {})}
        engine = BacktestEngine(cm, initial_capital=self.initial_capital, **kwargs)
        idx = self.eval_index
        return engine.run(close.loc[idx], w.loc[idx], open_wide=open_.loc[idx],
                          volume_wide=volume.loc[idx] if volume is not None else None)


def _row(result, ppy: float) -> dict:
    s = an.return_risk_stats(result.portfolio_value, ppy)
    return {"net_return": s["total_return"], "sharpe": s["sharpe"], "sortino": s["sortino"], "calmar": s["calmar"],
            "composite": s["composite"], "max_drawdown": s["max_drawdown"], "total_costs": result.total_fees,
            "gross_return": float(result.gross_value.iloc[-1] / result.initial_capital - 1), "fills": len(result.fills)}


# -- cost sensitivity --------------------------------------------------------------------

COST_KNOBS = {
    # knob -> how a value in bps maps onto a CostModel
    "fee_bps": lambda cm, bps: replace(cm, taker_fee=bps / 1e4, maker_fee=bps / 1e4 / 2),
    "slippage_bps": lambda cm, bps: replace(cm, slippage_bps=bps),
    "spread_bps": lambda cm, bps: replace(cm, spread_bps=bps),
}


def cost_sensitivity(spec: RunSpec, knob: str, values_bps: list[float]) -> pd.DataFrame:
    """Re-run with one cost knob set to each value (others at base)."""
    rows = []
    for bps in values_bps:
        r = spec.run(cost_model=COST_KNOBS[knob](spec.cost_model, bps))
        rows.append({"knob": knob, "bps": bps, **_row(r, spec.ppy)})
    return pd.DataFrame(rows)


def breakeven(table: pd.DataFrame, column: str = "net_return", level: float = 0.0) -> float | None:
    """Smallest swept cost (bps) at which `column` falls to `level`, by
    linear interpolation. None if it stays above `level` over the whole
    range; NaN if it is already at or below `level` at the lowest swept
    cost (then costs aren't what makes it unprofitable)."""
    t = table.sort_values("bps")
    x, y = t["bps"].to_numpy(), t[column].to_numpy()
    if len(y) == 0:
        return None
    if y[0] <= level:
        return float("nan")
    for i in range(1, len(y)):
        if y[i] <= level:
            return float(x[i - 1] + (x[i] - x[i - 1]) * (y[i - 1] - level) / (y[i - 1] - y[i]))
    return None


# -- stress scenarios ----------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    name: str
    fee_mult: float = 1.0
    slippage_mult: float = 1.0
    spread_bps: float | None = None
    extra_latency_bars: int = 0
    execution_price: str | None = None
    impact_coef: float = 0.0
    impact_alpha: float = 0.5
    volume_mult: float = 1.0
    return_scale: float = 1.0  # synthetic: amplify every bar's return (signals see it too)
    synthetic: bool = False


DEFAULT_SCENARIOS = [
    Scenario("Base"),
    Scenario("2x fees", fee_mult=2),
    Scenario("3x fees", fee_mult=3),
    Scenario("2x slippage", slippage_mult=2),
    Scenario("3x slippage", slippage_mult=3),
    Scenario("Wider spread (20 bps)", spread_bps=20),
    Scenario("+1 bar latency", extra_latency_bars=1),
    Scenario("+3 bars latency", extra_latency_bars=3),
    Scenario("Fill at close instead of open", execution_price="close"),
    Scenario("Market impact, normal liquidity", impact_coef=0.1),
    Scenario("Market impact, 10% of liquidity", impact_coef=0.1, volume_mult=0.1),
    Scenario("Volatility x1.5 (synthetic prices)", return_scale=1.5, synthetic=True),
    Scenario("Combined: 2x fees + 2x slippage + 1 bar latency + impact at 10% liquidity",
             fee_mult=2, slippage_mult=2, extra_latency_bars=1, impact_coef=0.1, volume_mult=0.1),
]


def _scaled_returns(frame: pd.DataFrame, scale: float) -> pd.DataFrame:
    """Prices whose every bar return (around its mean) is scaled by `scale`,
    starting from the same first price. Synthetic — for stress only."""
    r = frame.pct_change()
    adj = r.mean() + (r - r.mean()) * scale
    out = (1 + adj.fillna(0.0)).cumprod() * frame.bfill().iloc[0]
    return out.where(frame.notna())


def run_scenario(spec: RunSpec, sc: Scenario) -> dict:
    cm = spec.cost_model.scaled(fee_mult=sc.fee_mult, slippage_mult=sc.slippage_mult)
    if sc.spread_bps is not None:
        cm = replace(cm, spread_bps=sc.spread_bps)
    if sc.impact_coef > 0:
        cm = replace(cm, impact_coef=sc.impact_coef, impact_alpha=sc.impact_alpha)
    overrides = {}
    if sc.extra_latency_bars:
        overrides["execution_lag"] = spec.engine_kwargs.get("execution_lag", 1) + sc.extra_latency_bars
    if sc.execution_price:
        overrides["execution_price"] = sc.execution_price
    close, open_, volume = spec.close, spec.open, spec.volume
    if sc.return_scale != 1.0:
        close = _scaled_returns(spec.close, sc.return_scale)
        open_ = close.shift(1).fillna(close)  # consistent synthetic opens
    if sc.volume_mult != 1.0:
        volume = spec.volume * sc.volume_mult
    r = spec.run(cost_model=cm, engine_overrides=overrides, close=close, open_=open_, volume=volume)
    row = _row(r, spec.ppy)
    row["max_participation"] = float(r.fills["participation"].max()) if len(r.fills) and r.fills["participation"].notna().any() else float("nan")
    return {"scenario": sc.name, "synthetic": sc.synthetic, **row}


def stress_test(spec: RunSpec, scenarios: list[Scenario] = DEFAULT_SCENARIOS) -> pd.DataFrame:
    table = pd.DataFrame([run_scenario(spec, sc) for sc in scenarios])
    base = table.iloc[0]
    for col in ("net_return", "sharpe", "composite"):
        table[f"{col}_vs_base"] = table[col] - base[col]
    return table


# -- Monte Carlo (block bootstrap) ------------------------------------------------------------

@dataclass(frozen=True)
class MonteCarloConfig:
    simulations: int = 2000
    seeds: tuple = (42, 123, 2026, 9999)
    horizon_days: float = 14.0
    block_hours: float = 48.0
    cost_sigma: float = 0.0  # lognormal sigma of a per-path cost multiplier
    return_noise_bps: float = 0.0  # per-bar N(0, sd) added to returns
    percentiles: tuple = (5, 10, 25, 50, 75, 90, 95)
    drawdown_threshold: float = 0.10


def _block_indices(rng, n_obs: int, horizon: int, block: int, sims: int) -> np.ndarray:
    """(sims, horizon) indices built from random contiguous blocks."""
    n_blocks = int(np.ceil(horizon / block))
    starts = rng.integers(0, n_obs - block + 1, size=(sims, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(sims, -1)
    return idx[:, :horizon]


def monte_carlo(result, ppy: float, cfg: MonteCarloConfig, benchmark_equity: pd.Series | None = None) -> dict:
    """Bootstrap competition-length paths from the run's own bar returns.
    Gross returns and cost returns are resampled together (same blocks), so
    costs can be perturbed independently; a benchmark, if given, is sampled
    with the same blocks for a paired comparison."""
    prev = result.portfolio_value.shift(1).fillna(result.initial_capital).to_numpy()
    gross_r = result.gross_pnl.to_numpy() / prev
    cost_r = result.fees_per_period.to_numpy() / prev
    bench_r = benchmark_equity.pct_change().fillna(0.0).to_numpy() if benchmark_equity is not None else None
    bar_hours = 365.25 * 24 / ppy
    horizon = int(round(cfg.horizon_days * 24 / bar_hours))
    block = max(1, int(round(cfg.block_hours / bar_hours)))
    if len(gross_r) < block * 2 or horizon < 2:
        raise ValueError("not enough history for the requested block length / horizon")

    per_seed, all_rows = [], []
    for seed in cfg.seeds:
        rng = np.random.default_rng(seed)
        idx = _block_indices(rng, len(gross_r), horizon, block, cfg.simulations)
        mult = rng.lognormal(0.0, cfg.cost_sigma, size=(cfg.simulations, 1)) if cfg.cost_sigma > 0 else 1.0
        noise = rng.normal(0.0, cfg.return_noise_bps / 1e4, size=idx.shape) if cfg.return_noise_bps > 0 else 0.0
        r = gross_r[idx] - cost_r[idx] * mult + noise
        paths = np.cumprod(1 + r, axis=1)
        total = paths[:, -1] - 1
        peaks = np.maximum.accumulate(np.concatenate([np.ones((len(paths), 1)), paths], axis=1), axis=1)[:, 1:]
        mdd = (paths / peaks - 1).min(axis=1)
        sd = r.std(axis=1, ddof=1)
        sharpe = np.divide(r.mean(axis=1), sd, out=np.zeros_like(sd), where=sd > 0) * np.sqrt(ppy)
        rows = pd.DataFrame({"seed": seed, "total_return": total, "max_drawdown": mdd, "sharpe": sharpe,
                             "worst_bar": r.min(axis=1)})
        if bench_r is not None:
            b_total = np.cumprod(1 + bench_r[idx], axis=1)[:, -1] - 1
            rows["benchmark_return"] = b_total
        all_rows.append(rows)
        per_seed.append({
            "seed": seed, "median_return": float(np.median(total)), "p5_return": float(np.percentile(total, 5)),
            "p_loss": float((total < 0).mean()), f"p_dd_gt_{cfg.drawdown_threshold:.0%}": float((mdd < -cfg.drawdown_threshold).mean()),
            "p95_drawdown": float(np.percentile(mdd, 5)),
            **({"p_beats_benchmark": float((rows.total_return > rows.benchmark_return).mean())} if bench_r is not None else {}),
        })
    sims = pd.concat(all_rows, ignore_index=True)
    pct = {f"p{p}": {m: float(np.percentile(sims[m], p)) for m in ("total_return", "max_drawdown", "sharpe")}
           for p in cfg.percentiles}
    fan_idx = _block_indices(np.random.default_rng(cfg.seeds[0]), len(gross_r), horizon, block, min(cfg.simulations, 2000))
    fan_paths = np.cumprod(1 + gross_r[fan_idx] - cost_r[fan_idx], axis=1)
    fan = pd.DataFrame({f"p{p}": np.percentile(fan_paths, p, axis=0) for p in (5, 25, 50, 75, 95)})
    fan.index = np.arange(1, horizon + 1) * bar_hours / 24  # days
    summary = {
        "simulations_per_seed": cfg.simulations, "seeds": list(cfg.seeds), "horizon_days": cfg.horizon_days,
        "block_hours": cfg.block_hours, "cost_sigma": cfg.cost_sigma, "return_noise_bps": cfg.return_noise_bps,
        "p_loss": float((sims.total_return < 0).mean()),
        "p_below_minus_10pct": float((sims.total_return < -0.10).mean()),
        f"p_drawdown_worse_than_{cfg.drawdown_threshold:.0%}": float((sims.max_drawdown < -cfg.drawdown_threshold).mean()),
        "percentiles": pct,
        "seed_spread_median_return": float(pd.DataFrame(per_seed).median_return.max() - pd.DataFrame(per_seed).median_return.min()),
    }
    if bench_r is not None:
        summary["p_beats_benchmark"] = float((sims.total_return > sims.benchmark_return).mean())
    return {"summary": summary, "per_seed": pd.DataFrame(per_seed), "simulations": sims, "fan": fan}


# -- regimes -------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RegimeConfig:
    lookback_days: float = 30.0
    trend_threshold: float = 0.10  # trailing return above/below +-10% = bull/bear
    vol_quantile: float = 0.5  # trailing vol above/below its median = high/low vol


def classify_regimes(benchmark_close: pd.Series, ppy: float, cfg: RegimeConfig) -> pd.DataFrame:
    """Trailing (causal) regime labels from a benchmark: trend by trailing
    return, volatility by trailing realized vol relative to its own median."""
    bars = max(2, int(round(cfg.lookback_days * ppy / 365.25)))
    trail_ret = benchmark_close / benchmark_close.shift(bars) - 1
    trail_vol = benchmark_close.pct_change().rolling(bars, min_periods=bars).std() * np.sqrt(ppy)
    trend = pd.Series("sideways", index=benchmark_close.index, dtype=object)
    trend[trail_ret > cfg.trend_threshold] = "bull"
    trend[trail_ret < -cfg.trend_threshold] = "bear"
    vol = pd.Series(np.where(trail_vol > trail_vol.quantile(cfg.vol_quantile), "high vol", "low vol"), index=benchmark_close.index)
    labels = pd.DataFrame({"trend": trend, "volatility": vol})
    labels[trail_ret.isna() | trail_vol.isna()] = np.nan
    return labels


def regime_breakdown(result, benchmark_equity: pd.Series, labels: pd.DataFrame, ppy: float) -> pd.DataFrame:
    strat_r = result.portfolio_value.pct_change().fillna(0.0)
    bench_r = benchmark_equity.pct_change().fillna(0.0)
    exposure = result.weights_history.sum(axis=1)
    costs = result.fees_per_period
    rows = []
    for dim in ("trend", "volatility"):
        for label, idx in labels.groupby(dim).groups.items():
            s, b = strat_r.loc[idx], bench_r.loc[idx]

            def sharpe(x):
                return float(x.mean() / x.std() * np.sqrt(ppy)) if x.std() > 0 else 0.0
            rows.append({"dimension": dim, "regime": label, "share_of_bars": len(idx) / len(labels.dropna()),
                         "strategy_return": float((1 + s).prod() - 1), "benchmark_return": float((1 + b).prod() - 1),
                         "strategy_sharpe": sharpe(s), "benchmark_sharpe": sharpe(b),
                         "strategy_worst_bar": float(s.min()), "average_exposure": float(exposure.loc[idx].mean()),
                         "costs": float(costs.loc[idx].sum())})
    return pd.DataFrame(rows)


# -- parameter landscape ----------------------------------------------------------------------------

def parameter_landscape(spec: RunSpec, x: str, xs: list, y: str, ys: list, window_days: float = 14.0) -> pd.DataFrame:
    """Evaluate a 2-D grid (other params at their spec values). Metrics: the
    continuous run over the evaluation period, plus the mean and 10th
    percentile of 14-day windows from cash."""
    windows = rolling_windows(spec.eval_index, pd.Timedelta(days=window_days), pd.Timedelta(days=1))
    rows = []
    for xv in xs:
        for yv in ys:
            params = {**spec.params, x: xv, y: yv}
            w = spec.weights(params)
            r = spec.run(weights=w)
            row = {x: xv, y: yv, **_row(r, spec.ppy)}
            if windows:
                win = evaluate_windows(spec.close, w, windows, spec.cost_model, spec.ppy, spec.engine_kwargs,
                                       spec.initial_capital, open_wide=spec.open,
                                       volume_wide=spec.volume if spec.cost_model.uses_impact else None)
                row["window_mean_return"] = float(win["return"].mean())
                row["window_p10_return"] = float(win["return"].quantile(0.1))
            rows.append(row)
    return pd.DataFrame(rows)


def plateau_score(grid: pd.DataFrame, x: str, y: str, metric: str) -> dict:
    """Best cell vs the mean of its immediate neighbours: a broad plateau
    has neighbours close to the best; a narrow peak doesn't."""
    piv = grid.pivot(index=y, columns=x, values=metric)
    arr = piv.to_numpy()
    i, j = np.unravel_index(np.nanargmax(arr), arr.shape)
    neigh = [arr[a, b] for a in range(max(0, i - 1), min(arr.shape[0], i + 2))
             for b in range(max(0, j - 1), min(arr.shape[1], j + 2)) if (a, b) != (i, j)]
    return {"metric": metric, "best": float(arr[i, j]), "best_at": {x: piv.columns[j], y: piv.index[i]},
            "neighbour_mean": float(np.nanmean(neigh)) if neigh else float("nan"),
            "grid_median": float(np.nanmedian(arr))}


# -- random entry across splits ------------------------------------------------------------------------

def random_entry_test(spec: RunSpec, seeds: list[int]) -> dict:
    from backtest.benchmarks import random_entry_weights

    base = spec.run()
    idx = spec.eval_index
    strat_w = spec.weights().loc[idx]
    rows = []
    for seed in seeds:
        rw = random_entry_weights(spec.close.loc[idx], strat_w, seed)
        r = spec.run(weights=rw.reindex(spec.close.index).fillna(0.0))
        rows.append({"seed": seed, **_row(r, spec.ppy)})
    df = pd.DataFrame(rows)
    b = _row(base, spec.ppy)
    return {"strategy": b, "random": df,
            "return_percentile": float((df.net_return < b["net_return"]).mean()),
            "sharpe_percentile": float((df.sharpe < b["sharpe"]).mean()),
            "composite_percentile": float((df.composite < b["composite"]).mean())}

"""Backtest report: one self-contained HTML file (interactive Plotly charts,
plotly.js embedded, works offline) plus JSON/CSV exports.

`run_report` does the work: runs the strategy, benchmarks and random-entry
baselines through the same engine on the same bars, computes the analytics,
and writes everything to an output directory. The report labels every
input as observed (market data) or modeled (assumptions) and ends with the
limitations of the evidence.
"""
from __future__ import annotations

import hashlib
import html
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go

from backtest import analytics as an
from backtest.benchmarks import BENCHMARK_BAND, buy_and_hold_weights, random_entry_weights
from backtest.costs import CostModel
from backtest.engine import BacktestEngine
from backtest.windows import evaluate_windows, rolling_windows

REPO_ROOT = Path(__file__).resolve().parents[1]

COLORS = {"net": "#2563eb", "gross": "#94a3b8", "bench": ["#f59e0b", "#10b981", "#a855f7", "#64748b"],
          "neg": "#dc2626", "pos": "#16a34a"}


@dataclass
class ReportConfig:
    strategy_name: str
    strategy_fn: object  # callable(close, **params) -> weights
    strategy_params: dict
    cost_model: CostModel
    execution_price: str = "open"
    execution_lag: int = 1
    rebalance_threshold: float = 0.0
    rebalance_hours_utc: tuple = ()
    initial_capital: float = 100_000.0
    benchmark_assets: list = field(default_factory=lambda: ["BTC/USD", "ETH/USD"])
    random_seeds: list = field(default_factory=lambda: list(range(20)))
    window_days: int = 14
    rolling_days: int = 30
    period_label: str = ""
    data_source: str = "Binance spot klines (data/binance/5m), resampled to 1h"
    hypothesis: str = ""


# -- helpers ---------------------------------------------------------------------

def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() + (" (uncommitted changes)" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def _hash_frame(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()).hexdigest()[:16]


def _pct(x, digits=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:.{digits}f}%"


def _num(x, digits=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:,.{digits}f}"


def _money(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"−${-x:,.0f}" if x < 0 else f"${x:,.0f}"


def _td(x):
    if x is None or pd.isna(x):
        return "—"
    x = pd.Timedelta(x)
    return f"{x.days}d {x.seconds // 3600}h" if x.days else f"{x.seconds // 3600}h {(x.seconds % 3600) // 60}m"


def _chart(fig: go.Figure, height: int = 380) -> str:
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=50, r=20, t=78, b=40),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#6b7280", size=12),
                      title=dict(x=0, xanchor="left", y=0.98, yanchor="top", font=dict(size=14)),
                      legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom"),
                      hovermode="x unified")
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})


def _table(rows: list[tuple], headers: tuple, numeric_from: int = 1) -> str:
    head = "".join(f"<th{' class=num' if i >= numeric_from else ''}>{html.escape(str(h))}</th>" for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(f"<td{' class=num' if i >= numeric_from else ''}>{html.escape(str(c))}</td>" for i, c in enumerate(r)) + "</tr>"
        for r in rows
    )
    return f"<div class=tablewrap><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


# -- the run ------------------------------------------------------------------------

def _engine(cfg: ReportConfig, cost_model: CostModel | None = None, threshold: float | None = None,
            hours: tuple | None = None) -> BacktestEngine:
    return BacktestEngine(cost_model or cfg.cost_model, initial_capital=cfg.initial_capital,
                          execution_lag=cfg.execution_lag, execution_price=cfg.execution_price,
                          rebalance_threshold=cfg.rebalance_threshold if threshold is None else threshold,
                          rebalance_hours_utc=cfg.rebalance_hours_utc if hours is None else hours)


def run_report(cfg: ReportConfig, close: pd.DataFrame, open_: pd.DataFrame, volume: pd.DataFrame,
               eval_index: pd.DatetimeIndex, out_dir: Path) -> dict:
    """Run everything on bars `eval_index` (weights may warm up on earlier
    rows of `close`) and write the report + exports to `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ppy = an.periods_per_year(eval_index)
    weights_full = cfg.strategy_fn(close, **cfg.strategy_params)
    weights_full = weights_full.reindex(index=close.index, columns=close.columns).fillna(0.0)
    c, o, v, w = (x.loc[eval_index] for x in (close, open_, volume, weights_full))
    use_volume = v if cfg.cost_model.uses_impact or v.notna().any().any() else None

    result = _engine(cfg).run(c, w, open_wide=o, volume_wide=use_volume)
    net, gross = result.portfolio_value, result.gross_value

    benches = {}
    for a in cfg.benchmark_assets:
        benches[f"Buy & hold {a.split('/')[0]}"] = buy_and_hold_weights(c, [a])
    if len(cfg.benchmark_assets) > 1:
        benches["Equal-weight basket (" + "+".join(a.split("/")[0] for a in cfg.benchmark_assets) + ")"] = \
            buy_and_hold_weights(c, cfg.benchmark_assets)
    bench_results = {name: _engine(cfg, threshold=BENCHMARK_BAND, hours=()).run(c, bw, open_wide=o, volume_wide=use_volume)
                     for name, bw in benches.items()}

    random_rows = []
    for seed in cfg.random_seeds:
        rw = random_entry_weights(c, w, seed)
        rr = _engine(cfg).run(c, rw, open_wide=o, volume_wide=use_volume)
        st = an.return_risk_stats(rr.portfolio_value, ppy)
        random_rows.append({"seed": seed, "total_return": st["total_return"], "sharpe": st["sharpe"],
                            "max_drawdown": st["max_drawdown"], "composite": st["composite"],
                            "costs": rr.total_fees})
    random_df = pd.DataFrame(random_rows)

    stats_net = an.return_risk_stats(net, ppy)
    stats_gross = an.return_risk_stats(gross, ppy)
    trades, open_pos = an.fifo_trades(result.fills, c.ffill().iloc[-1])
    tstats = an.trade_statistics(trades)
    gross_trades = trades.assign(net_pnl=trades["gross_pnl"]) if len(trades) else trades
    tstats_gross = an.trade_statistics(gross_trades)
    dd_eps = an.drawdown_episodes(net, min_depth=0.01)  # table: drawdowns of at least 1%
    dd_sum = an.drawdown_summary(net)
    expo = an.exposure_stats(result, ppy)
    costs = an.cost_summary(result)
    cons = an.consistency(net)
    ctx = an.statistical_context(net, ppy, cfg.window_days)
    bar_seconds = an.SECONDS_PER_YEAR / ppy
    roll = an.rolling_metrics(net, int(round(cfg.rolling_days * 86400 / bar_seconds)), ppy)

    # Competition-horizon windows (from cash), strategy vs the first benchmark.
    windows = rolling_windows(eval_index, pd.Timedelta(days=cfg.window_days), pd.Timedelta(days=1))
    common = {"execution_price": cfg.execution_price, "execution_lag": cfg.execution_lag}
    win_engine_kw = {**common, "rebalance_threshold": cfg.rebalance_threshold, "rebalance_hours_utc": tuple(cfg.rebalance_hours_utc)}
    first_bench = next(iter(benches.items()))
    if windows:
        win_strategy = evaluate_windows(c, w, windows, cfg.cost_model, ppy, win_engine_kw, cfg.initial_capital,
                                        open_wide=o, volume_wide=use_volume)
        win_bench = evaluate_windows(c, first_bench[1], windows, cfg.cost_model, ppy,
                                     {**common, "rebalance_threshold": BENCHMARK_BAND}, cfg.initial_capital,
                                     open_wide=o, volume_wide=use_volume)
    else:
        win_strategy = win_bench = pd.DataFrame()

    bench_stats = {name: an.return_risk_stats(r.portfolio_value, ppy) for name, r in bench_results.items()}
    summary = {
        "strategy": cfg.strategy_name,
        "period": {"start": str(eval_index[0]), "end": str(eval_index[-1]), "label": cfg.period_label,
                   "bars": len(eval_index), "periods_per_year": ppy},
        "net": stats_net, "gross": stats_gross, "costs": costs, "exposure": expo, "drawdown": dd_sum,
        "trades_net": tstats, "trades_gross": tstats_gross, "consistency": cons, "statistical_context": ctx,
        "benchmarks": bench_stats,
        "random_entry": {
            "seeds": cfg.random_seeds,
            "strategy_return_percentile": float((random_df.total_return < stats_net["total_return"]).mean()) if len(random_df) else None,
            "strategy_sharpe_percentile": float((random_df.sharpe < stats_net["sharpe"]).mean()) if len(random_df) else None,
            "median_random_return": float(random_df.total_return.median()) if len(random_df) else None,
        },
        "windows": _window_summary(win_strategy, win_bench, first_bench[0]),
    }
    reproducibility = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "plotly": plotly.__version__,
        "data_hash_close": _hash_frame(c), "data_hash_open": _hash_frame(o),
        "config": {
            "strategy": cfg.strategy_name, "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.strategy_params.items()},
            "costs": asdict(cfg.cost_model), "execution_price": cfg.execution_price, "execution_lag": cfg.execution_lag,
            "rebalance_threshold": cfg.rebalance_threshold, "rebalance_hours_utc": list(cfg.rebalance_hours_utc),
            "initial_capital": cfg.initial_capital, "benchmark_assets": cfg.benchmark_assets,
            "random_seeds": cfg.random_seeds, "window_days": cfg.window_days, "data_source": cfg.data_source,
        },
    }
    summary["reproducibility"] = reproducibility

    # -- exports
    curves = pd.DataFrame({"net_equity": net, "gross_equity": gross, "drawdown": an.drawdown_series(net),
                           "costs": result.fees_per_period, "gross_pnl": result.gross_pnl})
    for name, r in bench_results.items():
        curves[name] = r.portfolio_value
    curves.to_csv(out_dir / "equity.csv")
    result.fills.to_csv(out_dir / "fills.csv", index=False)
    trades.to_csv(out_dir / "trades.csv", index=False)
    open_pos.to_csv(out_dir / "open_positions.csv", index=False)
    dd_eps.to_csv(out_dir / "drawdowns.csv", index=False)
    random_df.to_csv(out_dir / "random_entry.csv", index=False)
    if len(win_strategy):
        win_strategy.to_csv(out_dir / "windows_strategy.csv", index=False)
        win_bench.to_csv(out_dir / "windows_benchmark.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_dir / "config.json").write_text(json.dumps(reproducibility["config"], indent=2, default=str))

    page = _render(cfg, summary, result, net, gross, bench_results, dd_eps, trades, open_pos, roll,
                   random_df, win_strategy, win_bench, first_bench[0], c)
    (out_dir / "report.html").write_text(page)
    return summary


def _window_summary(ws: pd.DataFrame, wb: pd.DataFrame, bench_name: str) -> dict:
    if ws.empty:
        return {}

    def dist(df):
        r = df["return"]
        return {"windows": len(df), "mean": float(r.mean()), "median": float(r.median()),
                "p_positive": float((r > 0).mean()), "p10": float(r.quantile(0.1)), "p90": float(r.quantile(0.9)),
                "worst": float(r.min()), "worst_drawdown": float(df["max_drawdown"].min()),
                "median_trading_days": float(df["trading_days"].median()),
                "p_trading_days_ge_8": float((df["trading_days"] >= 8).mean())}
    return {"strategy": dist(ws), "benchmark": dist(wb), "benchmark_name": bench_name}


# -- rendering -------------------------------------------------------------------------

CSS = """
:root{--bg:#f8fafc;--card:#ffffff;--text:#0f172a;--muted:#64748b;--border:#e2e8f0;--accent:#2563eb;--warn:#b45309;--pos:#16a34a;--neg:#dc2626}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0b1120;--card:#111827;--text:#e5e7eb;--muted:#94a3b8;--border:#1f2937;--accent:#60a5fa;--warn:#fbbf24;--pos:#4ade80;--neg:#f87171}}
:root[data-theme="dark"]{--bg:#0b1120;--card:#111827;--text:#e5e7eb;--muted:#94a3b8;--border:#1f2937;--accent:#60a5fa;--warn:#fbbf24;--pos:#4ade80;--neg:#f87171}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 10px;padding-top:8px;border-top:1px solid var(--border)}
h3{font-size:14px;margin:18px 0 6px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.sub{color:var(--muted);margin:0 0 16px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:12px 14px}.card .k{color:var(--muted);font-size:12px}
.card .v{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums}.pos{color:var(--pos)}.neg{color:var(--neg)}
.panel{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:8px 12px;margin:10px 0}
.tablewrap{overflow-x:auto;margin:8px 0}table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--border);border-radius:8px;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;border-bottom:1px solid var(--border);text-align:left;white-space:nowrap}th{color:var(--muted);font-weight:600;font-size:12px}
td.num,th.num{text-align:right}.tag{display:inline-block;font-size:11px;padding:1px 7px;border-radius:9px;border:1px solid var(--border);color:var(--muted);margin-left:6px}
.tag.modeled{color:var(--warn);border-color:var(--warn)}.note{color:var(--muted);font-size:13px}
nav{display:flex;flex-wrap:wrap;gap:6px 14px;margin:8px 0 0;font-size:13px}nav a{color:var(--accent);text-decoration:none}
ul{margin:6px 0 6px 18px;padding:0}li{margin:3px 0}code{font-size:12px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}@media (max-width:760px){.grid2{grid-template-columns:1fr}}
"""


def _cls(x):
    return "pos" if x is not None and np.isfinite(x) and x > 0 else ("neg" if x is not None and np.isfinite(x) and x < 0 else "")


def _render(cfg, s, result, net, gross, bench_results, dd_eps, trades, open_pos, roll, random_df,
            win_s, win_b, bench_name, close) -> str:
    n, g, cst, ex, tn = s["net"], s["gross"], s["costs"], s["exposure"], s["trades_net"]
    sections = []

    cards = [("Net return", _pct(n["total_return"]), _cls(n["total_return"])),
             ("Gross return", _pct(g["total_return"]), _cls(g["total_return"])),
             ("Net Sharpe", _num(n["sharpe"]), ""), ("Net Sortino", _num(n["sortino"]), ""),
             ("Net Calmar", _num(n["calmar"]), ""), ("Composite", _num(n["composite"]), ""),
             ("Max drawdown", _pct(n["max_drawdown"]), "neg"), ("Trading costs", _money(cst["total_costs"]), ""),
             ("Fills / trading days", f"{ex['fills']} / {ex['trading_days']}", ""),
             ("Avg exposure", _pct(ex["average_exposure"], 0), "")]
    sections.append("<div class=cards>" + "".join(
        f"<div class=card><div class=k>{k}</div><div class='v {c}'>{v}</div></div>" for k, v, c in cards) + "</div>")

    # 1. assumptions
    cm = cfg.cost_model
    observed = [("Market data", cfg.data_source), ("Period", f"{s['period']['start'][:16]} → {s['period']['end'][:16]} UTC"),
                ("Bars", f"{s['period']['bars']:,} ({s['period']['periods_per_year']:.0f} per year)"),
                ("Assets", ", ".join(close.columns))]
    modeled = [("Execution", f"signal at bar close → fill at the {cfg.execution_price} of the bar {cfg.execution_lag} later"),
               ("Fee", f"{cm.fee_rate * 1e4:.1f} bps (taker {cm.taker_fee * 1e4:.1f}, maker {cm.maker_fee * 1e4:.1f}, maker share {cm.maker_fill_probability:.0%})"),
               ("Spread", f"{cm.spread_bps:.1f} bps full spread (half paid per trade)"),
               ("Slippage", f"{cm.slippage_bps:.1f} bps per trade"),
               ("Market impact", f"{cm.impact_coef} × participation^{cm.impact_alpha}" if cm.uses_impact else "off"),
               ("Rebalancing", f"band {cfg.rebalance_threshold:.0%}; exact rebalance at UTC hours {list(cfg.rebalance_hours_utc) or 'none'}"),
               ("Capital", _money(cfg.initial_capital))]
    sections.append("<h2 id=assumptions>Configuration and assumptions</h2><div class=grid2><div>"
                    "<h3>Observed</h3>" + _table(observed, ("Input", "Value"), 99) + "</div><div>"
                    "<h3>Modeled <span class='tag modeled'>assumptions</span></h3>" + _table(modeled, ("Input", "Value"), 99) +
                    "</div></div><h3>Strategy</h3>" + _table(
                        [(k, v) for k, v in s["reproducibility"]["config"]["params"].items()], ("Parameter", "Value"), 99) +
                    (f"<p class=note><b>Hypothesis (researcher's, not evidence):</b> {html.escape(cfg.hypothesis)}</p>" if cfg.hypothesis else ""))

    # 2. gross vs net
    fig = go.Figure()
    fig.add_scatter(x=net.index, y=gross, name="Gross (before costs)", line=dict(color=COLORS["gross"], dash="dash"))
    fig.add_scatter(x=net.index, y=net, name="Net (after costs)", line=dict(color=COLORS["net"], width=2))
    for (name, r), col in zip(bench_results.items(), COLORS["bench"]):
        fig.add_scatter(x=r.portfolio_value.index, y=r.portfolio_value, name=name, line=dict(color=col, width=1), visible="legendonly")
    fig.update_layout(title="Equity: gross vs net (benchmarks toggle in the legend)", yaxis_title="USD")
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06))
    rows = []
    for label, key, fmt in [("Total return", "total_return", _pct), ("CAGR", "cagr", _pct), ("Sharpe", "sharpe", _num),
                            ("Sortino", "sortino", _num), ("Calmar", "calmar", _num), ("Max drawdown", "max_drawdown", _pct),
                            ("Ending equity", "ending_equity", _money), ("Total P&L", "total_pnl", _money)]:
        diff = n[key] - g[key] if np.isfinite(n[key]) and np.isfinite(g[key]) else float("nan")
        rows.append((label, fmt(g[key]), fmt(n[key]), fmt(diff)))
    for label, key, fmt in [("Profit factor (realized trades)", "profit_factor", _num), ("Expectancy per trade", "expectancy", _money)]:
        gv, nv = s["trades_gross"].get(key, float("nan")), tn.get(key, float("nan"))
        rows.append((label, fmt(gv), fmt(nv), fmt(nv - gv) if np.isfinite(gv) and np.isfinite(nv) else "—"))
    cost_rows = [(k.replace("total_", "").capitalize(), _money(cst[f"total_{k2}"]))
                 for k, k2 in [("total_fee", "fee"), ("total_spread", "spread"), ("total_slippage", "slippage"), ("total_impact", "impact")]]
    cost_rows += [("Total costs", _money(cst["total_costs"])), ("Cost drag (% of capital)", _pct(cst["cost_drag_pct_of_capital"])),
                  ("Costs as share of gross P&L", _pct(cst["cost_share_of_gross_pnl"], 1))]
    sections.append("<h2 id=grossnet>Gross vs net P&L</h2><div class=panel>" + _chart(fig, 460) + "</div>"
                    "<div class=grid2><div><h3>Performance</h3>" + _table(rows, ("Metric", "Gross", "Net", "Net − gross")) +
                    "</div><div><h3>Costs <span class='tag modeled'>modeled</span></h3>" + _table(cost_rows, ("Component", "USD")) +
                    "<p class=note>Identity checked by tests: final − initial = Σ gross P&L − Σ costs. Gross = the same trades with no costs deducted.</p></div></div>")

    # 3. drawdowns
    dd = an.drawdown_series(net)
    fig = go.Figure(go.Scatter(x=dd.index, y=dd * 100, fill="tozeroy", line=dict(color=COLORS["neg"], width=1), name="Drawdown"))
    fig.update_layout(title="Underwater curve (net)", yaxis_title="%")
    dsum = s["drawdown"]
    dd_rows = [(str(r.peak)[:16], str(r.trough)[:16], str(r.recovery)[:16] if pd.notna(r.recovery) else "not recovered",
                _pct(-r.depth), _td(r.time_to_trough), _td(r.time_to_recovery)) for r in dd_eps.head(10).itertuples()]
    sections.append("<h2 id=drawdowns>Drawdowns</h2><div class=panel>" + _chart(fig, 300) + "</div>"
                    f"<p class=note>Max drawdown {_pct(dsum['max_drawdown'])}; {dsum['significant_drawdowns']} drawdowns deeper than "
                    f"{_pct(dsum['significant_threshold'], 0)}; longest recovery {_td(dsum['max_recovery_time'])}; median recovery "
                    f"{_td(dsum['median_recovery_time'])}{'; the deepest is still open' if dsum['open_drawdown'] else ''}.</p>"
                    + _table(dd_rows, ("Peak", "Trough", "Recovery", "Depth", "Time to trough", "Time to recovery"), 3))

    # 4. consistency
    mt = an.monthly_table(net)
    fig = go.Figure(go.Heatmap(z=mt.to_numpy() * 100, x=[pd.Timestamp(2000, m, 1).strftime("%b") for m in mt.columns],
                               y=[str(y) for y in mt.index], colorscale="RdYlGn", zmid=0,
                               text=np.vectorize(lambda z: "" if np.isnan(z) else f"{z:.1f}%")(mt.to_numpy() * 100),
                               texttemplate="%{text}", showscale=False))
    fig.update_layout(title="Monthly returns (net)", hovermode="closest")
    fig2 = go.Figure()
    fig2.add_scatter(x=roll.index, y=roll.rolling_sharpe, name=f"Rolling {cfg.rolling_days}d Sharpe", line=dict(color=COLORS["net"]))
    fig2.add_scatter(x=roll.index, y=roll.rolling_volatility, name=f"Rolling {cfg.rolling_days}d volatility", yaxis="y2",
                     line=dict(color=COLORS["bench"][0]))
    fig2.update_layout(title="Rolling risk-adjusted performance", yaxis2=dict(overlaying="y", side="right", tickformat=".0%"))
    co = s["consistency"]
    sections.append("<h2 id=consistency>Consistency</h2><div class=panel>" + _chart(fig, 120 + 45 * max(len(mt), 1)) + "</div>"
                    f"<p class=note>Profitable months {_pct(co['profitable_months'], 0)} of {co['n_months']}; quarters "
                    f"{_pct(co['profitable_quarters'], 0)} of {co['n_quarters']}; years {_pct(co['profitable_years'], 0)} of {co['n_years']} "
                    "(first and last periods may be partial).</p><div class=panel>" + _chart(fig2, 320) + "</div>")

    # 5. exposure
    wh = result.weights_history
    fig = go.Figure()
    for col, color in zip(wh.columns[(wh != 0).any()], [COLORS["net"]] + COLORS["bench"]):
        fig.add_scatter(x=wh.index, y=wh[col] * 100, name=col, stackgroup="w", line=dict(width=0.5, color=color))
    fig.update_layout(title="Portfolio weights over time (rest is cash)", yaxis_title="% of equity")
    ex_rows = [("Average exposure", _pct(ex["average_exposure"], 1)), ("Time in market", _pct(ex["time_in_market"], 1)),
               ("Annual turnover", f"{_num(ex['annual_turnover'], 1)}×"), ("Fills", f"{ex['fills']:,}"),
               ("Days with a fill", f"{ex['trading_days']:,}"), ("Max participation of bar volume", _pct(ex["max_participation"], 4))]
    sections.append("<h2 id=exposure>Exposure and turnover</h2><div class=panel>" + _chart(fig, 320) + "</div>" +
                    _table(ex_rows, ("Measure", "Value")))

    # 6. trades
    if len(trades):
        fig = go.Figure(go.Histogram(x=trades.net_pnl, nbinsx=60, marker_color=COLORS["net"], name="Net P&L"))
        fig.update_layout(title="Realized trade net P&L distribution (FIFO lots)", xaxis_title="USD", hovermode="closest")
        t_rows = [("Trades (FIFO-matched sales)", f"{tn['trades']:,}"), ("Win rate", _pct(tn["win_rate"], 1)),
                  ("Average win / loss", f"{_money(tn['average_win'])} / {_money(tn['average_loss'])}"),
                  ("Largest win / loss", f"{_money(tn['largest_win'])} / {_money(tn['largest_loss'])}"),
                  ("Expectancy", _money(tn["expectancy"])), ("Profit factor", _num(tn["profit_factor"])),
                  ("Payoff ratio", _num(tn["payoff_ratio"])), ("Median holding time", _td(tn["median_holding_time"])),
                  ("Max consecutive wins / losses", f"{tn['max_consecutive_wins']} / {tn['max_consecutive_losses']}")]
        conc = [(f"Top {k}", _pct(tn[f"top_{k}_share_of_realized_pnl"], 1), _money(tn[f"realized_pnl_without_top_{k}"]),
                 _money(tn[f"worst_{k}_sum"])) for k in (1, 5, 10)]
        cols = ["symbol", "entry_time", "exit_time", "quantity", "entry_price", "exit_price", "gross_pnl", "costs", "net_pnl", "holding_time"]

        def trade_rows(df):
            return [(r.symbol, str(r.entry_time)[:16], str(r.exit_time)[:16], f"{r.quantity:.5f}", _num(r.entry_price),
                     _num(r.exit_price), _money(r.gross_pnl), _money(r.costs), _money(r.net_pnl), _td(r.holding_time))
                    for r in df[cols].itertuples()]
        open_rows = []
        if len(open_pos):
            agg = open_pos.assign(cost_basis=open_pos.quantity * open_pos.entry_price).groupby("symbol").agg(
                lots=("quantity", "size"), quantity=("quantity", "sum"), cost_basis=("cost_basis", "sum"),
                last_price=("last_price", "last"), unrealized=("unrealized_gross_pnl", "sum"), first_entry=("entry_time", "min"))
            open_rows = [(sym, f"{r.lots}", str(r.first_entry)[:16], f"{r.quantity:.5f}", _num(r.cost_basis / r.quantity),
                          _num(r.last_price), _money(r.unrealized)) for sym, r in agg.iterrows()]
        sections.append("<h2 id=trades>Trades</h2><p class=note>The strategy rebalances continuously, so trades are FIFO-matched "
                        "sales against earlier buys (buy-leg costs allocated to the quantity sold). Full list: <code>trades.csv</code>, "
                        "fills: <code>fills.csv</code>.</p><div class=grid2><div>" + _table(t_rows, ("Statistic", "Value")) +
                        "</div><div><h3>Concentration</h3>" + _table(conc, ("Trades", "Share of realized P&L", "P&L without them", "Worst-N sum")) +
                        "</div></div><div class=panel>" + _chart(fig, 300) + "</div>"
                        "<h3>Best 10</h3>" + _table(trade_rows(trades.nlargest(10, "net_pnl")), cols, 3) +
                        "<h3>Worst 10</h3>" + _table(trade_rows(trades.nsmallest(10, "net_pnl")), cols, 3) +
                        ("<h3>Open at the end (unrealized, per asset; lots in <code>open_positions.csv</code>)</h3>" +
                         _table(open_rows, ("Symbol", "Open lots", "Oldest entry", "Quantity", "Avg entry price", "Last price", "Unrealized gross P&L"), 1)
                         if open_rows else ""))

    # 7. tail risk
    daily = net.resample("1D").last().dropna().pct_change().dropna()
    fig = go.Figure(go.Histogram(x=daily * 100, nbinsx=50, marker_color=COLORS["net"]))
    fig.update_layout(title="Daily return distribution (net)", xaxis_title="%", hovermode="closest")
    tail = [("VaR 95% (daily)", _pct(n["var_95_daily"])), ("CVaR 95% (daily)", _pct(n["cvar_95_daily"])),
            ("VaR 99% (daily)", _pct(n["var_99_daily"])), ("CVaR 99% (daily)", _pct(n["cvar_99_daily"])),
            ("Worst day", _pct(n["worst_day"])), ("Worst week", _pct(n["worst_week"])), ("Best day", _pct(n["best_day"])),
            ("Skewness (bar returns)", _num(n["skewness"])), ("Excess kurtosis", _num(n["excess_kurtosis"])),
            ("Omega (threshold 0)", _num(n["omega"]))]
    sections.append("<h2 id=tail>Tail risk</h2><div class=grid2><div>" + _table(tail, ("Measure", "Value")) +
                    "<p class=note>Historical (empirical) VaR/CVaR on daily net returns, shown as losses.</p></div><div class=panel>" +
                    _chart(fig, 300) + "</div></div>")

    # 8. competition windows
    ws = s["windows"]
    if ws:
        fig = go.Figure()
        fig.add_histogram(x=win_s["return"] * 100, name="Strategy", opacity=0.65, marker_color=COLORS["net"], nbinsx=40)
        fig.add_histogram(x=win_b["return"] * 100, name=bench_name, opacity=0.5, marker_color=COLORS["bench"][0], nbinsx=40)
        fig.update_layout(barmode="overlay", title=f"{cfg.window_days}-day returns starting from cash, every day", xaxis_title="%", hovermode="closest")
        wrows = [(label, f"{d['windows']}", _pct(d["mean"]), _pct(d["median"]), _pct(d["p_positive"], 0), _pct(d["p10"]),
                  _pct(d["worst"]), _pct(d["worst_drawdown"]), _num(d["median_trading_days"], 0), _pct(d["p_trading_days_ge_8"], 0))
                 for label, d in (("Strategy", ws["strategy"]), (ws["benchmark_name"], ws["benchmark"]))]
        sections.append(f"<h2 id=windows>{cfg.window_days}-day windows (the competition horizon)</h2><div class=panel>" + _chart(fig, 320) + "</div>" +
                        _table(wrows, ("", "Windows", "Mean", "Median", "P(>0)", "10th pct", "Worst", "Worst DD", "Median trading days", "≥ 8 days")) +
                        "<p class=note>Windows overlap (step 1 day), so they are not independent observations.</p>")

    # 9. benchmarks + random entry
    b_rows = [("Strategy (net)",) + tuple(f(n[k]) for k, f in _BENCH_COLS)]
    b_rows += [(name,) + tuple(f(st[k]) for k, f in _BENCH_COLS) for name, st in s["benchmarks"].items()]
    b_rows += [("Cash", "0.00%", "0.00%", "0.00%", "0.00", "0.00", "0.00%", "—")]
    re_ = s["random_entry"]
    rand = ""
    if len(random_df):
        fig = go.Figure(go.Histogram(x=random_df.total_return * 100, nbinsx=25, marker_color=COLORS["gross"], name="Random-entry seeds"))
        fig.add_vline(x=n["total_return"] * 100, line=dict(color=COLORS["net"], width=3), annotation_text="strategy")
        fig.update_layout(title=f"Random-entry baseline: net return over {len(random_df)} seeds", xaxis_title="%", hovermode="closest")
        rand = ("<h3>Random-entry baseline</h3><p class=note>Same assets, same per-asset exposure levels and switching rate as the "
                f"strategy, random timing. The strategy's net return beats {_pct(re_['strategy_return_percentile'], 0)} of seeds and its "
                f"Sharpe beats {_pct(re_['strategy_sharpe_percentile'], 0)} (median random return {_pct(re_['median_random_return'])}).</p>"
                "<div class=panel>" + _chart(fig, 300) + "</div>")
    sections.append("<h2 id=benchmarks>Benchmarks and baselines</h2><p class=note>Same bars, same engine and cost model; "
                    "buy-and-hold is bought once and never rebalanced.</p>" +
                    _table(b_rows, ("", "Return", "CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar")) + rand)

    # 10. statistical context + limitations + reproducibility
    ctx = s["statistical_context"]
    sections.append("<h2 id=stats>Statistical context</h2>" + _table([
        ("Bars", f"{ctx['bars']:,}"), ("Days", _num(ctx["days"], 0)),
        (f"Non-overlapping {cfg.window_days}-day periods", _num(ctx["independent_14d_periods"], 1)),
        ("Realized trades", f"{tn.get('trades', 0):,}"),
        ("Sharpe (net)", _num(ctx["sharpe"])), ("Sharpe standard error (iid approx.)", _num(ctx["sharpe_standard_error"])),
        ("Sharpe 95% interval", f"{_num(ctx['sharpe_95ci_low'])} to {_num(ctx['sharpe_95ci_high'])}")], ("Measure", "Value")) +
        "<p class=note>The Sharpe interval assumes independent returns, so it's optimistic for autocorrelated strategies. "
        "Judge significance from the sample size shown; no trade count makes a backtest proof of a future edge.</p>")

    limitations = [
        "Prices are Binance spot (USDT-quoted) klines, not Roostoo's own prices; the Roostoo–Binance gap hasn't been measured on recorded tickers.",
        "OHLC(V) bars only: no quotes, order book or intrabar sequence. Spread, slippage and market impact are modeled assumptions, not observations.",
        "Fills are exact fractional quantities: exchange precision and Roostoo's minimum order size ($1) aren't applied (the live bot applies them).",
        "The live bot's 12:00 UTC no-trade fallback rebalance and 0.5% cash buffer aren't simulated.",
        "Universe: pairs listed on Roostoo today. Delisted coins are absent (survivorship bias) — minor for a BTC/ETH strategy.",
        f"Limited history: {_num(ctx['days'], 0)} days, about {_num(ctx['independent_14d_periods'], 0)} independent {cfg.window_days}-day periods.",
        "Parameters were chosen by searching grids on the train split; results on data used for selection are optimistic.",
        "Composite scores over long periods are inflated by annualization (Calmar especially); compare rows, not absolute values.",
    ]
    rep = s["reproducibility"]
    sections.append("<h2 id=limits>Limitations</h2><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in limitations) + "</ul>"
                    "<h2 id=repro>Reproducibility</h2>" + _table([
                        ("Generated", rep["generated_at"][:19] + " UTC"), ("Git commit", rep["git_commit"]),
                        ("Data hash (close / open)", f"{rep['data_hash_close']} / {rep['data_hash_open']}"),
                        ("Python / pandas / numpy / plotly", f"{rep['python']} / {rep['pandas']} / {rep['numpy']} / {rep['plotly']}"),
                        ("Random seeds", ", ".join(map(str, cfg.random_seeds)))], ("Item", "Value"), 99) +
                    "<p class=note>Full configuration: <code>config.json</code>; all numbers: <code>summary.json</code>.</p>")

    nav = "".join(f"<a href='#{a}'>{t}</a>" for a, t in [("assumptions", "Assumptions"), ("grossnet", "Gross vs net"), ("drawdowns", "Drawdowns"),
                                                         ("consistency", "Consistency"), ("exposure", "Exposure"), ("trades", "Trades"),
                                                         ("tail", "Tail risk"), ("windows", "14-day windows"), ("benchmarks", "Benchmarks"),
                                                         ("stats", "Statistics"), ("limits", "Limitations"), ("repro", "Reproducibility")])
    title = f"{cfg.strategy_name} backtest"
    return ("<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style><script>{plotly.offline.get_plotlyjs()}</script></head><body><main>"
            f"<h1>{html.escape(title)}</h1><p class=sub>{html.escape(cfg.period_label)} · {s['period']['start'][:10]} → {s['period']['end'][:10]} · "
            f"costs and fills are modeled; see Assumptions and Limitations</p><nav>{nav}</nav>" + "".join(sections) + "</main></body></html>")


_BENCH_COLS = [("total_return", _pct), ("cagr", _pct), ("annualized_volatility", _pct), ("sharpe", _num),
               ("sortino", _num), ("max_drawdown", _pct), ("calmar", _num)]

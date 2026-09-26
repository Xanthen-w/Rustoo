"""Robustness report: runs every analysis in backtest/robustness.py for one
strategy configuration and writes a self-contained HTML page + CSV/JSON."""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go

from backtest import robustness as rb
from backtest.benchmarks import BENCHMARK_BAND, buy_and_hold_weights
from backtest.report import COLORS, CSS, _chart, _git_commit, _hash_frame, _money, _num, _pct, _table


def _benchmark_run(spec: rb.RunSpec, asset: str):
    w = buy_and_hold_weights(spec.close, [asset])
    return spec.run(weights=w, engine_overrides={"rebalance_threshold": BENCHMARK_BAND, "rebalance_hours_utc": ()})


def run_robustness(
    spec: rb.RunSpec,
    out_dir: Path,
    *,
    title: str,
    period_label: str,
    benchmark_asset: str = "BTC/USD",
    cost_grid_bps: dict | None = None,
    mc: rb.MonteCarloConfig = rb.MonteCarloConfig(),
    regime_cfg: rb.RegimeConfig = rb.RegimeConfig(),
    landscapes: dict | None = None,  # label -> (spec_for_split, [(x, xs, y, ys), ...])
    random_entry: dict | None = None,  # split label -> spec
    random_seeds: list[int] = list(range(30)),
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ppy = spec.ppy
    cost_grid_bps = cost_grid_bps or {"fee_bps": [0, 5, 10, 20, 50, 100], "slippage_bps": [0, 5, 10, 20, 50, 100],
                                      "spread_bps": [0, 5, 10, 20, 50, 100]}

    base = spec.run()
    bench = _benchmark_run(spec, benchmark_asset)

    sens = pd.concat([rb.cost_sensitivity(spec, knob, vals) for knob, vals in cost_grid_bps.items()], ignore_index=True)
    breakevens = {knob: rb.breakeven(sens[sens.knob == knob]) for knob in cost_grid_bps}
    stress = rb.stress_test(spec)
    mcres = rb.monte_carlo(base, ppy, mc, benchmark_equity=bench.portfolio_value)
    labels = rb.classify_regimes(spec.close[benchmark_asset].ffill(), ppy, regime_cfg).loc[spec.eval_index]
    regimes = rb.regime_breakdown(base, bench.portfolio_value, labels, ppy)

    land_out = {}
    for label, (lspec, grids) in (landscapes or {}).items():
        for x, xs, y, ys in grids:
            grid = rb.parameter_landscape(lspec, x, xs, y, ys)
            key = f"{label}: {x} × {y}"
            land_out[key] = {"grid": grid, "x": x, "y": y, "current": (lspec.params.get(x), lspec.params.get(y)),
                             "plateau": [rb.plateau_score(grid, x, y, m) for m in ("window_mean_return", "sharpe")]}
            grid.to_csv(out_dir / f"landscape_{label}_{x}_{y}.csv", index=False)

    rand_out = {label: rb.random_entry_test(s, random_seeds) for label, s in (random_entry or {}).items()}

    worst = stress[~stress.synthetic].sort_values("net_return").iloc[0]
    summary = {
        "title": title, "period": period_label,
        "base": rb._row(base, ppy), "benchmark": {"asset": benchmark_asset, **rb._row(bench, ppy)},
        "breakeven_bps": breakevens,
        "worst_stress": {"scenario": worst.scenario, "net_return": float(worst.net_return), "sharpe": float(worst.sharpe)},
        "monte_carlo": mcres["summary"],
        "random_entry": {k: {"return_percentile": v["return_percentile"], "sharpe_percentile": v["sharpe_percentile"],
                             "composite_percentile": v["composite_percentile"], "seeds": len(random_seeds)}
                         for k, v in rand_out.items()},
        "landscape_plateaus": {k: v["plateau"] for k, v in land_out.items()},
        "reproducibility": {"generated_at": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit(),
                            "data_hash_close": _hash_frame(spec.close.loc[spec.eval_index]),
                            "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.params.items()},
                            "engine": {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.engine_kwargs.items()},
                            "costs": spec.cost_model.__dict__, "mc_seeds": list(mc.seeds), "random_seeds": random_seeds},
    }
    sens.to_csv(out_dir / "cost_sensitivity.csv", index=False)
    stress.to_csv(out_dir / "stress.csv", index=False)
    mcres["per_seed"].to_csv(out_dir / "monte_carlo_per_seed.csv", index=False)
    mcres["simulations"].to_csv(out_dir / "monte_carlo_simulations.csv", index=False)
    regimes.to_csv(out_dir / "regimes.csv", index=False)
    for k, v in rand_out.items():
        v["random"].to_csv(out_dir / f"random_entry_{k}.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_dir / "robustness.html").write_text(_render(summary, sens, stress, mcres, regimes, land_out, rand_out, mc,
                                                     regime_cfg, title, period_label))
    return summary


def _breakeven_text(v) -> str:
    if v is None:
        return "not reached in the swept range"
    if isinstance(v, float) and np.isnan(v):
        return "n/a: negative even at zero cost"
    return f"{v:.0f} bps"


def _render(s, sens, stress, mcres, regimes, land, rand, mc, regime_cfg, title, period_label) -> str:
    sec = []
    b, mcs = s["base"], s["monte_carlo"]
    be = s["breakeven_bps"]
    cards = [("Base net return", _pct(b["net_return"])), ("Worst stress return", _pct(s["worst_stress"]["net_return"])),
             ("Fee breakeven", _breakeven_text(be["fee_bps"])),
             (f"P(loss) over {mc.horizon_days:.0f}d", _pct(mcs["p_loss"], 0)),
             (f"5th pct {mc.horizon_days:.0f}d return", _pct(mcs["percentiles"]["p5"]["total_return"])),
             (f"P(beat {s['benchmark']['asset'].split('/')[0]}) over {mc.horizon_days:.0f}d", _pct(mcs.get("p_beats_benchmark"), 0))]
    for k, v in s["random_entry"].items():
        cards.append((f"Beats random timing ({k})", _pct(v["return_percentile"], 0)))
    sec.append("<div class=cards>" + "".join(f"<div class=card><div class=k>{k}</div><div class=v>{v}</div></div>" for k, v in cards) + "</div>")

    # cost sensitivity
    figs = []
    for metric, label in (("net_return", "Net return"), ("sharpe", "Net Sharpe")):
        fig = go.Figure()
        for knob, col in zip(sens.knob.unique(), [COLORS["net"]] + COLORS["bench"]):
            t = sens[sens.knob == knob]
            fig.add_scatter(x=t.bps, y=t[metric] * (100 if metric == "net_return" else 1), name=knob.replace("_bps", ""),
                            mode="lines+markers", line=dict(color=col))
        fig.update_layout(title=f"{label} vs cost (one knob varied, others at base)", xaxis_title="bps",
                          yaxis_title="%" if metric == "net_return" else "", hovermode="x")
        figs.append(_chart(fig, 320))
    rows = [(r.knob.replace("_bps", ""), f"{r.bps:g}", _pct(r.net_return), _num(r.sharpe), _pct(r.max_drawdown), _money(r.total_costs))
            for r in sens.itertuples()]
    be_txt = "; ".join(f"{k.replace('_bps', '')}: {_breakeven_text(v)}" for k, v in be.items())
    sec.append("<h2 id=costs>Cost sensitivity <span class='tag modeled'>modeled</span></h2><div class=grid2><div class=panel>" + figs[0] +
               "</div><div class=panel>" + figs[1] + f"</div></div><p class=note>Breakeven (net return reaches 0): {be_txt}. "
               "Fee here is the taker fee (maker set to half).</p>" +
               _table(rows, ("Knob", "bps", "Net return", "Sharpe", "Max DD", "Costs"), 1))

    # stress
    st = stress.sort_values("net_return")
    fig = go.Figure(go.Bar(y=st.scenario, x=st.net_return * 100, orientation="h",
                           marker_color=[COLORS["neg"] if v < 0 else COLORS["pos"] for v in st.net_return]))
    fig.update_layout(title="Net return by scenario", xaxis_title="%", hovermode="closest", margin=dict(l=380))
    rows = [(r.scenario + (" (synthetic prices)" if r.synthetic else ""), _pct(r.net_return), _pct(r.net_return_vs_base),
             _num(r.sharpe), _pct(r.max_drawdown), _money(r.total_costs)) for r in st.itertuples()]
    w = s["worst_stress"]
    sec.append("<h2 id=stress>Stress tests <span class='tag modeled'>modeled</span></h2><div class=panel>" + _chart(fig, 120 + 32 * len(st)) +
               "</div>" + _table(rows, ("Scenario", "Net return", "vs base", "Sharpe", "Max DD", "Costs")) +
               f"<p class=note>Worst non-synthetic scenario: <b>{html.escape(w['scenario'])}</b> ({_pct(w['net_return'])}). "
               "These are adverse but plausible assumptions, not the theoretical worst case.</p>")

    # Monte Carlo
    fan = mcres["fan"]
    fig = go.Figure()
    fig.add_scatter(x=fan.index, y=(fan.p95 - 1) * 100, line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=fan.index, y=(fan.p5 - 1) * 100, fill="tonexty", fillcolor="rgba(37,99,235,0.12)", line=dict(width=0), name="5–95%")
    fig.add_scatter(x=fan.index, y=(fan.p75 - 1) * 100, line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=fan.index, y=(fan.p25 - 1) * 100, fill="tonexty", fillcolor="rgba(37,99,235,0.28)", line=dict(width=0), name="25–75%")
    fig.add_scatter(x=fan.index, y=(fan.p50 - 1) * 100, line=dict(color=COLORS["net"], width=2), name="median")
    fig.update_layout(title=f"Bootstrapped {mc.horizon_days:.0f}-day paths (net)", xaxis_title="days", yaxis_title="return %")
    sims = mcres["simulations"]
    fig2 = go.Figure()
    fig2.add_histogram(x=sims.total_return * 100, name="Strategy", opacity=0.65, marker_color=COLORS["net"], nbinsx=60)
    if "benchmark_return" in sims:
        fig2.add_histogram(x=sims.benchmark_return * 100, name=s["benchmark"]["asset"], opacity=0.5, marker_color=COLORS["bench"][0], nbinsx=60)
    fig2.update_layout(barmode="overlay", title=f"{mc.horizon_days:.0f}-day return distribution (paired blocks)", xaxis_title="%", hovermode="closest")
    prow = [(p, _pct(v["total_return"]), _pct(v["max_drawdown"]), _num(v["sharpe"])) for p, v in mcs["percentiles"].items()]
    dd_key = next(k for k in mcs if k.startswith("p_drawdown_worse_than"))
    seed_rows = [tuple([str(r["seed"])] + [(_pct(v, 1) if isinstance(v, float) else v) for k, v in r.items() if k != "seed"])
                 for r in mcres["per_seed"].to_dict("records")]
    seed_cols = ("Seed",) + tuple(c.replace("_", " ") for c in mcres["per_seed"].columns if c != "seed")
    sec.append("<h2 id=mc>Monte Carlo <span class='tag modeled'>bootstrap</span></h2>"
               f"<p class=note>{mcs['simulations_per_seed']:,} paths × {len(mcs['seeds'])} seeds, each {mc.horizon_days:.0f} days built from random "
               f"{mc.block_hours:.0f}-hour blocks of this run's own hourly gross returns and costs (blocks keep short-term dependence). "
               f"Cost multiplier per path: lognormal σ={mc.cost_sigma}; return noise {mc.return_noise_bps} bps. Assumes the future resembles this "
               "period — it describes uncertainty *within* the sample, not regime change.</p>"
               "<div class=grid2><div class=panel>" + _chart(fig, 330) + "</div><div class=panel>" + _chart(fig2, 330) + "</div></div>"
               "<div class=grid2><div>" + _table(prow, ("Percentile", "Return", "Max drawdown", "Sharpe")) + "</div><div>" +
               _table([(f"P(loss over {mc.horizon_days:.0f}d)", _pct(mcs["p_loss"], 1)), ("P(return < −10%)", _pct(mcs["p_below_minus_10pct"], 1)),
                       (f"P(drawdown worse than {mc.drawdown_threshold:.0%})", _pct(mcs[dd_key], 1)),
                       (f"P(beats {s['benchmark']['asset']})", _pct(mcs.get("p_beats_benchmark"), 1)),
                       ("Spread of median return across seeds", _pct(mcs["seed_spread_median_return"], 2))], ("Probability", "Value")) +
               "</div></div><h3>By seed</h3>" + _table(seed_rows, seed_cols))

    # regimes
    fig = go.Figure()
    reg = regimes.assign(label=regimes.dimension + ": " + regimes.regime)
    fig.add_bar(x=reg.label, y=reg.strategy_return * 100, name="Strategy", marker_color=COLORS["net"])
    fig.add_bar(x=reg.label, y=reg.benchmark_return * 100, name=s["benchmark"]["asset"], marker_color=COLORS["bench"][0])
    fig.update_layout(barmode="group", title="Compounded return within each regime", yaxis_title="%", hovermode="closest")
    rows = [(r.label, _pct(r.share_of_bars, 0), _pct(r.strategy_return), _pct(r.benchmark_return), _num(r.strategy_sharpe),
             _num(r.benchmark_sharpe), _pct(r.average_exposure, 0), _money(r.costs)) for r in reg.itertuples()]
    sec.append(f"<h2 id=regimes>Market regimes</h2><p class=note>Labels from {s['benchmark']['asset']}'s trailing {regime_cfg.lookback_days:.0f}-day "
               f"return (bull above +{regime_cfg.trend_threshold:.0%}, bear below −{regime_cfg.trend_threshold:.0%}) and trailing volatility "
               "(above/below its median over this period, so the vol split is ex-post).</p><div class=panel>" + _chart(fig, 330) + "</div>" +
               _table(rows, ("Regime", "Share of bars", "Strategy", "Benchmark", "Strategy Sharpe", "Benchmark Sharpe", "Avg exposure", "Costs")))

    # landscapes
    if land:
        parts = []
        for key, v in land.items():
            g, x, y = v["grid"], v["x"], v["y"]
            charts = []
            for metric, label, fmt in (("window_mean_return", "Mean 14-day return", 100), ("sharpe", "Net Sharpe (continuous)", 1),
                                       ("max_drawdown", "Max drawdown", 100)):
                piv = g.pivot(index=y, columns=x, values=metric)
                fig = go.Figure(go.Heatmap(z=piv.to_numpy() * fmt, x=[str(c) for c in piv.columns], y=[str(i) for i in piv.index],
                                           colorscale="RdYlGn", text=np.round(piv.to_numpy() * fmt, 2), texttemplate="%{text}",
                                           showscale=False))
                cx, cy = v["current"]
                if cx in list(piv.columns) and cy in list(piv.index):
                    fig.add_scatter(x=[str(cx)], y=[str(cy)], mode="markers", marker=dict(symbol="square-open", size=34, color="#111", line=dict(width=3)),
                                    name="current setting", hoverinfo="skip")
                fig.update_layout(title=f"{label}", xaxis_title=x, yaxis_title=y, hovermode="closest", showlegend=False)
                charts.append("<div class=panel>" + _chart(fig, 330) + "</div>")
            pl = [(p["metric"], _num(p["best"], 4), json.dumps(p["best_at"], default=str), _num(p["neighbour_mean"], 4), _num(p["grid_median"], 4))
                  for p in v["plateau"]]
            parts.append(f"<h3>{html.escape(key)}</h3><div class=grid2>" + "".join(charts[:2]) + "</div>" + charts[2] +
                         _table(pl, ("Metric", "Best cell", "At", "Neighbour mean", "Grid median"), 1))
        sec.append("<h2 id=landscape>Parameter landscapes</h2><p class=note>The whole grid is shown, not just the best cell; the square marks the live "
                   "setting. A broad plateau (neighbours close to the best) is more trustworthy than a narrow peak. Train is the data the "
                   "parameters were selected on.</p>" + "".join(parts))

    # random entry
    if rand:
        rows = [(k, f"{len(v['random'])}", _pct(v["strategy"]["net_return"]), _pct(v["random"].net_return.median()),
                 _pct(v["return_percentile"], 0), _pct(v["sharpe_percentile"], 0), _pct(v["composite_percentile"], 0))
                for k, v in rand.items()]
        sec.append("<h2 id=random>Timing vs random entry</h2><p class=note>Random timing with the strategy's own per-asset exposure levels and "
                   "switching rate (see backtest/benchmarks.py). A percentile near 50% means the trend signal's timing adds nothing measurable "
                   "beyond position sizing in that period.</p>" +
                   _table(rows, ("Split", "Seeds", "Strategy return", "Median random return", "Beats (return)", "Beats (Sharpe)", "Beats (composite)")))

    rep = s["reproducibility"]
    sec.append("<h2 id=repro>Reproducibility</h2>" + _table([
        ("Generated", rep["generated_at"][:19] + " UTC"), ("Git commit", rep["git_commit"]), ("Data hash", rep["data_hash_close"]),
        ("Monte Carlo seeds", ", ".join(map(str, rep["mc_seeds"]))), ("Random-entry seeds", f"{len(rep['random_seeds'])} (0…{max(rep['random_seeds'])})")],
        ("Item", "Value"), 99) + "<p class=note>All tables are also in the CSV files next to this page; parameters in summary.json.</p>")

    nav = "".join(f"<a href='#{a}'>{t}</a>" for a, t in [("costs", "Cost sensitivity"), ("stress", "Stress"), ("mc", "Monte Carlo"),
                                                         ("regimes", "Regimes"), ("landscape", "Parameters"), ("random", "Random entry"),
                                                         ("repro", "Reproducibility")])
    return ("<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style><script>{plotly.offline.get_plotlyjs()}</script></head><body><main>"
            f"<h1>{html.escape(title)}</h1><p class=sub>{html.escape(period_label)} · stressed and bootstrapped results are modeled scenarios, "
            f"not observations</p><nav>{nav}</nav>" + "".join(sec) + "</main></body></html>")

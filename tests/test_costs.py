from backtest.costs import BASE, CostModel


def test_zero_notional_has_zero_cost():
    assert BASE.trade_cost(0.0) == 0.0


def test_taker_only_cost_matches_fee_plus_slippage():
    model = CostModel(taker_fee=0.0010, maker_fee=0.0005, slippage_bps=5.0, maker_fill_probability=0.0)
    notional = 10_000.0
    expected = notional * 0.0010 + notional * (5.0 / 10_000.0)
    assert abs(model.trade_cost(notional) - expected) < 1e-9


def test_higher_maker_probability_lowers_cost():
    low_maker = CostModel(taker_fee=0.0010, maker_fee=0.0005, slippage_bps=5.0, maker_fill_probability=0.0)
    high_maker = CostModel(taker_fee=0.0010, maker_fee=0.0005, slippage_bps=5.0, maker_fill_probability=1.0)
    notional = 10_000.0
    assert high_maker.trade_cost(notional) < low_maker.trade_cost(notional)


def test_pessimistic_scenario_costs_more_than_optimistic():
    from backtest.costs import OPTIMISTIC, PESSIMISTIC

    notional = 10_000.0
    assert PESSIMISTIC.trade_cost(notional) > OPTIMISTIC.trade_cost(notional)

import math

import pytest

from src.data.market_data import Ticker, tickers_from_response


def test_omitted_zero_bid_is_parsed_as_empty_book_side():
    # Roostoo omits zero-valued fields entirely (docs/API_NOTES.md).
    t = Ticker.from_roostoo("A/USD", {"LastPrice": 1.0, "MinAsk": 1.1}, 0)
    assert t.bid == 0.0
    assert t.ask == 1.1
    assert t.has_two_sided_quote is False
    assert math.isnan(t.mid)


def test_two_sided_quote_mid():
    t = Ticker.from_roostoo("A/USD", {"LastPrice": 1.0, "MaxBid": 0.9, "MinAsk": 1.1}, 0)
    assert t.mid == pytest.approx(1.0)


def test_pair_without_last_price_is_skipped_not_fatal():
    response = {
        "ServerTime": 1_700_000_000_000,
        "Data": {
            "A/USD": {"LastPrice": 1.0, "MaxBid": 0.9, "MinAsk": 1.1},
            "B/USD": {"MaxBid": 0.9},
        },
    }
    tickers = tickers_from_response(response)
    assert list(tickers) == ["A/USD"]

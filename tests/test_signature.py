"""Verifies our HMAC signing against the exact worked example published in
the official Roostoo API docs (see docs/API_NOTES.md), so we know signing is
correct without needing live credentials.
"""
from src.execution.client import build_total_params, sign

SECRET_KEY = "S1XP1e3UZj6A7H5fATj0jNhqPxxdSJYdInClVN65XAbvqqMKjVHjA7PZj4W12oep"

PAYLOAD = {
    "timestamp": "1580774512000",
    "pair": "BNB/USD",
    "quantity": "2000",
    "side": "BUY",
    "type": "MARKET",
}

EXPECTED_TOTAL_PARAMS = "pair=BNB/USD&quantity=2000&side=BUY&timestamp=1580774512000&type=MARKET"
EXPECTED_SIGNATURE = "20b7fd5550b67b3bf0c1684ed0f04885261db8fdabd38611e9e6af23c19b7fff"


def test_total_params_matches_doc_example():
    assert build_total_params(PAYLOAD) == EXPECTED_TOTAL_PARAMS


def test_signature_matches_doc_example():
    total_params = build_total_params(PAYLOAD)
    signature = sign(SECRET_KEY, total_params)
    assert signature == EXPECTED_SIGNATURE

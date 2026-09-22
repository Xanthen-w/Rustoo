"""Transaction cost model (Phase 7). Configurable maker/taker fee + slippage,
with optimistic/base/pessimistic presets for sensitivity analysis.

The live client never uses this module for accounting its own fills — it
reads the real `CommissionChargeValue` from Roostoo's order response. This
model exists purely so backtests don't overstate edge.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    taker_fee: float = 0.0010
    maker_fee: float = 0.0005
    slippage_bps: float = 5.0
    maker_fill_probability: float = 0.0  # backtests can't know maker/taker in advance

    def trade_cost(self, notional: float) -> float:
        """Total cost in currency units for a trade of the given absolute
        notional value, blending maker/taker by `maker_fill_probability` and
        adding slippage on top."""
        if notional <= 0:
            return 0.0
        blended_fee = (
            self.maker_fill_probability * self.maker_fee
            + (1 - self.maker_fill_probability) * self.taker_fee
        )
        slippage_cost = notional * (self.slippage_bps / 10_000.0)
        return notional * blended_fee + slippage_cost


# Scenario presets for Phase 7's required sensitivity analysis. "base" mirrors
# the hackathon-stated fee schedule (taker 0.10% / maker 0.05%).
OPTIMISTIC = CostModel(taker_fee=0.0008, maker_fee=0.0003, slippage_bps=2.0, maker_fill_probability=0.5)
BASE = CostModel(taker_fee=0.0010, maker_fee=0.0005, slippage_bps=5.0, maker_fill_probability=0.0)
PESSIMISTIC = CostModel(taker_fee=0.0015, maker_fee=0.0008, slippage_bps=15.0, maker_fill_probability=0.0)

SCENARIOS = {"optimistic": OPTIMISTIC, "base": BASE, "pessimistic": PESSIMISTIC}

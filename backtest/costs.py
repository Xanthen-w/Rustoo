"""Transaction cost model (Phase 7).

Every cost is charged as a fraction of traded notional and reported
separately, so gross and net P&L can be reconciled component by component:

- fee:      exchange commission, maker/taker blended by `maker_fill_probability`
            (Roostoo decides the role, so backtests can only assume a mix)
- spread:   half of a *modeled* bid/ask spread (`spread_bps` is the full
            spread). OHLC data has no quotes, so this is an assumption.
- slippage: fixed `slippage_bps` per trade — a *modeled* allowance for fills
            worse than the reference price.
- impact:   *modeled* market impact, `impact_coef * participation ** impact_alpha`,
            where participation = trade notional / bar traded notional. Needs
            volume data; never fabricated when volume is missing.

The live client never uses this module for its own fills — it reads the real
`CommissionChargeValue` from Roostoo's order response.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COST_COMPONENTS = ("fee", "spread", "slippage", "impact")


@dataclass(frozen=True)
class CostModel:
    taker_fee: float = 0.0010
    maker_fee: float = 0.0005
    slippage_bps: float = 5.0
    maker_fill_probability: float = 0.0  # backtests can't know maker/taker in advance
    spread_bps: float = 0.0  # full modeled spread; half is paid per trade
    impact_coef: float = 0.0
    impact_alpha: float = 0.5

    def __post_init__(self):
        if not 0.0 <= self.maker_fill_probability <= 1.0:
            raise ValueError("maker_fill_probability must be in [0, 1]")
        if min(self.taker_fee, self.maker_fee, self.slippage_bps, self.spread_bps, self.impact_coef) < 0:
            raise ValueError("costs must be non-negative")
        if self.impact_alpha <= 0:
            raise ValueError("impact_alpha must be positive")

    @property
    def fee_rate(self) -> float:
        return (self.maker_fill_probability * self.maker_fee
                + (1 - self.maker_fill_probability) * self.taker_fee)

    @property
    def spread_rate(self) -> float:
        return self.spread_bps / 2 / 10_000.0

    @property
    def slippage_rate(self) -> float:
        return self.slippage_bps / 10_000.0

    @property
    def cost_rate(self) -> float:
        """Cost per unit of traded notional excluding market impact (which
        depends on trade size)."""
        return self.fee_rate + self.spread_rate + self.slippage_rate

    @property
    def uses_impact(self) -> bool:
        return self.impact_coef > 0

    def impact_rate(self, participation) -> np.ndarray:
        """Impact as a fraction of notional for the given participation
        rate(s). NaN participation (no volume) -> 0 only if impact is off."""
        participation = np.asarray(participation, dtype=float)
        if not self.uses_impact:
            return np.zeros_like(participation)
        return self.impact_coef * np.power(np.clip(participation, 0.0, None), self.impact_alpha)

    def trade_cost(self, notional: float) -> float:
        """Total cost (excluding impact) for a trade of the given absolute
        notional value."""
        if notional <= 0:
            return 0.0
        return notional * self.cost_rate

    def scaled(self, fee_mult: float = 1.0, slippage_mult: float = 1.0, spread_mult: float = 1.0,
               impact_mult: float = 1.0) -> "CostModel":
        """A stressed copy (used by sensitivity and stress tests)."""
        return CostModel(
            taker_fee=self.taker_fee * fee_mult, maker_fee=self.maker_fee * fee_mult,
            slippage_bps=self.slippage_bps * slippage_mult, maker_fill_probability=self.maker_fill_probability,
            spread_bps=self.spread_bps * spread_mult, impact_coef=self.impact_coef * impact_mult,
            impact_alpha=self.impact_alpha,
        )


# Scenario presets for Phase 7's required sensitivity analysis. "base" mirrors
# the competition fee schedule (taker 0.10% / maker 0.05%).
OPTIMISTIC = CostModel(taker_fee=0.0008, maker_fee=0.0003, slippage_bps=2.0, maker_fill_probability=0.5)
BASE = CostModel(taker_fee=0.0010, maker_fee=0.0005, slippage_bps=5.0, maker_fill_probability=0.0)
PESSIMISTIC = CostModel(taker_fee=0.0015, maker_fee=0.0008, slippage_bps=15.0, maker_fill_probability=0.0)

SCENARIOS = {"optimistic": OPTIMISTIC, "base": BASE, "pessimistic": PESSIMISTIC}

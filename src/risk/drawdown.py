"""Drawdown state machine (Phase 11): NORMAL -> CAUTION -> DEFENSIVE ->
EMERGENCY, each with an exposure multiplier applied to the strategy's target
weights.

Used both by the backtest engine (BacktestEngine(risk_overlay=...)) and,
later, by the live bot. `update` is fed the portfolio's equity *after* each
bar and returns the multiplier for the *next* bar's targets, so it never
sizes a trade using the price that trade executes at.

Escalation is immediate when drawdown from the high-water mark crosses a
threshold. De-escalation is one level at a time after `cooldown_bars` in the
current state, re-arming the high-water mark at the current equity —
without that, a strategy moved to cash could never recover from EMERGENCY,
since its drawdown can't shrink while it holds no risk.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class RiskState(IntEnum):
    NORMAL = 0
    CAUTION = 1
    DEFENSIVE = 2
    EMERGENCY = 3


@dataclass
class DrawdownRiskManager:
    caution: float = 0.05
    defensive: float = 0.10
    emergency: float = 0.20
    caution_exposure: float = 0.75
    defensive_exposure: float = 0.5
    emergency_exposure: float = 0.0
    cooldown_bars: int = 72

    state: RiskState = field(default=RiskState.NORMAL, init=False)
    peak: float = field(default=0.0, init=False)
    bars_in_state: int = field(default=0, init=False)

    def __post_init__(self):
        if not 0 < self.caution < self.defensive < self.emergency < 1:
            raise ValueError("need 0 < caution < defensive < emergency < 1")
        exposures = (1.0, self.caution_exposure, self.defensive_exposure, self.emergency_exposure)
        if any(not 0.0 <= e <= 1.0 for e in exposures) or list(exposures) != sorted(exposures, reverse=True):
            raise ValueError("exposures must be in [0, 1] and non-increasing with severity")
        if self.cooldown_bars < 1:
            raise ValueError("cooldown_bars must be >= 1")

    def exposure_for(self, state: RiskState) -> float:
        return {
            RiskState.NORMAL: 1.0,
            RiskState.CAUTION: self.caution_exposure,
            RiskState.DEFENSIVE: self.defensive_exposure,
            RiskState.EMERGENCY: self.emergency_exposure,
        }[state]

    def _level(self, drawdown: float) -> RiskState:
        if drawdown >= self.emergency:
            return RiskState.EMERGENCY
        if drawdown >= self.defensive:
            return RiskState.DEFENSIVE
        if drawdown >= self.caution:
            return RiskState.CAUTION
        return RiskState.NORMAL

    def reset(self, equity: float) -> None:
        self.state = RiskState.NORMAL
        self.peak = equity
        self.bars_in_state = 0

    @property
    def exposure(self) -> float:
        return self.exposure_for(self.state)

    def update(self, equity: float) -> float:
        self.peak = max(self.peak, equity)
        drawdown = 1.0 - equity / self.peak if self.peak > 0 else 0.0
        level = self._level(drawdown)
        if level > self.state:
            self.state = level
            self.bars_in_state = 0
        else:
            self.bars_in_state += 1
            if self.state > RiskState.NORMAL and self.bars_in_state >= self.cooldown_bars:
                self.state = RiskState(self.state - 1)
                self.bars_in_state = 0
                self.peak = equity
        return self.exposure

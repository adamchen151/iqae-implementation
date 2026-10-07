"""Default market parameters and global settings for the QAE option-pricing study."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketParams:
    """Black-Scholes-Merton market and contract parameters.

    Attributes:
        S0: Spot price of the underlying asset.
        K: Strike price of the European option.
        r: Continuously-compounded risk-free rate.
        sigma: Annualised volatility.
        T: Time to maturity in years.
    """

    S0: float = 100.0
    K: float = 100.0
    r: float = 0.05
    sigma: float = 0.20
    T: float = 1.0

    @property
    def discount(self) -> float:
        """Discount factor ``exp(-r T)``."""
        import math

        return math.exp(-self.r * self.T)


DEFAULT_PARAMS = MarketParams()
DEFAULT_SEED = 42
RESULTS_DIR = "results"

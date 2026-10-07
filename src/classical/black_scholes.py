"""Analytical Black-Scholes-Merton pricing and Greeks."""
from __future__ import annotations

import math
from typing import Dict

from scipy.stats import norm

from config import MarketParams


def _d1_d2(S0: float, K: float, r: float, sigma: float, T: float) -> tuple[float, float]:
    """Return the ``(d1, d2)`` terms of the Black-Scholes formula."""
    if sigma <= 0 or T <= 0:
        raise ValueError("sigma and T must be strictly positive.")
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    return d1, d1 - sigma * sqrt_t


def call_price(p: MarketParams) -> float:
    """Closed-form price of a European call option.

    Args:
        p: Market parameters.

    Returns:
        The Black-Scholes call price.
    """
    d1, d2 = _d1_d2(p.S0, p.K, p.r, p.sigma, p.T)
    return float(p.S0 * norm.cdf(d1) - p.K * p.discount * norm.cdf(d2))


def put_price(p: MarketParams) -> float:
    """Closed-form price of a European put option (via the same d1/d2 terms)."""
    d1, d2 = _d1_d2(p.S0, p.K, p.r, p.sigma, p.T)
    return float(p.K * p.discount * norm.cdf(-d2) - p.S0 * norm.cdf(-d1))


def call_greeks(p: MarketParams) -> Dict[str, float]:
    """Analytical Greeks of a European call.

    Args:
        p: Market parameters.

    Returns:
        Dict with ``delta, gamma, vega, theta, rho`` (theta per year, vega per
        unit of volatility, rho per unit of rate).
    """
    d1, d2 = _d1_d2(p.S0, p.K, p.r, p.sigma, p.T)
    pdf1 = norm.pdf(d1)
    sqrt_t = math.sqrt(p.T)
    return {
        "delta": float(norm.cdf(d1)),
        "gamma": float(pdf1 / (p.S0 * p.sigma * sqrt_t)),
        "vega": float(p.S0 * pdf1 * sqrt_t),
        "theta": float(-p.S0 * pdf1 * p.sigma / (2 * sqrt_t) - p.r * p.K * p.discount * norm.cdf(d2)),
        "rho": float(p.K * p.T * p.discount * norm.cdf(d2)),
    }

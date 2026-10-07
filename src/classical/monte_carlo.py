"""Vectorised classical Monte Carlo pricing engine with error bounds."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from config import DEFAULT_SEED, MarketParams
from src.classical.black_scholes import call_price

DEFAULT_SAMPLE_SIZES = tuple(int(x) for x in np.logspace(2, 6, 9))  # 1e2 ... 1e6


@dataclass(frozen=True)
class MCResult:
    """Result of one Monte Carlo pricing run.

    Attributes:
        n_samples: Number of simulated paths ``M``.
        price: Discounted sample-mean payoff.
        std_error: Estimated standard error ``sigma_hat / sqrt(M)`` (discounted).
        abs_error: ``|price - analytical|``.
        ci_low: Lower end of the 95% confidence interval.
        ci_high: Upper end of the 95% confidence interval.
    """

    n_samples: int
    price: float
    std_error: float
    abs_error: float
    ci_low: float
    ci_high: float


def simulate_terminal_prices(p: MarketParams, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ``S_T`` exactly from the risk-neutral GBM solution.

    Args:
        p: Market parameters.
        n: Number of samples.
        rng: NumPy random generator.

    Returns:
        Array of shape ``(n,)`` of terminal asset prices.
    """
    z = rng.standard_normal(n)
    drift = (p.r - 0.5 * p.sigma**2) * p.T
    return p.S0 * np.exp(drift + p.sigma * math.sqrt(p.T) * z)


def mc_call_price(p: MarketParams, n: int, rng: Optional[np.random.Generator] = None) -> MCResult:
    """Price a European call by plain Monte Carlo.

    Args:
        p: Market parameters.
        n: Number of samples (``>= 2``).
        rng: Optional generator; defaults to one seeded with ``DEFAULT_SEED``.

    Returns:
        An :class:`MCResult`.
    """
    if n < 2:
        raise ValueError("n must be >= 2.")
    rng = rng or np.random.default_rng(DEFAULT_SEED)
    payoff = p.discount * np.maximum(simulate_terminal_prices(p, n, rng) - p.K, 0.0)
    price = float(payoff.mean())
    se = float(payoff.std(ddof=1) / math.sqrt(n))
    truth = call_price(p)
    return MCResult(n, price, se, abs(price - truth), price - 1.96 * se, price + 1.96 * se)


def mc_convergence(
    p: MarketParams,
    sizes: Sequence[int] = DEFAULT_SAMPLE_SIZES,
    n_repeats: int = 30,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Run MC at several sample sizes, averaging over independent repeats.

    Args:
        p: Market parameters.
        sizes: Sample sizes ``M`` to evaluate.
        n_repeats: Independent repetitions per size (smooths the error curve).
        seed: Base RNG seed.

    Returns:
        DataFrame with columns ``n_samples, price, mean_abs_error, rmse,
        std_error, coverage`` (``coverage`` = fraction of 95% CIs containing
        the analytical price).
    """
    rng = np.random.default_rng(seed)
    truth = call_price(p)
    rows = []
    for m in sizes:
        res = [mc_call_price(p, int(m), rng) for _ in range(n_repeats)]
        errs = np.array([r.abs_error for r in res])
        rows.append(
            {
                "n_samples": int(m),
                "price": float(np.mean([r.price for r in res])),
                "mean_abs_error": float(errs.mean()),
                "rmse": float(np.sqrt((errs**2).mean())),
                "std_error": float(np.mean([r.std_error for r in res])),
                "coverage": float(np.mean([r.ci_low <= truth <= r.ci_high for r in res])),
            }
        )
    return pd.DataFrame(rows)

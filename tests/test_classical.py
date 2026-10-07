"""Tests: analytical Black-Scholes and classical Monte Carlo."""
import math

import numpy as np
import pytest

from config import DEFAULT_PARAMS, MarketParams
from src.classical.black_scholes import call_greeks, call_price, put_price
from src.classical.monte_carlo import mc_call_price, mc_convergence


def test_bs_reference_value() -> None:
    assert call_price(DEFAULT_PARAMS) == pytest.approx(10.4506, abs=1e-3)


def test_put_call_parity() -> None:
    p = MarketParams(S0=105, K=95, r=0.03, sigma=0.3, T=0.5)
    lhs = call_price(p) - put_price(p)
    assert lhs == pytest.approx(p.S0 - p.K * math.exp(-p.r * p.T), abs=1e-10)


def test_greeks_finite_difference() -> None:
    p, h = DEFAULT_PARAMS, 1e-3
    up = MarketParams(p.S0 + h, p.K, p.r, p.sigma, p.T)
    dn = MarketParams(p.S0 - h, p.K, p.r, p.sigma, p.T)
    g = call_greeks(p)
    assert g["delta"] == pytest.approx((call_price(up) - call_price(dn)) / (2 * h), abs=1e-5)
    assert g["gamma"] == pytest.approx((call_price(up) - 2 * call_price(p) + call_price(dn)) / h**2, abs=1e-3)


def test_mc_within_5_percent_atm() -> None:
    res = mc_call_price(DEFAULT_PARAMS, 200_000, np.random.default_rng(1))
    assert abs(res.price - call_price(DEFAULT_PARAMS)) / call_price(DEFAULT_PARAMS) < 0.05
    assert res.ci_low < call_price(DEFAULT_PARAMS) < res.ci_high


def test_mc_error_scales_like_inverse_sqrt() -> None:
    df = mc_convergence(DEFAULT_PARAMS, sizes=[100, 10_000, 1_000_000], n_repeats=30, seed=3)
    slope = np.polyfit(np.log(df["n_samples"]), np.log(df["mean_abs_error"]), 1)[0]
    assert -0.65 < slope < -0.35

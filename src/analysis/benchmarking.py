"""Convergence, query-complexity and circuit-resource benchmarks."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from config import DEFAULT_SEED, MarketParams
from src.classical.monte_carlo import DEFAULT_SAMPLE_SIZES, mc_convergence
from src.quantum.qae_pricer import QAEPricer, default_sampler


def fit_loglog_slope(x: Sequence[float], y: Sequence[float]) -> float:
    """Least-squares slope of ``log y`` against ``log x``."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])


def classical_convergence(params: MarketParams, sizes: Sequence[int] = DEFAULT_SAMPLE_SIZES,
                          n_repeats: int = 30, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Monte Carlo error vs. samples (effort = number of samples)."""
    df = mc_convergence(params, sizes, n_repeats, seed)
    df["effort"] = df["n_samples"]
    return df


def quantum_convergence(
    params: MarketParams,
    n_qubits: int = 4,
    epsilons: Sequence[float] = (0.008, 0.004, 0.002, 0.001, 0.0005),
    n_repeats: int = 3,
    shots: int = 100,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """IQAE error vs. oracle queries (effort = applications of ``A``).

    Error is measured against the exact price implied by the circuit, so that
    the (fixed) discretisation and ``c``-approximation biases do not create an
    artificial error floor; the bias w.r.t. Black-Scholes is reported separately.

    Returns:
        DataFrame with ``epsilon, effort, mean_abs_error, bs_error``.
    """
    from src.classical.black_scholes import call_price

    truth_bs = call_price(params)
    rows = []
    for eps in epsilons:
        errs, qs, bs_errs = [], [], []
        pricer0 = QAEPricer(params, n_qubits)
        target = pricer0.exact_circuit_price()
        for rep in range(n_repeats):
            pricer = QAEPricer(params, n_qubits, sampler=default_sampler(shots, seed + rep))
            res = pricer.price(epsilon_target=eps)
            errs.append(abs(res.price - target))
            bs_errs.append(abs(res.price - truth_bs))
            qs.append(res.num_oracle_queries)
        rows.append({"epsilon": eps, "effort": float(np.mean(qs)), "mean_abs_error": float(np.mean(errs)),
                     "bs_error": float(np.mean(bs_errs))})
    return pd.DataFrame(rows)


def circuit_profile(params: MarketParams, qubit_range: Sequence[int] = (3, 4, 5, 6)) -> pd.DataFrame:
    """Resource counts after transpiling to ``{cx, u}`` for the operator ``A`` and Grover ``Q``.

    Returns:
        DataFrame with ``n_qubits, total_qubits, A_depth, A_cx, A_gates, Q_depth, Q_cx, Q_gates``.
    """
    from qiskit import transpile

    rows = []
    for n in qubit_range:
        pricer = QAEPricer(params, n)
        problem = pricer.estimation_problem()
        row = {"n_qubits": n, "total_qubits": pricer.payoff.num_qubits}
        for tag, circ in (("A", problem.state_preparation), ("Q", problem.grover_operator)):
            t = transpile(circ, basis_gates=["cx", "u"], optimization_level=1, seed_transpiler=DEFAULT_SEED)
            ops = t.count_ops()
            row.update({f"{tag}_depth": t.depth(), f"{tag}_cx": int(ops.get("cx", 0)),
                        f"{tag}_gates": int(sum(ops.values()))})
        rows.append(row)
    return pd.DataFrame(rows)

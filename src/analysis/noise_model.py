"""Depolarizing-noise evaluation of IQAE on ``qiskit-aer``."""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from config import DEFAULT_SEED, MarketParams
from src.quantum.qae_pricer import QAEPricer

BASIS_GATES = ["u", "cx"]


def build_depolarizing_noise_model(p1: float, p2: Optional[float] = None) -> Any:
    """Depolarizing noise on the ``{u, cx}`` basis.

    Args:
        p1: Single-qubit depolarizing probability (on ``u``).
        p2: Two-qubit depolarizing probability (on ``cx``); defaults to ``10 * p1``,
            a typical hardware ratio.

    Returns:
        A ``qiskit_aer.noise.NoiseModel``.
    """
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    p2 = 10.0 * p1 if p2 is None else p2
    nm = NoiseModel(basis_gates=BASIS_GATES)
    if p1 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["u"])
    if p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    return nm


def noisy_sampler(p1: float, p2: Optional[float] = None, shots: int = 100, seed: int = DEFAULT_SEED) -> Any:
    """Aer-backed V1 ``Sampler`` with depolarizing noise (circuits are transpiled to ``{u, cx}``)."""
    from qiskit_aer.primitives import Sampler

    return Sampler(
        backend_options={"noise_model": build_depolarizing_noise_model(p1, p2), "seed_simulator": seed},
        transpile_options={"basis_gates": BASIS_GATES, "optimization_level": 1, "seed_transpiler": seed},
        run_options={"shots": shots, "seed": seed},
    )


def noise_sweep(
    params: MarketParams,
    n_qubits: int = 3,
    p1_values: Sequence[float] = (0.0, 1e-5, 1e-4, 5e-4, 1e-3),
    epsilon_target: float = 0.002,
    shots: int = 100,
    n_repeats: int = 2,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Run IQAE under increasing depolarizing noise.

    For each ``p1`` the 2-qubit error is ``10 * p1``.  Errors are measured
    against the noiseless exact-circuit price, and compared with the error a
    classical Monte Carlo run would achieve with the *same number of queries*
    (``std_payoff / sqrt(queries)``) to show where the advantage disappears.

    Returns:
        DataFrame with columns ``p1, p2, mean_abs_error, mean_queries,
        mc_equiv_error, quantum_advantage``.
    """
    from src.classical.monte_carlo import simulate_terminal_prices

    ref = QAEPricer(params, n_qubits)
    target = ref.exact_circuit_price()
    rng = np.random.default_rng(seed)
    payoff_std = float(
        np.std(params.discount * np.maximum(simulate_terminal_prices(params, 200_000, rng) - params.K, 0.0))
    )
    rows = []
    for p1 in p1_values:
        errs, qs = [], []
        for rep in range(n_repeats):
            pricer = QAEPricer(params, n_qubits, sampler=noisy_sampler(p1, shots=shots, seed=seed + rep))
            try:
                res = pricer.price(epsilon_target=epsilon_target)
                errs.append(abs(res.price - target))
                qs.append(res.num_oracle_queries)
            except Exception as exc:  # IQAE can fail to converge under heavy noise
                errs.append(float("nan"))
                qs.append(float("nan"))
                print(f"  [warn] p1={p1}: IQAE failed ({exc})")
        err, q = float(np.nanmean(errs)), float(np.nanmean(qs))
        mc_err = payoff_std / math.sqrt(q) if q == q and q > 0 else float("nan")
        rows.append(
            {"p1": p1, "p2": 10 * p1, "mean_abs_error": err, "mean_queries": q,
             "mc_equiv_error": mc_err, "quantum_advantage": bool(err < mc_err)}
        )
    return pd.DataFrame(rows)

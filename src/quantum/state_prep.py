"""Discretised log-normal state preparation.

The terminal price ``S_T`` under Black-Scholes-Merton is log-normal,
``ln S_T ~ N(mu, s^2)`` with ``mu = ln S0 + (r - sigma^2/2) T`` and
``s = sigma sqrt(T)``.  We truncate to ``[low, high] = exp(mu +- k s)`` and
place ``N = 2^n`` equispaced grid points ``S_i`` on it.  Grid point ``i`` carries
the probability mass of its surrounding bin,

    p_i = (F(S_i + h/2) - F(S_i - h/2)) / Z,     h = (high - low) / (N - 1),

where ``F`` is the log-normal CDF and ``Z`` renormalises the truncated mass.
The circuit prepares ``|psi> = sum_i sqrt(p_i) |i>``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import lognorm

from config import MarketParams

if TYPE_CHECKING:  # pragma: no cover
    from qiskit import QuantumCircuit


@dataclass(frozen=True)
class LogNormalGrid:
    """A discretised log-normal distribution on ``2**n_qubits`` grid points.

    Attributes:
        n_qubits: Number of qubits ``n``.
        grid: Asset prices ``S_i`` (length ``2**n``).
        probabilities: Bin probabilities ``p_i`` (sum to one).
        low: Lower truncation bound.
        high: Upper truncation bound.
        mu: Mean of ``ln S_T``.
        s: Standard deviation of ``ln S_T``.
    """

    n_qubits: int
    grid: np.ndarray
    probabilities: np.ndarray
    low: float
    high: float
    mu: float
    s: float

    @property
    def num_points(self) -> int:
        """Number of grid points ``2**n``."""
        return len(self.grid)

    def expected_call_payoff(self, strike: float) -> float:
        """Exact (undiscounted) ``E[max(S - K, 0)]`` on the discretised grid."""
        return float(np.dot(self.probabilities, np.maximum(self.grid - strike, 0.0)))


def build_lognormal_grid(p: MarketParams, n_qubits: int, num_std: float = 3.0) -> LogNormalGrid:
    """Discretise the risk-neutral terminal distribution of ``S_T``.

    Args:
        p: Market parameters.
        n_qubits: Number of uncertainty qubits ``n`` (``2**n`` grid points).
        num_std: Truncation width ``k`` in log-space standard deviations.

    Returns:
        A :class:`LogNormalGrid`.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1.")
    mu = math.log(p.S0) + (p.r - 0.5 * p.sigma**2) * p.T
    s = p.sigma * math.sqrt(p.T)
    low, high = math.exp(mu - num_std * s), math.exp(mu + num_std * s)
    n_pts = 2**n_qubits
    grid = np.linspace(low, high, n_pts)
    h = grid[1] - grid[0]
    edges = np.concatenate([[grid[0] - h / 2], 0.5 * (grid[:-1] + grid[1:]), [grid[-1] + h / 2]])
    probs = np.diff(lognorm(s=s, scale=math.exp(mu)).cdf(edges))
    probs = probs / probs.sum()
    return LogNormalGrid(n_qubits, grid, probs, low, high, mu, s)


def lognormal_state_circuit(grid: LogNormalGrid) -> "QuantumCircuit":
    """Build the circuit loading ``sum_i sqrt(p_i)|i>`` on ``n`` qubits.

    Uses Qiskit's ``StatePreparation`` (generic amplitude loading, ``O(2^n)``
    gates).  Structured alternatives (e.g. Grover-Rudolph, qGANs) can lower the
    cost but are not needed at the sizes studied here.

    Args:
        grid: Discretised distribution.

    Returns:
        A circuit with ``grid.n_qubits`` qubits (little-endian: qubit 0 is the LSB).
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation

    qc = QuantumCircuit(grid.n_qubits, name="lognormal")
    qc.append(StatePreparation(np.sqrt(grid.probabilities)), range(grid.n_qubits))
    return qc

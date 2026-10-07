"""Piecewise-linear call payoff operator via controlled-Ry rotations.

For ``f(S) = max(S - K, 0)`` on ``[low, high]`` we rescale to ``f~ in [0, 1]``,
``f~ = f / f_max`` with ``f_max = high - K``, and apply

    |i>|0>  ->  |i> ( cos(g(i)) |0> + sin(g(i)) |1> ),
    g(i) = pi/4 + c (f~(S_i) - 1/2),

(the "approximation" trick of Stamatopoulos et al., 2020), so that

    P(ancilla = 1) = sin^2(g) = 1/2 + c (f~ - 1/2) + O(c^3).

The kink at ``S = K`` is handled by an integer comparator that selects between
two linear-Ry segments (Qiskit's ``LinearAmplitudeFunction``).  Inverting the
affine map recovers ``E[f~]`` and hence ``E[f]`` from the measured probability.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.quantum.state_prep import LogNormalGrid

if TYPE_CHECKING:  # pragma: no cover
    from qiskit.circuit.library import LinearAmplitudeFunction


def call_payoff_on_grid(grid: LogNormalGrid, strike: float) -> np.ndarray:
    """Classical reference payoff ``max(S_i - K, 0)`` on the grid."""
    return np.maximum(grid.grid - strike, 0.0)


def build_call_payoff(grid: LogNormalGrid, strike: float, c_approx: float = 0.25) -> "LinearAmplitudeFunction":
    """Build the payoff operator for a European call.

    Args:
        grid: Discretised price distribution.
        strike: Strike ``K``; must satisfy ``low < K < high``.
        c_approx: Rescaling parameter ``c`` in ``(0, 1/2]``.  Smaller ``c`` reduces
            the ``O(c^3)`` bias but amplifies estimation error by ``1/c``.

    Returns:
        A ``LinearAmplitudeFunction`` on ``n + 1 (+ ancilla)`` qubits whose
        objective qubit has index ``n``; its ``post_processing`` maps the
        measured probability back to ``E[f]`` in price units.
    """
    from qiskit.circuit.library import LinearAmplitudeFunction

    if not grid.low < strike < grid.high:
        raise ValueError(f"Strike {strike} must lie inside the truncated domain ({grid.low:.2f}, {grid.high:.2f}).")
    f_max = grid.high - strike
    # f(x) = slope_i * (x - breakpoint_i) + offset_i on [breakpoint_i, breakpoint_{i+1})
    return LinearAmplitudeFunction(
        grid.n_qubits,
        [0.0, 1.0],
        [0.0, 0.0],
        domain=(grid.low, grid.high),
        image=(0.0, f_max),
        breakpoints=[grid.low, strike],
        rescaling_factor=c_approx,
    )

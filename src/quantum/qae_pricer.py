"""Iterative Quantum Amplitude Estimation (IQAE) pricer for a European call."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from config import DEFAULT_SEED, MarketParams
from src.quantum.payoff import build_call_payoff
from src.quantum.state_prep import LogNormalGrid, build_lognormal_grid, lognormal_state_circuit


def default_sampler(shots: int = 100, seed: Optional[int] = DEFAULT_SEED) -> Any:
    """Ideal (noise-free) reference ``Sampler`` from ``qiskit.primitives``.

    Args:
        shots: Shots per circuit execution inside IQAE.
        seed: Sampling seed.
    """
    from qiskit.primitives import Sampler

    return Sampler(options={"shots": shots, "seed": seed})


@dataclass
class QAEResult:
    """Result of one IQAE pricing run.

    Attributes:
        price: Discounted option value ``exp(-rT) * E[f]``.
        ci_low / ci_high: Confidence interval on the price.
        amplitude: Raw estimated amplitude ``a`` (probability of the good state).
        num_oracle_queries: Total applications of the state-prep operator ``A``
            (Grover powers included), as counted by Qiskit.
        epsilon_target: Requested half-width of the CI on ``a``.
        alpha: Confidence-level parameter (CI holds w.p. ``1 - alpha``).
        n_qubits: Number of uncertainty qubits.
    """

    price: float
    ci_low: float
    ci_high: float
    amplitude: float
    num_oracle_queries: int
    epsilon_target: float
    alpha: float
    n_qubits: int


class QAEPricer:
    """Prices a European call with IQAE on a discretised log-normal model.

    Args:
        params: Market parameters.
        n_qubits: Uncertainty qubits (``2**n`` grid points).
        c_approx: Payoff rescaling parameter ``c``.
        num_std: Log-space truncation width.
        sampler: Qiskit V1 ``Sampler``; defaults to the ideal reference sampler.
    """

    def __init__(
        self,
        params: MarketParams,
        n_qubits: int = 5,
        c_approx: float = 0.25,
        num_std: float = 3.0,
        sampler: Any = None,
    ) -> None:
        self.params = params
        self.n_qubits = n_qubits
        self.c_approx = c_approx
        self.grid: LogNormalGrid = build_lognormal_grid(params, n_qubits, num_std)
        self.payoff = build_call_payoff(self.grid, params.K, c_approx)
        self.sampler = sampler if sampler is not None else default_sampler()
        self._problem: Any = None

    # ----- circuits -------------------------------------------------------
    def state_preparation(self) -> Any:
        """The operator ``A``: log-normal loading followed by the payoff rotation."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(self.payoff.num_qubits, name="A")
        qc.append(lognormal_state_circuit(self.grid).to_gate(), range(self.n_qubits))
        qc.append(self.payoff.to_gate(), range(self.payoff.num_qubits))
        return qc

    def estimation_problem(self) -> Any:
        """Qiskit ``EstimationProblem`` (objective qubit = index ``n``)."""
        if self._problem is None:
            from qiskit_algorithms import EstimationProblem

            self._problem = EstimationProblem(
                state_preparation=self.state_preparation(),
                objective_qubits=[self.n_qubits],
                post_processing=self.payoff.post_processing,
            )
        return self._problem

    # ----- reference values ----------------------------------------------
    @property
    def discretised_price(self) -> float:
        """Exact discounted price on the grid (no circuit, no estimation error)."""
        return self.params.discount * self.grid.expected_call_payoff(self.params.K)

    def exact_circuit_price(self) -> float:
        """Price implied by the *exact* amplitude of the circuit (statevector).

        This is what IQAE converges to as ``epsilon -> 0``; it differs from
        :attr:`discretised_price` only by the ``O(c^3)`` payoff-approximation bias.
        Comparing IQAE against this value isolates estimation error.
        """
        from qiskit.quantum_info import Statevector

        a = float(Statevector(self.state_preparation()).probabilities([self.n_qubits])[1])
        return self.params.discount * float(self.payoff.post_processing(a))

    # ----- estimation -----------------------------------------------------
    def price(self, epsilon_target: float = 0.001, alpha: float = 0.05) -> QAEResult:
        """Run IQAE and convert the estimate to a dollar option value.

        ``Price = exp(-rT) * rescaled_estimation``, where ``rescaled_estimation``
        is ``E[max(S - K, 0)]`` obtained by inverting the amplitude encoding.

        Args:
            epsilon_target: Target half-width on the *amplitude* ``a``.  The price
                error scales as ``epsilon * f_max / c`` (``f_max = high - K``).
            alpha: Confidence-level parameter.

        Returns:
            A :class:`QAEResult`.
        """
        from qiskit_algorithms import IterativeAmplitudeEstimation

        iqae = IterativeAmplitudeEstimation(epsilon_target=epsilon_target, alpha=alpha, sampler=self.sampler)
        res = iqae.estimate(self.estimation_problem())
        disc = self.params.discount
        lo, hi = res.confidence_interval_processed
        return QAEResult(
            price=disc * float(res.estimation_processed),
            ci_low=disc * float(lo),
            ci_high=disc * float(hi),
            amplitude=float(res.estimation),
            num_oracle_queries=int(res.num_oracle_queries),
            epsilon_target=epsilon_target,
            alpha=alpha,
            n_qubits=self.n_qubits,
        )

    def price_error_scale(self) -> float:
        """Approximate dollars of price error per unit of amplitude error."""
        return self.params.discount * (self.grid.high - self.params.K) / self.c_approx

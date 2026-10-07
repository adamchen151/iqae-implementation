"""Tests: quantum state preparation and IQAE accuracy."""
import numpy as np
import pytest

from config import DEFAULT_PARAMS
from src.classical.black_scholes import call_price
from src.quantum.state_prep import build_lognormal_grid

BS = call_price(DEFAULT_PARAMS)


# ---- pure NumPy checks (no Qiskit needed) -------------------------------
@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_grid_is_normalised(n: int) -> None:
    g = build_lognormal_grid(DEFAULT_PARAMS, n)
    assert g.num_points == 2**n
    assert g.probabilities.sum() == pytest.approx(1.0)
    assert np.all(g.probabilities >= 0)


@pytest.mark.parametrize("n", [4, 5, 6])
def test_discretisation_error_below_5_percent(n: int) -> None:
    g = build_lognormal_grid(DEFAULT_PARAMS, n)
    price = DEFAULT_PARAMS.discount * g.expected_call_payoff(DEFAULT_PARAMS.K)
    assert abs(price - BS) / BS < 0.05


# ---- Qiskit checks ------------------------------------------------------
def _need_qiskit() -> None:
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_algorithms")


def test_state_prep_amplitudes() -> None:
    _need_qiskit()
    from qiskit.quantum_info import Statevector
    from src.quantum.state_prep import lognormal_state_circuit

    g = build_lognormal_grid(DEFAULT_PARAMS, 4)
    probs = Statevector(lognormal_state_circuit(g)).probabilities()
    np.testing.assert_allclose(probs, g.probabilities, atol=1e-8)


def test_payoff_amplitude_matches_classical() -> None:
    _need_qiskit()
    from src.quantum.qae_pricer import QAEPricer

    pr = QAEPricer(DEFAULT_PARAMS, 4)
    # exact circuit value differs from grid value only by the O(c^3) bias (<2%)
    assert abs(pr.exact_circuit_price() - pr.discretised_price) / pr.discretised_price < 0.02


@pytest.mark.slow
def test_iqae_within_5_percent_of_black_scholes() -> None:
    _need_qiskit()
    from src.quantum.qae_pricer import QAEPricer

    pr = QAEPricer(DEFAULT_PARAMS, 4)
    res = pr.price(epsilon_target=0.001, alpha=0.05)
    assert abs(res.price - BS) / BS < 0.05
    assert res.ci_low <= res.price <= res.ci_high
    assert res.num_oracle_queries > 0

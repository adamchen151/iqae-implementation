"""CLI entrypoint: pricing, benchmarks, circuit profiling, noise study and tests.

Examples:
    python main.py price --qubits 5 --epsilon 0.001
    python main.py benchmark
    python main.py profile
    python main.py noise
    python main.py test
    python main.py all
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from config import DEFAULT_PARAMS, RESULTS_DIR, MarketParams
from src.classical import black_scholes as bs
from src.classical.monte_carlo import mc_call_price
from src.analysis.benchmarking import classical_convergence, fit_loglog_slope


def _params(a: argparse.Namespace) -> MarketParams:
    return MarketParams(S0=a.S0, K=a.K, r=a.r, sigma=a.sigma, T=a.T)


def cmd_price(a: argparse.Namespace) -> None:
    p = _params(a)
    truth = bs.call_price(p)
    mc = mc_call_price(p, a.mc_samples)
    print(f"Black-Scholes : {truth:.4f}")
    print(f"Monte Carlo   : {mc.price:.4f}  (±{1.96 * mc.std_error:.4f}, M={mc.n_samples})")
    from src.quantum.qae_pricer import QAEPricer

    pr = QAEPricer(p, a.qubits)
    q = pr.price(a.epsilon)
    print(f"Discretised   : {pr.discretised_price:.4f}  (n={a.qubits} qubits, exact on grid)")
    print(f"IQAE          : {q.price:.4f}  CI=[{q.ci_low:.4f}, {q.ci_high:.4f}], queries={q.num_oracle_queries}")


def cmd_benchmark(a: argparse.Namespace) -> None:
    from src.analysis.plotting import plot_convergence

    p = _params(a)
    os.makedirs(a.outdir, exist_ok=True)
    mc = classical_convergence(p)
    mc.to_csv(f"{a.outdir}/mc_convergence.csv", index=False)
    print(f"MC log-log slope: {fit_loglog_slope(mc['effort'], mc['mean_abs_error']):.3f} (theory -0.5)")
    qdf = None
    if not a.skip_quantum:
        from src.analysis.benchmarking import quantum_convergence

        qdf = quantum_convergence(p, n_qubits=a.qubits)
        qdf.to_csv(f"{a.outdir}/iqae_convergence.csv", index=False)
        print(qdf.to_string(index=False))
        print(f"IQAE log-log slope: {fit_loglog_slope(qdf['effort'], qdf['mean_abs_error']):.3f} (theory -1.0)")
    plot_convergence(mc, qdf, f"{a.outdir}/convergence.png")
    print(f"Saved {a.outdir}/convergence.png")


def cmd_profile(a: argparse.Namespace) -> None:
    from src.analysis.benchmarking import circuit_profile
    from src.analysis.plotting import plot_circuit_profile

    os.makedirs(a.outdir, exist_ok=True)
    df = circuit_profile(_params(a))
    df.to_csv(f"{a.outdir}/circuit_profile.csv", index=False)
    print(df.to_string(index=False))
    plot_circuit_profile(df, f"{a.outdir}/circuit_profile.png")


def cmd_noise(a: argparse.Namespace) -> None:
    from src.analysis.noise_model import noise_sweep
    from src.analysis.plotting import plot_noise

    os.makedirs(a.outdir, exist_ok=True)
    df = noise_sweep(_params(a), n_qubits=a.noise_qubits)
    df.to_csv(f"{a.outdir}/noise_sweep.csv", index=False)
    print(df.to_string(index=False))
    plot_noise(df, f"{a.outdir}/noise.png")


def cmd_test(a: argparse.Namespace) -> None:
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-v"] + (["-m", "not slow"] if a.fast else [])))


def cmd_all(a: argparse.Namespace) -> None:
    for fn in (cmd_price, cmd_benchmark, cmd_profile, cmd_noise):
        fn(a)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="QAE vs classical European call pricing.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = DEFAULT_PARAMS
    for name, fn, helptext in (
        ("price", cmd_price, "Price once with BS, MC and IQAE"),
        ("benchmark", cmd_benchmark, "Error-vs-effort convergence study"),
        ("profile", cmd_profile, "Circuit depth / CNOT / gate counts for n=3..6"),
        ("noise", cmd_noise, "Depolarizing-noise IQAE study"),
        ("test", cmd_test, "Run the pytest suite"),
        ("all", cmd_all, "price + benchmark + profile + noise"),
    ):
        sp = sub.add_parser(name, help=helptext)
        sp.set_defaults(func=fn)
        for k in ("S0", "K", "r", "sigma", "T"):
            sp.add_argument(f"--{k}", type=float, default=getattr(d, k))
        sp.add_argument("--qubits", type=int, default=5, help="uncertainty qubits for IQAE")
        sp.add_argument("--epsilon", type=float, default=0.001, help="IQAE target amplitude half-width")
        sp.add_argument("--mc-samples", type=int, default=100_000)
        sp.add_argument("--noise-qubits", type=int, default=3)
        sp.add_argument("--outdir", default=RESULTS_DIR)
        sp.add_argument("--skip-quantum", action="store_true", help="benchmark: classical only")
        sp.add_argument("--fast", action="store_true", help="test: skip slow quantum simulations")
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)

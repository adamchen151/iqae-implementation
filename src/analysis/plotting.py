"""Publication-style figures for the benchmark results."""
from __future__ import annotations

from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.benchmarking import fit_loglog_slope  # noqa: E402

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 120})


def plot_convergence(mc: pd.DataFrame, qae: Optional[pd.DataFrame], path: str) -> None:
    """Log-log plot of absolute error vs. computational effort with fitted slopes."""
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    s_mc = fit_loglog_slope(mc["effort"], mc["mean_abs_error"])
    ax.loglog(mc["effort"], mc["mean_abs_error"], "o-", label=f"Classical MC (slope {s_mc:.2f})")
    x0, y0 = mc["effort"].iloc[0], mc["mean_abs_error"].iloc[0]
    xs = np.array([mc["effort"].min(), mc["effort"].max()])
    ax.loglog(xs, y0 * (xs / x0) ** -0.5, "k--", alpha=0.5, label=r"$\mathcal{O}(N^{-1/2})$")
    if qae is not None and len(qae) > 1:
        s_q = fit_loglog_slope(qae["effort"], qae["mean_abs_error"])
        ax.loglog(qae["effort"], qae["mean_abs_error"], "s-", label=f"IQAE (slope {s_q:.2f})")
        xq = np.array([qae["effort"].min(), qae["effort"].max()])
        ax.loglog(xq, qae["mean_abs_error"].iloc[0] * (xq / qae["effort"].iloc[0]) ** -1.0, "r:", alpha=0.6,
                  label=r"$\mathcal{O}(N^{-1})$")
    ax.set_xlabel("Computational effort (MC samples / IQAE oracle queries)")
    ax.set_ylabel("Absolute pricing error ($)")
    ax.set_title("Convergence: classical MC vs. IQAE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_circuit_profile(df: pd.DataFrame, path: str) -> None:
    """Depth / CNOT / gate counts of the Grover operator vs. number of qubits."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for col, lab in (("Q_depth", "Depth"), ("Q_cx", "CNOT count"), ("Q_gates", "Total gates")):
        ax.semilogy(df["n_qubits"], df[col], "o-", label=lab)
    ax.set_xticks(df["n_qubits"])
    ax.set_xlabel("Uncertainty qubits n")
    ax.set_ylabel("Count (transpiled to {cx, u})")
    ax.set_title("Grover operator Q resource scaling")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_noise(df: pd.DataFrame, path: str) -> None:
    """IQAE error vs. depolarizing noise against the equal-query MC error."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    d = df[df["p1"] > 0]
    ax.loglog(d["p1"], d["mean_abs_error"], "s-", label="IQAE (noisy)")
    ax.loglog(d["p1"], d["mc_equiv_error"], "o--", label="Classical MC, same #queries")
    if (df["p1"] == 0).any():
        ax.axhline(df.loc[df["p1"] == 0, "mean_abs_error"].iloc[0], color="g", ls=":", label="IQAE (noiseless)")
    ax.set_xlabel("1-qubit depolarizing probability $p_1$ ($p_2 = 10\\,p_1$)")
    ax.set_ylabel("Absolute pricing error ($)")
    ax.set_title("Where NISQ noise erases the advantage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

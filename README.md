# QAE for European Call Pricing

Prices a European call under Black-Scholes-Merton with (1) the analytical formula, (2) classical Monte Carlo, and (3) Iterative Quantum Amplitude Estimation (IQAE) in Qiskit, then benchmarks error vs. computational effort, circuit cost, and NISQ noise sensitivity.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # Qiskit 1.x (<2.0) is required, see note below
```

> **Why `qiskit<2.0`?** `qiskit_algorithms.IterativeAmplitudeEstimation` uses the V1 `Sampler` primitives, which were removed in Qiskit 2.0.

## Usage

```bash
python main.py price --qubits 5 --epsilon 0.001   # BS vs MC vs IQAE
python main.py benchmark                          # convergence plot -> results/convergence.png
python main.py benchmark --skip-quantum           # classical only (no Qiskit needed)
python main.py profile                            # depth / CNOT / gates for n = 3..6
python main.py noise                              # depolarizing-noise study (needs qiskit-aer)
python main.py test --fast                        # pytest, skipping the slow IQAE test
python main.py all
```
Market parameters are overridable (`--S0 --K --r --sigma --T`); defaults live in `config.py` (S0=K=100, r=5%, σ=20%, T=1; BS price 10.4506).

## Mathematical formulation

**Model.** Under the risk-neutral measure `ln S_T ~ N(μ, s²)`, `μ = ln S0 + (r − σ²/2)T`, `s = σ√T`. Price `= e^{−rT} E[max(S_T − K, 0)]`.

**Log-normal discretisation.** Truncate to `[S_min, S_max] = exp(μ ± k s)` (k=3), take `N = 2^n` equispaced points `S_i`, spacing `h`, and assign bin probabilities

`p_i = [F(S_i + h/2) − F(S_i − h/2)] / Z`, with `F` the log-normal CDF and `Z` the truncated mass.

The loader `U` satisfies `U|0⟩ⁿ = Σ_i √p_i |i⟩`. Discretisation error with k=3: 1.1% (n=4), 0.7% (n=5), 0.9% (n=6) (n=3: 3.6%); it does not vanish with n because the truncation and the fixed-width grid dominate.

**Controlled-Ry amplitude encoding.** With `f̃(S) = max(S − K, 0)/f_max ∈ [0,1]`, `f_max = S_max − K`, a payoff operator acts on an ancilla:

`|i⟩|0⟩ → |i⟩( cos g_i |0⟩ + sin g_i |1⟩ )`, `g_i = π/4 + c (f̃(S_i) − ½)`.

Then `P(1) = sin² g_i = ½ + c(f̃_i − ½) + O(c³)`, so the amplitude to estimate is

`a = ½ + c (E[f̃] − ½) + O(c³)`  ⟹  `E[f] = f_max · ((a − ½)/c + ½)`,  `Price = e^{−rT} E[f]`.

The angle `g_i` is linear in `f̃`, so it is implemented with Ry rotations controlled on the index qubits (angle ∝ 2^j for qubit j); the kink at `K` is handled by an integer comparator that switches between the zero and unit-slope segments (Qiskit's `LinearAmplitudeFunction`). Trade-off: error on the price is `≈ ε · f_max / c` for amplitude error ε, while the bias is `O(c³)`. Default `c = 0.25`.

**IQAE.** Estimates `a` to ±ε with `O((1/ε) log(1/α))` applications of the Grover operator `Q = A S₀ A† S_ψ`, versus `O(1/ε²)` samples for MC.

## Architecture

```
config.py                  market parameters
src/classical/             black_scholes.py (price + Greeks), monte_carlo.py
src/quantum/               state_prep.py, payoff.py, qae_pricer.py
src/analysis/              benchmarking.py, noise_model.py, plotting.py
tests/                     test_classical.py, test_quantum.py
main.py                    CLI
```
`state_prep.py` / `payoff.py` implement the circuits directly (a custom log-normal loader plus Qiskit's core `LinearAmplitudeFunction`), so `qiskit-finance` is not required.

## Benchmark methodology

* **Effort axis**: MC = number of samples; IQAE = `result.num_oracle_queries` (total applications of `A`, Grover powers included).
* **Error**: MC vs analytical price (averaged over 30 repeats). IQAE vs the *exact value implied by the circuit* (statevector), so the fixed discretisation and `c`-bias don't create an artificial error floor; bias vs Black-Scholes is reported separately in `bs_error`.
* The plotted slopes are fits on few points with noisy, shot-based IQAE data; expect values around −1 but with visible scatter, and a deviation from the ideal at coarse ε because IQAE's guarantee is an upper bound with log factors.
* Circuit profiling transpiles `A` and one Grover operator `Q` to `{cx, u}` at optimization level 1. Cost of `Q` grows roughly exponentially in `n` because generic amplitude loading is `O(2^n)`.
* Noise: depolarizing error `p1` on `u` and `p2 = 10 p1` on `cx`, compared against the error MC would reach at the same query count.

## NISQ vs. Fault-Tolerant Reality Check

**What the quadratic speedup actually buys.** IQAE needs `~1/ε` sequential Grover iterations; MC needs `~1/ε²` samples that are embarrassingly parallel. A price to 1 bp with MC is ~10⁸ samples, trivially run on a GPU in seconds. The quantum advantage only exists if one coherent circuit of depth `~1/ε × depth(Q)` runs with high fidelity.

**NISQ.** The Grover operator here already has hundreds to thousands of CNOTs at n = 5–6 (see `circuit_profile.csv`). With two-qubit error rates of ~10⁻³, a circuit with `10³` CNOTs has overall fidelity ≈ e⁻¹ — and IQAE at useful ε needs *powers* `Q^k`, `k` in the hundreds. `python main.py noise` sweeps `p1` and reports where IQAE's error exceeds the equal-query MC error (column `quantum_advantage`); expect the crossover at fairly small `p1` even for a 3-qubit toy problem, and it moves sharply worse as `n` grows. NISQ devices cannot show an advantage for this algorithm.

**Fault tolerance.** Needed: error-corrected logical qubits, T-gate–heavy arithmetic (comparators, reversible payoff/path logic), and a fast logical clock. Published resource estimates (e.g. Chakrabarti et al., *Quantum* 2021, "A threshold for quantum advantage in derivative pricing") put the break-even for a realistic derivative (path-dependent, multi-asset) at on the order of ~10⁴ logical qubits, T-depth of order 10⁷–10⁸ per Grover iteration-block and a logical clock rate of ~10 MHz or higher.

**Caveats that erode the advantage even with perfect hardware.**
1. *State loading*: generic loading costs `O(2^n)`; efficient loading needs structured distributions (or qGANs / Grover-Rudolph, with their own overhead). For path-dependent products, the loader scales with time steps × assets.
2. *Classical competition*: quasi-Monte Carlo, variance reduction, and GPU parallelism reduce the effective classical constant (QMC already approaches `~1/ε` for smooth payoffs).
3. *Constant factors*: the `1/c` payoff-approximation overhead and the truncation/discretisation errors add multiplicative cost.
4. *Fair comparison*: quantum cost is wall-clock sequential; classical cost parallelises.

**Bottom line.** For single-asset Europeans, no practical quantum advantage exists, now or probably ever. The plausible regime is high-dimensional, path-dependent, or XVA-type risk workloads with strict precision targets on fault-tolerant hardware with fast logical clocks — a regime this repo can model but not demonstrate.

## Testing

`pytest` (see `pytest.ini`): analytical sanity (put-call parity, Greeks vs finite differences), MC within 5% and slope ≈ −0.5, state-prep fidelity, discretisation error, and IQAE within 5% of Black-Scholes (marked `slow`). Quantum tests auto-skip if Qiskit is not installed.

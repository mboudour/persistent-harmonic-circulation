# Persistent Harmonic Circulation on Filtered Cell Complexes

Code and data for the paper *Persistent Harmonic Circulation on Filtered Cell Complexes* (M. Boudourides).

The paper studies the node–edge cochain system

    x' = -d0* u,    u' = -beta * Delta_1 u + d0 x

(Boudourides, *Chaos, Solitons and Fractals* 214 (2027) 119473) when the cell complex changes along a filtration or a zigzag. On each stage the system is integrated exactly with matrix exponentials. Edge states move across inclusions by extension by zero and across removals by restriction.

## Contents

- `code/phc.py`: cell complexes, Hodge projectors, harmonic transport `Phi`, leakage operator `Lambda`, the switched system, and two persistent-Laplacian routines:
  - `persistent_laplacian_1(K, L)` builds the operator as defined in the paper: `d0 d0^* + D D^*`, with `D = rho d1^*` on an orthonormal basis of the constrained 2-chain domain.
  - `persistent_harmonic_kernel(K, L)` returns only its kernel, computed as `Z_1(K) ∩ (B_1(L) ∩ C_1(K))^perp`. This gives the same subspace without the constrained domain and is used for large Rips complexes.
- `code/test_suite.py`: assertion-based checks of the paper's statements. Matrix identities use tolerance 1e-9, and singular values below 1e-8 count as zero. It includes a filled-triangle regression test for the persistent Laplacian and a comparison of GUDHI persistent Betti numbers (over F_11 and F_13) with ranks over R.
- `code/test_theorems.py`: transport and persistent-Laplacian identities on 25 random Rips filtrations.
- `code/test_dynamics.py`: dwell-time checks on random 2-complex filtrations.
- `code/experiments.py`: Experiments 1–5 of the paper. Writes `figures/*.pdf` and `results/summary.json`. The files `fig1_generation_and_dwell.pdf`, `fig2_rips.pdf` and `fig3_leakage_combinatorics.pdf` are Figures 2, 3 and 4 of the paper; Figure 1 is drawn in TikZ.

GUDHI computes persistence over a finite field; the default is F_11. The theory is over R. The test suite compares the two on the complexes used.

## Reproduce

    pip install -r requirements.txt
    python code/test_suite.py
    python code/experiments.py
    python code/test_theorems.py
    python code/test_dynamics.py

`test_suite.py` and `experiments.py` each finish in under a minute on a laptop. `test_theorems.py` takes about two minutes. All random seeds are fixed in the scripts.

## Data

No external data are used. `results/summary.json` and `figures/*.pdf` are generated outputs, and `code/experiments.py` reproduces them: the JSON byte for byte, and the PDFs without creation timestamps. They are committed only for convenience.

## Tolerances

- **Assertions:** identities between matrices are asserted with an absolute tolerance of 1e-9 (`rtol=0`). Comparisons of simulated limits with predictions use 1e-6, and the Rips dwell-time check uses 1e-5.
- **Ranks:** in tests, singular values below 1e-8 count as zero.
- **Hodge bases:** harmonic subspaces are computed exactly over Q as integer null spaces with FLINT. The bases are LLL-reduced and then orthonormalized in floating point. The exact projector is the orthogonal complement of the harmonic and exact projectors. The rank of the exact space is certified by exact integer arithmetic.
- **`persistent_harmonic_kernel`:** the subspace is computed exactly over Q. It is the integer null space of the cycle conditions together with an exact integer generating set of B_1(L) ∩ C_1(K), and the result is checked exactly in integer arithmetic. Only the final orthonormalization is floating point.
- **`persistent_laplacian_1`:** this reference operator uses an exact basis of its constrained domain. Its eigenvalues near zero are still limited by the conditioning of the complex, so it is compared with the exact kernel routine only on small complexes.

All four scripts, including `experiments.py`, assert the bounds, identities and vanishing statements they report. Other printed numbers (ratios, frequencies, errors at particular dwell times) are descriptive.

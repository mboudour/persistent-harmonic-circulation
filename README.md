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
- `code/experiments.py`: Experiments 1–5 of the paper. Writes `figures/*.pdf` and `results/summary.json`.

GUDHI computes persistence over a finite field; the default is F_11. The theory is over R. The test suite compares the two on the complexes used.

## Reproduce

    pip install -r requirements.txt
    python code/test_suite.py
    python code/experiments.py
    python code/test_theorems.py
    python code/test_dynamics.py

`test_suite.py` and `experiments.py` each finish in under a minute on a laptop. `test_theorems.py` takes about two minutes. All random seeds are fixed in the scripts.

## Data

No external data are used. `results/summary.json` and `figures/*.pdf` are generated outputs, and `code/experiments.py` reproduces them exactly. They are committed only for convenience.

## Tolerances

- **Identities:** assertions compare with an absolute tolerance of 1e-9 (`rtol=0`).
- **Ranks:** singular values below 1e-8 count as zero.
- **Harmonic bases:** computed with `scipy.linalg.null_space`, using a cutoff of 1e-10 relative to the largest singular value of the Hodge Laplacian.
- **`persistent_harmonic_kernel`:** uses absolute cutoffs of 1e-10 on blocks of orthonormal matrices and checks every returned vector against the defining constraints.

All four scripts assert their checks.

# Persistent Harmonic Circulation on Filtered Cell Complexes

Code and data for the paper *Persistent Harmonic Circulation on Filtered Cell Complexes* (M. Boudourides).

The paper studies the node–edge cochain system

    x' = -d0* u,    u' = -beta * Delta_1 u + d0 x

(Boudourides, *Chaos, Solitons and Fractals* 214 (2027) 119473) when the cell complex changes along a filtration or a zigzag. On each stage the system is integrated exactly with matrix exponentials. Edge states move across inclusions by extension by zero and across removals by restriction.

## Contents

- `code/phc.py`: cell complexes, Hodge projectors, harmonic transport `Phi`, leakage operator `Lambda`, persistent Laplacian, and the switched system.
- `code/test_theorems.py`: numerical checks of the switching identity, harmonic transport, persistent Laplacian kernels and adjoint transport on 25 random Rips filtrations.
- `code/test_dynamics.py`: dwell-time checks on random 2-complex filtrations.
- `code/experiments.py`: Experiments 1–4 of the paper. Writes `figures/*.pdf` and `results/summary.json`.

## Reproduce

    pip install -r requirements.txt
    python code/experiments.py
    python code/test_theorems.py
    python code/test_dynamics.py

`experiments.py` finishes in under a minute on a laptop. `test_theorems.py` takes about two minutes.

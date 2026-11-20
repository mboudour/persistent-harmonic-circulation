"""Checks of the dynamical statements on random graph/2-complex filtrations with large Betti numbers."""
import numpy as np
import itertools
from phc import Complex, inclusion, Phi, Lambda, simulate_forward, adiabatic_prediction, exact_rate

rng = np.random.default_rng(7)


def random_filtration(n=14, m=4, rng=rng):
    """Stage 0: random spanning tree + extra edges; later stages add vertices, edges, triangles."""
    V = list(range(n))
    perm = rng.permutation(n)
    E = set()
    for k in range(1, n):
        a, b = perm[k], perm[rng.integers(k)]
        E.add((min(a, b), max(a, b)))
    allpairs = [p for p in itertools.combinations(V, 2) if p not in E]
    rng.shuffle(allpairs)
    for p in allpairs[:5]:
        E.add(p)
    F = set()
    stages = [Complex(V, sorted(E), [])]
    nextv = n
    for s in range(m):
        # add a new vertex attached to two old vertices (new class, leaks), random edges, triangles
        a, b = rng.choice(V, 2, replace=False)
        V = V + [nextv]
        E |= {(int(a), nextv), (int(b), nextv)}
        nextv += 1
        cand = [p for p in itertools.combinations(V, 2) if p not in E]
        rng.shuffle(cand)
        for p in cand[:3]:
            E.add(p)
        tri = [t for t in itertools.combinations(V, 3)
               if all(q in E for q in itertools.combinations(t, 2)) and t not in F]
        rng.shuffle(tri)
        for t in tri[:2]:
            F.add(t)
        stages.append(Complex(V, sorted(E), sorted(F)))
    return stages


if __name__ == "__main__":
    beta = 0.6
    print("tau   max_err/bound   max_err")
    for trial in range(5):
        Ks = random_filtration()
        print("betti:", [K.betti1() for K in Ks], " ||Lambda||:", [round(np.linalg.norm(Lambda(Ks[i], Ks[i+1]), 2), 3) for i in range(len(Ks)-1)])
        x0 = rng.normal(size=Ks[0].nV)
        u0 = rng.normal(size=Ks[0].nE)
        xi = [None] + [rng.normal(size=Ks[i].nV - Ks[i-1].nV) for i in range(1, len(Ks))]
        zeta = [None] + [rng.normal(size=Ks[i].nE - Ks[i-1].nE) for i in range(1, len(Ks))]
        pred = adiabatic_prediction(Ks, u0, zeta)
        # dwell-time independent consensus value
        cpred = x0.sum()
        for i in range(1, len(Ks)):
            cpred += xi[i].sum()
        cpred /= Ks[-1].nV
        # bound B^2 = E_0 + sum injections
        B2 = np.sum((x0 - x0.mean()) ** 2) + np.sum(u0 ** 2)
        mean = x0.mean()
        for i in range(1, len(Ks)):
            B2 += np.sum(zeta[i] ** 2) + np.sum((xi[i] - mean) ** 2)
            mean = (mean * Ks[i-1].nV + xi[i].sum()) / Ks[i].nV
        B = np.sqrt(B2)
        for tau in [1, 2, 4, 8, 16, 32]:
            c, uinf, _ = simulate_forward(Ks, beta, [tau] * (len(Ks) - 1), x0, u0, xi, zeta)
            err = np.linalg.norm(uinf - pred)
            # crude bound with M=1 (energy contractivity): err <= B * sum ||Lambda_j||
            bound = B * sum(np.linalg.norm(Lambda(Ks[i], Ks[i+1]), 2) for i in range(len(Ks)-1))
            print(f"{tau:4d}  {err/bound:.3e}  {err:.3e}  consensus err {abs(c-cpred):.1e}")

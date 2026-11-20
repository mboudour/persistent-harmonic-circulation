"""Numerical checks of every algebraic statement in the paper on random filtrations."""
import numpy as np
import gudhi
from scipy.linalg import expm
from phc import Complex, inclusion, Phi, Lambda, persistent_harmonic_kernel, simulate_forward, \
    adiabatic_prediction, exact_rate, generator

rng = np.random.default_rng(1)


def rips_stage(st, r):
    V, E, F = set(), [], []
    for s, f in st.get_filtration():
        if f > r:
            continue
        if len(s) == 1:
            V.add(s[0])
        elif len(s) == 2:
            E.append(tuple(sorted(s)))
        elif len(s) == 3:
            F.append(tuple(sorted(s)))
    return Complex(sorted(V), E, F)


def ring(n, R, w, c, rng):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(-.05, .05, n)
    r = R + rng.uniform(-w, w, n)
    return np.c_[c[0] + r * np.cos(th), c[1] + r * np.sin(th)]


worst = {}
def rec(name, val):
    worst[name] = max(worst.get(name, 0), val)

nontrivial = dict(pairs=0, rank_pos=0, composition_pos=0, adjoint_pos=0, adiabatic_pos=0)
ok_rank = True
for trial in range(25):
    # two rings of random sizes; stages are taken inside the window in which classes are alive
    R2 = rng.uniform(0.35, 0.55)
    pts = np.r_[ring(int(rng.integers(20, 26)), 1.0, 0.06, (0, 0), rng),
                ring(int(rng.integers(10, 13)), R2, 0.03, (1.25 + R2 + rng.uniform(0, .1), 0), rng)]
    st = gudhi.RipsComplex(points=pts, max_edge_length=4.0).create_simplex_tree(max_dimension=2)
    st.compute_persistence(homology_coeff_field=11, persistence_dim_max=True)
    deaths = [d for dim, (b, d) in st.persistence() if dim == 1 and d - b > 0.05]
    radii = np.sort(np.unique([f for s, f in st.get_filtration() if len(s) == 2]))
    conn = [r for r in radii if (lambda K: K.nV == len(pts) and K.connected())(rips_stage(st, r))][:1]
    window = radii[(radii >= conn[0]) & (radii < max(deaths))]
    rs = list(window[np.linspace(0, len(window) - 1, 6).astype(int)])
    Ks = [rips_stage(st, r) for r in rs]
    for i in range(len(Ks) - 1):
        K, L = Ks[i], Ks[i + 1]
        _, I1 = inclusion(K, L)
        u = rng.normal(size=K.nE)
        PE, PH, PC = K.P_exact(), K.P_harm(), K.P_coexact()
        lhs = L.P_harm() @ I1 @ u
        rhs = Phi(K, L) @ u + Lambda(K, L) @ u
        rec('switching identity', np.abs(lhs - rhs).max())
        rec('coexact does not leak', np.abs(L.P_harm() @ I1 @ PC @ u).max())
        h = PH @ u
        ph = Phi(K, L) @ h
        rec('contraction violation', max(0, np.linalg.norm(ph) - np.linalg.norm(h)))
        # energy loss = ||P_B(L) iota h||^2
        loss = np.linalg.norm(I1 @ h) ** 2 - np.linalg.norm(ph) ** 2
        rec('energy-loss identity', abs(loss - np.linalg.norm(L.P_coexact() @ I1 @ h) ** 2))
    # rank of composite vs persistent Betti; kernel of persistent Laplacian
    for j in range(len(Ks)):
        for k in range(j + 1, len(Ks)):
            K, L = Ks[j], Ks[k]
            M = Phi(K, L)
            rk = np.linalg.matrix_rank(M, tol=1e-8)
            nontrivial["pairs"] += 1
            nontrivial["rank_pos"] += rk > 0
            pb = st.persistent_betti_numbers(rs[j], rs[k])[1]
            if rk != pb:
                ok_rank = False
            ker = persistent_harmonic_kernel(K, L)     # exact over Q (operator form is tested in test_suite)
            if ker.shape[1] != rk:
                ok_rank = False
            # ker Delta^{K,L} = H(K) ∩ (ker Phi)^perp  <=> P_ker = projector onto row space of M
            Q = np.linalg.svd(M)[2][:rk].T if rk else np.zeros((K.nE, 0))
            rec('ker persistent Laplacian = coimage of Phi', np.abs(ker @ ker.T - Q @ Q.T).max())
            # composition
            if k > j + 1:
                mat = K.P_harm()
                for t in range(j + 1, k + 1):
                    _, Ii = inclusion(Ks[t - 1], Ks[t])
                    mat = Ks[t].P_harm() @ Ii @ mat
                rec('composition Phi', np.abs(mat - M).max())
                nontrivial["composition_pos"] += rk > 0
    # adjoint (backward rule) between consecutive stages
    for i in range(len(Ks) - 1):
        K, L = Ks[i], Ks[i + 1]
        _, I1 = inclusion(K, L)
        fwd = L.P_harm() @ I1 @ K.P_harm()
        bwd = K.P_harm() @ I1.T @ L.P_harm()
        rec('backward = adjoint', np.abs(fwd.T - bwd).max())
        nontrivial["adjoint_pos"] += np.abs(fwd).max() > 1e-8
    # dynamics: residual vs adiabatic prediction + leakage bound
    beta = 0.7
    x0 = rng.normal(size=Ks[0].nV)
    u0 = rng.normal(size=Ks[0].nE)
    zeta = [None] + [rng.normal(size=Ks[i].nE - Ks[i - 1].nE) for i in range(1, len(Ks))]
    pred = adiabatic_prediction(Ks, u0, zeta)
    tau = 60.0
    c, uinf, _ = simulate_forward(Ks, beta, [tau] * (len(Ks) - 1), x0, u0, None, zeta)
    # Theorem 3 with the trivial gain bound m_K <= 1: ||u_inf - u_ad|| <= B * sum ||Lambda_i||
    B = np.sqrt(np.sum((x0 - x0.mean()) ** 2) + np.sum(u0 ** 2) + sum(np.sum(z ** 2) for z in zeta[1:]))
    bound = B * sum(np.linalg.norm(Lambda(Ks[i], Ks[i + 1]), 2) for i in range(len(Ks) - 1))
    rec('dwell-time bound violation', max(0.0, np.linalg.norm(uinf - pred) - bound))
    # Theorem 2(b): exact recursion at every switch, with the simulated exact components
    xx, uu = x0.copy(), u0.copy(); a = Ks[0].P_harm() @ uu
    for i in range(1, len(Ks)):
        Kp, K = Ks[i - 1], Ks[i]
        z = expm(tau * generator(Kp, beta)) @ np.r_[xx, uu]
        xx, uu = z[:Kp.nV], z[Kp.nV:]
        eps = Kp.P_exact() @ uu
        I0, I1 = inclusion(Kp, K)
        newe = [K.ei[e] for e in K.E if e not in Kp.ei]
        xx = I0 @ xx; uu = I1 @ uu; uu[newe] = zeta[i]
        inj = np.zeros(K.nE); inj[newe] = zeta[i]
        a = Phi(Kp, K) @ a + Lambda(Kp, K) @ eps + K.P_harm() @ inj
    rec('recursion error (Theorem 2)', np.abs(a - uinf).max())
    nontrivial["adiabatic_pos"] += np.linalg.norm(pred) > 1e-8
    rec('consensus value error', abs(c - x0.mean()))  # no new vertices -> mean conserved

for k, v in worst.items():
    print(f"{k:45s} {v:.2e}")
print("rank Phi == persistent Betti (GUDHI, F_11) == dim ker persistent Laplacian:", ok_rank)
assert ok_rank, "rank mismatch"
print("nontrivial cases:", nontrivial)
assert nontrivial["rank_pos"] >= nontrivial["pairs"] // 2, "too few pairs with a surviving class"
assert nontrivial["composition_pos"] > 0 and nontrivial["adjoint_pos"] > 0 and nontrivial["adiabatic_pos"] > 0
for k, v in worst.items():
    lim = 1e-9 if 'error' not in k else 1e-6
    assert v < lim, (k, v)
print("all assertions passed (absolute tolerance 1e-9; dynamics 1e-6)")

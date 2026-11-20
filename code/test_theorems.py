"""Numerical checks of every algebraic statement in the paper on random filtrations."""
import numpy as np
import gudhi
from phc import Complex, inclusion, Phi, Lambda, persistent_laplacian_1, simulate_forward, \
    adiabatic_prediction, exact_rate

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


def annulus(n, rng):
    th = rng.uniform(0, 2 * np.pi, n)
    rad = rng.uniform(0.8, 1.2, n)
    return np.c_[rad * np.cos(th), rad * np.sin(th)]


worst = {}
def rec(name, val):
    worst[name] = max(worst.get(name, 0), val)

ok_rank = True
for trial in range(25):
    pts = np.r_[annulus(18, rng), annulus(10, rng) * 0.45 + [2.4, 0]]
    st = gudhi.RipsComplex(points=pts, max_edge_length=3.0).create_simplex_tree(max_dimension=2)
    # radii: smallest connected one, then increasing
    radii = np.sort(np.unique([f for s, f in st.get_filtration() if len(s) == 2]))
    stages = []
    for r in radii:
        K = rips_stage(st, r)
        if K.nV == len(pts) and K.connected():
            stages.append((r, K))
    sel = [stages[k] for k in np.linspace(0, len(stages) - 1, 6).astype(int)]
    Ks = [K for _, K in sel]
    rs = [r for r, _ in sel]
    st.compute_persistence(homology_coeff_field=11, persistence_dim_max=True)
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
            pb = st.persistent_betti_numbers(rs[j], rs[k])[1]
            if rk != pb:
                ok_rank = False
            D = persistent_laplacian_1(K, L)
            w, Vv = np.linalg.eigh(D)
            ker = Vv[:, w < 1e-8]
            if ker.shape[1] != rk:
                ok_rank = False
            # ker Delta^{K,L} = H(K) ∩ (ker Phi)^perp  <=> P_ker = projector onto row space of M
            Q = np.linalg.svd(M)[2][:rk].T if rk else np.zeros((K.nE, 0))
            rec('ker persistent Laplacian = coimage of Phi', np.abs(ker @ ker.T - Q @ Q.T).max())
            # composition
            if k > j + 1:
                _, I1jk = inclusion(K, L)
                comp = np.eye(K.nE)
                cur = K
                mat = K.P_harm()
                for t in range(j + 1, k + 1):
                    _, Ii = inclusion(Ks[t - 1], Ks[t])
                    mat = Ks[t].P_harm() @ Ii @ mat
                rec('composition Phi', np.abs(mat - M).max())
    # adjoint (backward rule) between consecutive stages
    for i in range(len(Ks) - 1):
        K, L = Ks[i], Ks[i + 1]
        _, I1 = inclusion(K, L)
        fwd = L.P_harm() @ I1 @ K.P_harm()
        bwd = K.P_harm() @ I1.T @ L.P_harm()
        rec('backward = adjoint', np.abs(fwd.T - bwd).max())
    # dynamics: residual vs adiabatic prediction + leakage bound
    beta = 0.7
    x0 = rng.normal(size=Ks[0].nV)
    u0 = rng.normal(size=Ks[0].nE)
    zeta = [None] + [rng.normal(size=Ks[i].nE - Ks[i - 1].nE) for i in range(1, len(Ks))]
    pred = adiabatic_prediction(Ks, u0, zeta)
    for tau in [60.0]:
        c, uinf, _ = simulate_forward(Ks, beta, [tau] * (len(Ks) - 1), x0, u0, None, zeta)
        rec('adiabatic error at tau=60', np.abs(uinf - pred).max())
    rec('consensus value error', abs(c - x0.mean()))  # no new vertices -> mean conserved

for k, v in worst.items():
    print(f"{k:45s} {v:.2e}")
print("rank Phi == persistent Betti (GUDHI, F_11) == dim ker persistent Laplacian:", ok_rank)
assert ok_rank, "rank mismatch"
for k, v in worst.items():
    lim = 1e-9 if 'error' not in k else 1e-6
    assert v < lim, (k, v)
print("all assertions passed (absolute tolerance 1e-9; dynamics 1e-6)")

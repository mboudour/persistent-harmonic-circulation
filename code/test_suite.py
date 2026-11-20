"""Assertion-based checks of the statements in the paper. Run: python test_suite.py
Every check raises AssertionError on failure; tolerances are stated in TOL / RANK_TOL."""
import itertools
import numpy as np
import gudhi
import networkx as nx
from scipy.linalg import expm, null_space
from phc import (Complex, inclusion, generator, Phi, Lambda, persistent_laplacian_1,
                 persistent_harmonic_kernel, leakage_single_chord, leakage_new_path, simulate_forward,
                 adiabatic_prediction)

TOL = 1e-9          # tolerance for identities between floating-point matrices
RANK_TOL = 1e-8     # singular values below this are treated as zero
rng = np.random.default_rng(2026)
passed = []


def ok(name, cond):
    assert cond, name
    if name not in passed:
        passed.append(name)


def close(a, b, atol=TOL):
    """Absolute-tolerance comparison (rtol = 0)."""
    return np.allclose(a, b, rtol=0.0, atol=atol)


def rank(M):
    return 0 if M.size == 0 else int(np.sum(np.linalg.svd(M, compute_uv=False) > RANK_TOL))


def rips_stage(st, r):
    V, E, F = set(), [], []
    for s, f in st.get_filtration():
        if f > r:
            continue
        (V.add(s[0]) if len(s) == 1 else E.append(tuple(sorted(s))) if len(s) == 2
         else F.append(tuple(sorted(s))))
    return Complex(sorted(V), E, F)


def random_complex(n, p_edge, n_tri, rng):
    V = list(range(n))
    perm = rng.permutation(n)
    E = {tuple(sorted((int(perm[k]), int(perm[rng.integers(k)])))) for k in range(1, n)}
    E |= {p for p in itertools.combinations(V, 2) if rng.random() < p_edge}
    tri = [t for t in itertools.combinations(V, 3) if all(q in E for q in itertools.combinations(t, 2))]
    rng.shuffle(tri)
    return V, sorted(E), sorted(tri[:n_tri])


def main():
    # 1. persistent Laplacian: regression test and kernel identities --------------------------
    T = Complex([0, 1, 2], [(0, 1), (1, 2), (0, 2)], [(0, 1, 2)])
    ok("persistent Laplacian of filled triangle with K=L equals Delta_1 = 3I",
       close(np.linalg.eigvalsh(persistent_laplacian_1(T, T)), [3, 3, 3]))
    # regression: pendant edge added outside K; the K-part of B_1(L) is all of B_1(K) (no constraint)
    E5 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4)]
    F5 = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (1, 2, 4)]
    K5 = Complex(range(5), E5, F5); L5 = Complex(range(6), sorted(E5 + [(0, 5)]), F5)
    H5 = persistent_harmonic_kernel(K5, L5)
    D5 = persistent_laplacian_1(K5, L5)
    ok("regression: pendant edge gives empty persistent kernel (fast routine agrees with operator)",
       H5.shape[1] == 0 and np.linalg.eigvalsh(D5).min() > 1)
    # zero sector: single vertex
    from phc import exact_rate
    ok("zero exact sector: single vertex has alpha_E = infinity", exact_rate(Complex([0], []), 0.5)[0] == np.inf)

    for trial in range(30):
        V, E, F = random_complex(9, 0.35, 6, rng)
        L = Complex(V, E, F)
        keepE = [e for e in E if rng.random() < 0.75]
        # make K a connected subcomplex containing a spanning tree of L
        G = nx.Graph(); G.add_edges_from(E)
        keepE = sorted(set(keepE) | {tuple(sorted(e)) for e in nx.minimum_spanning_edges(G, data=False)})
        keepF = [f for f in F if all(tuple(sorted(q)) in keepE for q in itertools.combinations(f, 2)) and rng.random() < 0.5]
        K = Complex(V, keepE, keepF)
        D = persistent_laplacian_1(K, L)
        w, Vv = np.linalg.eigh(D)
        ker = Vv[:, w < RANK_TOL]
        ker2 = persistent_harmonic_kernel(K, L)
        M = Phi(K, L)
        r = rank(M)
        ok("dim ker Delta^{K,L} = rank Phi_{KL}", ker.shape[1] == r == ker2.shape[1])
        ok("kernel-equivalent routine has the same kernel", close(ker @ ker.T, ker2 @ ker2.T, atol=TOL))
        if ker.shape[1]:
            ok("fast kernel vectors satisfy the operator", close(D @ ker2, 0, atol=1e-8))
        Lp = Complex(V + [99], sorted(E + [(int(rng.integers(len(V))), 99)]), F)
        Dp = persistent_laplacian_1(K, Lp)
        wp, Vp = np.linalg.eigh(Dp)
        kp = Vp[:, wp < RANK_TOL]; kp2 = persistent_harmonic_kernel(K, Lp)
        ok("fast kernel = operator kernel after adding a pendant edge",
           kp.shape[1] == kp2.shape[1] and close(kp @ kp.T, kp2 @ kp2.T))
        Q = np.linalg.svd(M)[2][:r].T if r else np.zeros((K.nE, 0))
        ok("ker Delta^{K,L} = im Phi^* (coimage)", close(ker @ ker.T, Q @ Q.T, atol=TOL))
        # rank over R of the inclusion-induced map, computed homologically
        _, I1 = inclusion(K, L)
        ZK = null_space(K.B1, rcond=1e-12)
        BL = L.B2 if L.B2.shape[1] else np.zeros((L.nE, 0))
        r_hom = rank(np.hstack([I1 @ ZK, BL])) - rank(BL)
        ok("rank Phi = dim image of H_1(K;R) -> H_1(L;R)", r == r_hom)

    # 2. GUDHI persistent Betti numbers over F_11 and F_13 agree with real ranks ---------------
    for trial in range(8):
        th = rng.uniform(0, 2 * np.pi, 24)
        pts = np.c_[np.cos(th), np.sin(th)] * rng.uniform(0.85, 1.15, (24, 1))
        pts[12:] = pts[12:] * 0.4 + [2.0, 0]
        st = gudhi.RipsComplex(points=pts, max_edge_length=2.5).create_simplex_tree(max_dimension=2)
        radii = np.unique([f for s, f in st.get_filtration() if len(s) == 2])
        Ks, rs = [], []
        for rr in radii[::12]:
            Kr = rips_stage(st, rr)
            if Kr.nV == len(pts) and Kr.connected():
                Ks.append(Kr); rs.append(rr)
        Ks, rs = Ks[:6], rs[:6]
        for p in (11, 13):
            st.compute_persistence(homology_coeff_field=p, persistence_dim_max=True)
            for j in range(len(Ks)):
                for k in range(j, len(Ks)):
                    ok("GUDHI persistent Betti (F_11, F_13) = rank Phi over R",
                       st.persistent_betti_numbers(rs[j], rs[k])[1] == rank(Phi(Ks[j], Ks[k])))

    # 3. switching identities, transport properties, adjoint, removal --------------------------
    for trial in range(40):
        V, E, F = random_complex(10, 0.3, 8, rng)
        L = Complex(V, E, F)
        G = nx.Graph(); G.add_edges_from(E)
        tree = {tuple(sorted(e)) for e in nx.minimum_spanning_edges(G, data=False)}
        KE = sorted(tree | {e for e in E if rng.random() < 0.6})
        KF = [f for f in F if all(tuple(sorted(q)) in KE for q in itertools.combinations(f, 2)) and rng.random() < 0.6]
        K = Complex(V, KE, KF)
        _, I1 = inclusion(K, L)
        u = rng.normal(size=K.nE)
        ok("Theorem 1 switching identity",
           close(L.P_harm() @ I1 @ u, Phi(K, L) @ u + Lambda(K, L) @ u, atol=TOL))
        ok("coexact flow does not leak at inclusions", close(L.P_harm() @ I1 @ K.P_coexact() @ u, 0, atol=TOL))
        h = K.P_harm() @ u
        ok("Proposition 1(c) energy identity",
           abs(np.linalg.norm(h) ** 2 - np.linalg.norm(Phi(K, L) @ h) ** 2
               - np.linalg.norm(L.P_coexact() @ I1 @ h) ** 2) < TOL)
        g = rng.normal(size=L.nE)
        ok("Proposition 1(e) adjoint", close((L.P_harm() @ I1 @ K.P_harm()).T,
                                                     K.P_harm() @ I1.T @ L.P_harm(), atol=TOL))
        # Theorem 4: removal L -> K (restriction)
        rho = I1.T
        lhs = K.P_harm() @ rho @ g
        rhs = K.P_harm() @ rho @ L.P_harm() @ g + K.P_harm() @ rho @ L.P_coexact() @ g
        ok("Theorem 4 removal identity", close(lhs, rhs, atol=TOL))
        ok("exact flow does not leak at removals", close(K.P_harm() @ rho @ L.P_exact() @ g, 0, atol=TOL))
        # release criterion: Lambda_flat = 0 iff every h in H(K) has zero circulation on removed cells
        Lflat = K.P_harm() @ rho @ L.P_coexact()
        removed = [k for k, f in enumerate(L.F) if f not in set(K.F)]
        HK = K.harm_basis()
        circ = L.B2[:, removed].T @ I1 @ HK if removed else np.zeros((0, HK.shape[1]))
        ok("Theorem 4 release criterion", (np.abs(Lflat).max() < 1e-8) == (np.abs(circ).max(initial=0) < 1e-8))
        # Proposition 3(b): leakage criterion via net new-edge flow at old vertices
        HL = L.harm_basis()
        newE = [L.ei[e] for e in L.E if e not in K.ei]
        net = (L.B1[:, newE] @ HL[newE, :]) if newE else np.zeros((L.nV, HL.shape[1]))
        ok("Proposition 3(b) leakage criterion",
           (np.abs(Lambda(K, L)).max() < 1e-8) == (np.abs(net).max(initial=0) < 1e-8))

    # 4. leakage examples ---------------------------------------------------------------------
    K0 = Complex(['a', 'b'], [('a', 'b')])
    K1 = Complex(['a', 'b', 'c'], [('a', 'b'), ('b', 'c'), ('c', 'a')])
    ok("leakage through a path of new edges (no new edge has two old endpoints)",
       abs(np.linalg.norm(Lambda(K0, K1), 2) - np.sqrt(1 / 3)) < TOL)
    Esq = [(0, 1), (1, 2), (2, 3), (0, 3)]
    S0, S1 = Complex(range(4), Esq), Complex(range(4), Esq + [(0, 2)], [(0, 1, 2)])
    ok("leakage without a birth (H_1 isomorphism)",
       S0.betti1() == S1.betti1() == 1 and rank(Phi(S0, S1)) == 1 and np.linalg.norm(Lambda(S0, S1), 2) > 0.1)
    for trial in range(60):     # pendant trees attached at one old vertex never leak
        V, E, F = random_complex(8, 0.4, 4, rng)
        K = Complex(V, E, F)
        root = int(rng.integers(8)); new = [100 + k for k in range(4)]
        extra = [(root, new[0])] + [(new[int(rng.integers(k))], new[k]) for k in range(1, 4)]
        extra += [(new[0], new[2])]                      # a new cycle among new vertices
        L = Complex(V + new, E + extra, F)
        ok("Proposition 3(c): component meeting one old vertex does not leak", np.abs(Lambda(K, L)).max() < 1e-9)
        # single chord formula ||Lambda||^2 = R/(1+R)
        nonE = [p for p in itertools.combinations(V, 2) if p not in set(E)]
        if nonE:
            a, b = nonE[int(rng.integers(len(nonE)))]
            L2 = Complex(V, E + [(a, b)], F)
            val, R = leakage_single_chord(K, a, b)
            ok("Proposition (single chord): ||Lambda||^2 = R/(1+R)",
               abs(np.linalg.norm(Lambda(K, L2), 2) - val) < 1e-9)
            for ell in (2, 3, 5):
                mids = [200 + k for k in range(ell - 1)]
                chain = [a] + mids + [b]
                L3 = Complex(V + mids, E + [(chain[k], chain[k + 1]) for k in range(ell)], F)
                val3, _ = leakage_new_path(K, a, b, ell)
                ok("Proposition (new path): ||Lambda||^2 = R/(R+ell)",
                   abs(np.linalg.norm(Lambda(K, L3), 2) - val3) < 1e-9)

    # 5. dynamics: exact recursion with finite dwell times, consensus, bound ------------------
    from test_dynamics import random_filtration
    beta = 0.5
    for trial in range(6):
        Ks = random_filtration(rng=rng)
        x = rng.normal(size=Ks[0].nV); u = rng.normal(size=Ks[0].nE)
        xi = [None] + [rng.normal(size=Ks[i].nV - Ks[i - 1].nV) for i in range(1, len(Ks))]
        ze = [None] + [rng.normal(size=Ks[i].nE - Ks[i - 1].nE) for i in range(1, len(Ks))]
        taus = rng.uniform(0.5, 6, len(Ks) - 1)
        # recursion of Theorem 2 using the simulated exact components
        xx, uu = x.copy(), u.copy()
        a = Ks[0].P_harm() @ uu
        for i in range(1, len(Ks)):
            Kp, K = Ks[i - 1], Ks[i]
            z = expm(taus[i - 1] * generator(Kp, beta)) @ np.r_[xx, uu]
            xx, uu = z[:Kp.nV], z[Kp.nV:]
            eps = Kp.P_exact() @ uu
            I0, I1 = inclusion(Kp, K)
            newv = [K.vi[v] for v in K.V if v not in Kp.vi]; newe = [K.ei[e] for e in K.E if e not in Kp.ei]
            xx = I0 @ xx; xx[newv] = xi[i]
            uu = I1 @ uu; uu[newe] = ze[i]
            inj = np.zeros(K.nE); inj[newe] = ze[i]
            a = Phi(Kp, K) @ a + Lambda(Kp, K) @ eps + K.P_harm() @ inj
            ok("Theorem 2(b) exact recursion", close(a, K.P_harm() @ uu, atol=TOL))
        c, uinf, _ = simulate_forward(Ks, beta, list(taus), x, u, xi, ze)
        ok("Theorem 2(d) limit", close(uinf, a, atol=1e-8))
        cpred = (x.sum() + sum(v.sum() for v in xi[1:])) / Ks[-1].nV
        ok("Theorem 2(c) consensus value", abs(c - cpred) < 1e-9)

    # 6. zigzag theta example and one-turn lower bound -----------------------------------------
    def path(n, l):
        vs = ['s'] + [n + str(k) for k in range(1, l)] + ['t']
        return vs[1:-1], [(vs[k], vs[k + 1]) for k in range(l)]
    for lA, lB, lC in [(3, 3, 3), (2, 4, 3), (5, 1, 2)]:
        vA, eA = path('a', lA); vB, eB = path('b', lB); vC, eC = path('c', lC)
        Z0 = Complex(['s', 't'] + vA + vB, eA + eB); Z1 = Complex(['s', 't'] + vA + vB + vC, eA + eB + eC)
        Z2 = Complex(['s', 't'] + vB + vC, eB + eC)
        h = Z0.harm_basis()[:, 0]
        T = Phi(Z2, Z1).T @ Phi(Z0, Z1)
        _, I01 = inclusion(Z0, Z1)
        res = Z2.P_harm() @ inclusion(Z2, Z1)[1].T @ Z1.P_harm() @ I01 @ h
        ok("Example 1 residual formula", abs(np.linalg.norm(res) - lB / np.sqrt((lA + lB) * (lB + lC))) < TOL)
        f = Phi(Z0, Z1) @ Z0.harm_basis(); g = Phi(Z2, Z1) @ Z2.harm_basis()
        ok("Example 1 has no full bar", rank(f) + rank(g) - rank(np.c_[f, g]) == 0)

    # 7. release formula on a filled k-gon --------------------------------------------------------
    for k in (4, 8, 11):
        V = list(range(k)); E = [(j, j + 1) for j in range(k - 1)] + [(0, k - 1)]
        Kf, Ku = Complex(V, E, [tuple(V)]), Complex(V, E)
        w = Kf.B2[:, 0] / np.linalg.norm(Kf.B2[:, 0])
        for tau in (0.0, 0.3, 1.0):
            z = expm(tau * generator(Kf, beta)) @ np.r_[np.zeros(k), w]
            ok("Proposition 5 release formula", abs(np.linalg.norm(Ku.P_harm() @ z[k:]) - np.exp(-k * beta * tau)) < 1e-12)

    print(f"all {len(passed)} checks passed:")
    for p in passed:
        print("  -", p)


if __name__ == "__main__":
    main()

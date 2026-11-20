"""Core routines: cell complexes, Hodge decomposition, switched node-edge dynamics.

Model on each stage K_i (Boudourides, CSF 2027):
    x' = -d0^* u,   u' = -beta * Delta_1 u + d0 x.
Chains are identified with cochains through the Euclidean inner product,
so d0^* = B1 (vertex-edge boundary) and d1^* = B2 (edge-face boundary).
"""
import numpy as np
from scipy.linalg import expm, null_space, orth

TOL = 1e-10


class Complex:
    """Finite oriented 2-dimensional cell complex inside a fixed ambient complex.

    vertices: list of hashable labels; edges: list of (a, b) oriented a->b;
    faces: list of tuples of vertices giving the oriented boundary cycle.
    """

    def __init__(self, vertices, edges, faces=()):
        self.V = list(vertices)
        self.E = [tuple(e) for e in edges]
        self.F = [tuple(f) for f in faces]
        self.vi = {v: k for k, v in enumerate(self.V)}
        self.ei = {e: k for k, e in enumerate(self.E)}
        self.B1 = np.zeros((len(self.V), len(self.E)))
        for k, (a, b) in enumerate(self.E):
            self.B1[self.vi[a], k] -= 1.0
            self.B1[self.vi[b], k] += 1.0
        self.B2 = np.zeros((len(self.E), len(self.F)))
        for k, f in enumerate(self.F):
            n = len(f)
            for j in range(n):
                a, b = f[j], f[(j + 1) % n]
                if (a, b) in self.ei:
                    self.B2[self.ei[(a, b)], k] += 1.0
                elif (b, a) in self.ei:
                    self.B2[self.ei[(b, a)], k] -= 1.0
                else:
                    raise ValueError(f"face {f} uses missing edge {(a, b)}")
        assert np.allclose(self.B1 @ self.B2, 0)

    @property
    def nV(self):
        return len(self.V)

    @property
    def nE(self):
        return len(self.E)

    def L0(self):
        return self.B1 @ self.B1.T

    def L1(self):
        return self.B1.T @ self.B1 + self.B2 @ self.B2.T

    # orthogonal projectors of the Hodge decomposition of C^1
    def P_exact(self):
        Q = orth(self.B1.T) if self.nE else np.zeros((0, 0))
        return Q @ Q.T

    def P_coexact(self):
        if self.B2.shape[1] == 0:
            return np.zeros((self.nE, self.nE))
        Q = orth(self.B2)
        return Q @ Q.T if Q.size else np.zeros((self.nE, self.nE))

    def harm_basis(self):
        return null_space(self.L1(), rcond=1e-10)

    def P_harm(self):
        H = self.harm_basis()
        return H @ H.T

    def betti1(self):
        return self.harm_basis().shape[1]

    def connected(self):
        ev = np.linalg.eigvalsh(self.L0())
        return np.sum(ev < 1e-9) == 1


def inclusion(K, L):
    """Extension-by-zero matrices iota0 (C^0(K)->C^0(L)) and iota1 (C^1(K)->C^1(L)).
    Requires K to be a subcomplex of L with inherited orientations."""
    I0 = np.zeros((L.nV, K.nV))
    for v, k in K.vi.items():
        I0[L.vi[v], k] = 1.0
    I1 = np.zeros((L.nE, K.nE))
    for e, k in K.ei.items():
        I1[L.ei[e], k] = 1.0
    return I0, I1


def generator(K, beta):
    n, m = K.nV, K.nE
    A = np.zeros((n + m, n + m))
    A[:n, n:] = -K.B1
    A[n:, :n] = K.B1.T
    A[n:, n:] = -beta * K.L1()
    return A


def exact_rate(K, beta):
    """Decay rate of the exact sector and the critical-damping flag."""
    mu = np.linalg.eigvalsh(K.L0())
    mu = mu[mu > 1e-9]
    if mu.size == 0:            # single vertex: empty exact sector, convention alpha_E = infinity
        return np.inf, False
    r = []
    crit = False
    for m_ in mu:
        disc = beta ** 2 * m_ ** 2 - 4 * m_
        if abs(disc) < 1e-9:
            crit = True
            r.append(beta * m_ / 2)
        elif disc < 0:
            r.append(beta * m_ / 2)
        else:
            r.append((beta * m_ - np.sqrt(disc)) / 2)
    return min(r), crit


def Phi(K, L):
    """Harmonic transport H^1(K) -> H^1(L) in ambient coordinates of C^1(L):
    returns matrix acting on C^1(K) vectors: P_L iota P_K."""
    _, I1 = inclusion(K, L)
    return L.P_harm() @ I1 @ K.P_harm()


def Lambda(K, L):
    """Leakage operator P_L iota restricted to exact cochains of K."""
    _, I1 = inclusion(K, L)
    return L.P_harm() @ I1 @ K.P_exact()


def persistent_laplacian_1(K, L):
    """Persistent 1-Laplacian Delta_1^{K,L} on C^1(K) (Wang-Nguyen-Wei; Memoli-Wan-Wang):
        Delta = d0 d0^* + D D^*,  D = rho d1^* restricted to C_2^{L,K},
    where C_2^{L,K} = {w in C^2(L): d1^* w supported on edges of K}, with the
    Euclidean metric inherited from C^2(L)."""
    _, I1 = inclusion(K, L)
    outside = [k for k, e in enumerate(L.E) if e not in K.ei]
    nF = L.B2.shape[1]
    if nF == 0:
        D = np.zeros((K.nE, 0))
    else:
        N = null_space(L.B2[outside, :], rcond=1e-12) if outside else np.eye(nF)
        D = I1.T @ L.B2 @ N            # N: orthonormal basis of the constrained 2-chain domain
    return K.B1.T @ K.B1 + D @ D.T


def null_abs(M, tol=1e-10):
    """Orthonormal basis of ker M using an ABSOLUTE singular-value cutoff tol.
    (scipy's null_space uses a cutoff relative to the largest singular value,
    which treats round-off as a constraint when M is numerically zero.)"""
    M = np.atleast_2d(M)
    n = M.shape[1]
    if M.shape[0] == 0 or np.abs(M).max(initial=0.0) <= tol:
        return np.eye(n)
    _, sv, Vt = np.linalg.svd(M, full_matrices=True)
    r = int(np.sum(sv > tol))
    return Vt[r:].T


def persistent_harmonic_kernel(K, L, tol=1e-10, check=True):
    """Orthonormal basis of ker Delta_1^{K,L}, computed without forming the operator:
    ker Delta_1^{K,L} = Z_1(K) ∩ (B_1(L) ∩ C_1(K))^perp. Same kernel as persistent_laplacian_1,
    but uses only an orthonormal basis Q of B_1(L), so it scales to large Rips complexes.
    All cutoffs are absolute (Q is orthonormal, so singular values of its blocks lie in [0,1])."""
    _, I1 = inclusion(K, L)
    outside = [k for k, e in enumerate(L.E) if e not in K.ei]
    cols = [K.B1.T]
    Bup = np.zeros((K.nE, 0))
    if L.B2.shape[1]:
        U, sv, _ = np.linalg.svd(L.B2, full_matrices=False)
        Q = U[:, sv > tol * max(1.0, sv.max(initial=0.0))]      # orthonormal basis of B_1(L)
        if Q.size:
            N = null_abs(Q[outside, :], tol) if outside else np.eye(Q.shape[1])
            Bup = I1.T @ Q @ N                                   # basis of B_1(L) ∩ C_1(K)
            cols.append(Bup)
    H = null_abs(np.hstack(cols).T, tol)
    if check and H.size:
        # verify against the defining constraints: cycle in K, orthogonal to K-supported L-boundaries
        assert np.abs(K.B1 @ H).max() < 1e-8, "kernel vector is not a cycle"
        assert np.abs(Bup.T @ H).max(initial=0.0) < 1e-8, "kernel vector not orthogonal to boundaries"
    return H


def effective_resistance(K, a, b):
    Lp = np.linalg.pinv(K.L0())
    e = np.zeros(K.nV); e[K.vi[a]] = 1; e[K.vi[b]] = -1
    return float(e @ Lp @ e)


def leakage_new_path(K, a, b, ell):
    """Closed form for adding a path of ell new edges (ell-1 new vertices) between old a != b,
    lying on no 2-cell: ||Lambda||^2 = R/(R+ell), R = effective resistance in K."""
    R = effective_resistance(K, a, b)
    return np.sqrt(R / (R + ell)), R


def leakage_single_chord(K, a, b):
    return leakage_new_path(K, a, b, 1)


def simulate_forward(stages, beta, taus, x0, u0, xi=None, zeta=None):
    """Forward switched system. stages: list of Complex K_0 c ... c K_m.
    taus: dwell times of stages 0..m-1 (last stage runs to infinity).
    xi[i], zeta[i]: values on new vertices/edges when entering stage i (i>=1).
    Returns final limit (consensus value, u_inf) computed exactly, plus records."""
    x, u = np.array(x0, float), np.array(u0, float)
    rec = []
    for i, K in enumerate(stages):
        if i > 0:
            Kp = stages[i - 1]
            I0, I1 = inclusion(Kp, K)
            newv = [K.vi[v] for v in K.V if v not in Kp.vi]
            newe = [K.ei[e] for e in K.E if e not in Kp.ei]
            x = I0 @ x
            u = I1 @ u
            if xi is not None and xi[i] is not None:
                x[newv] = xi[i]
            if zeta is not None and zeta[i] is not None:
                u[newe] = zeta[i]
        rec.append(dict(stage=i, harm_in=K.P_harm() @ u))
        if i < len(stages) - 1:
            z = expm(taus[i] * generator(K, beta)) @ np.concatenate([x, u])
            x, u = z[:K.nV], z[K.nV:]
    K = stages[-1]
    return x.mean(), K.P_harm() @ u, rec


def adiabatic_prediction(stages, u0, zeta=None):
    """u_inf^ad = Phi_{0->m} P_0 u0 + sum_j Phi_{j->m} P_j zeta_j."""
    m = len(stages) - 1
    a = stages[0].P_harm() @ u0
    for i in range(1, m + 1):
        Kp, K = stages[i - 1], stages[i]
        _, I1 = inclusion(Kp, K)
        inj = np.zeros(K.nE)
        if zeta is not None and zeta[i] is not None:
            newe = [K.ei[e] for e in K.E if e not in Kp.ei]
            inj[newe] = zeta[i]
        a = K.P_harm() @ (I1 @ a + inj)
    return a

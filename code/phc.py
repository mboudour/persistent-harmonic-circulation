"""Core routines: cell complexes, Hodge decomposition, switched node-edge dynamics.

Model on each stage K_i (Boudourides, CSF 2027):
    x' = -d0^* u,   u' = -beta * Delta_1 u + d0 x.
Chains are identified with cochains through the Euclidean inner product,
so d0^* = B1 (vertex-edge boundary) and d1^* = B2 (edge-face boundary).
"""
import numpy as np
from scipy.linalg import expm, null_space, orth

TOL = 1e-10


def _fm(A):
    """FLINT integer matrix from an integer-valued numpy array (or pass through a FLINT matrix)."""
    import flint
    if isinstance(A, flint.fmpz_mat):
        return A
    A = np.asarray(A)
    Ai = np.rint(A).astype(np.int64)
    assert np.array_equal(Ai, A), "integer matrix required"
    return flint.fmpz_mat(Ai.tolist()) if Ai.size else flint.fmpz_mat(Ai.shape[0], Ai.shape[1])


def int_nullspace(A, n=None):
    """Exact basis (as a FLINT integer matrix, columns) of the rational null space of an integer matrix."""
    import flint
    M = _fm(A)
    if M.nrows() == 0:
        k = M.ncols() if n is None else n
        return flint.fmpz_mat([[int(i == j) for j in range(k)] for i in range(k)]) if k else flint.fmpz_mat(0, 0)
    X, k = M.nullspace()
    return flint.fmpz_mat([[X[i, j] for j in range(k)] for i in range(X.nrows())]) if k else flint.fmpz_mat(X.nrows(), 0)


def orthonormalize_exact_basis(X):
    """Orthonormal float basis of the column space of an exact integer basis X (FLINT, full column rank).
    The basis is first LLL-reduced (exactly, same lattice and hence same column space) to improve its
    conditioning; each column is then scaled exactly by its largest entry before conversion."""
    n, k = X.nrows(), X.ncols()
    if k == 0:
        return np.zeros((n, 0))
    try:
        X = X.transpose().lll().transpose()
    except Exception:
        pass
    rows = X.tolist()
    B = np.empty((n, k))
    for j in range(k):
        m = max(abs(int(rows[i][j])) for i in range(n)) or 1
        for i in range(n):
            B[i, j] = int(rows[i][j]) / m
    Q, _ = np.linalg.qr(B)
    Q, _ = np.linalg.qr(Q)          # second pass of Gram-Schmidt for orthogonality
    return Q


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

    # orthogonal projectors of the Hodge decomposition of C^1.
    # Dimensions are certified by exact integer arithmetic; only orthonormalization is floating point.
    def harm_basis(self):
        if not hasattr(self, "_H"):
            A = np.vstack([self.B1, self.B2.T]) if self.B2.shape[1] else self.B1
            self._H = orthonormalize_exact_basis(int_nullspace(A, self.nE))
        return self._H

    def P_harm(self):
        H = self.harm_basis()
        return H @ H.T

    def P_exact(self):
        if not hasattr(self, "_PE"):
            r = exact_rank(self.B1) if self.nE else 0
            if r == 0:
                self._PE = np.zeros((self.nE, self.nE))
            else:
                U = np.linalg.svd(self.B1.T, full_matrices=False)[0][:, :r]
                self._PE = U @ U.T
        return self._PE

    def P_coexact(self):
        return np.eye(self.nE) - self.P_exact() - self.P_harm()

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
    r, crit_k = [], []
    for m_ in mu:
        disc = beta ** 2 * m_ ** 2 - 4 * m_
        if disc >= 0:
            r.append(2 * m_ / (beta * m_ + np.sqrt(disc)))   # stable form of (beta*mu - sqrt(disc))/2
        else:
            r.append(beta * m_ / 2)
        crit_k.append(abs(disc) <= 1e-12 * (beta ** 2 * m_ ** 2 + 4 * m_))
    rmin = min(r)
    # the polynomial factor matters only if a slowest mode is (near-)critical
    crit = any(c for rk, c in zip(r, crit_k) if rk <= rmin * (1 + 1e-12))
    return rmin, crit


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
    where C_2^{L,K} = {w in C^2(L): d1^* w supported on edges of K}, with the Euclidean metric
    inherited from C^2(L). The domain C_2^{L,K} is obtained from an exact integer null space."""
    _, I1 = inclusion(K, L)
    outside = [k for k, e in enumerate(L.E) if e not in K.ei]
    nF = L.B2.shape[1]
    if nF == 0:
        D = np.zeros((K.nE, 0))
    else:
        N = orthonormalize_exact_basis(int_nullspace(L.B2[outside, :], nF)) if outside else np.eye(nF)
        D = I1.T @ L.B2 @ N
    return K.B1.T @ K.B1 + D @ D.T


P1, P2 = 2147483629, 2147483587      # two primes below 2^31


def exact_rank(M):
    """Certified rank over Q of an integer matrix, by exact integer arithmetic (FLINT fmpz_mat).
    (Ranks over finitely many prime fields are only lower bounds and are NOT used for decisions.)"""
    import flint
    A = np.rint(np.asarray(M)).astype(np.int64)
    if A.size == 0:
        return 0
    assert np.array_equal(A, np.asarray(M)), "exact_rank requires an integer matrix"
    return int(flint.fmpz_mat(A.tolist()).rank())


def null_abs(M, tol=1e-10):
    """Orthonormal basis of ker M using an ABSOLUTE singular-value cutoff tol."""
    M = np.atleast_2d(M)
    n = M.shape[1]
    if M.shape[0] == 0 or not np.any(M):
        return np.eye(n)
    _, sv, Vt = np.linalg.svd(M, full_matrices=True)
    return Vt[int(np.sum(sv > tol)):].T


def persistent_harmonic_kernel(K, L, check=True):
    """Orthonormal basis of ker Delta_1^{K,L} = Z_1(K) ∩ (B_1(L) ∩ C_1(K))^perp.
    The subspace is computed EXACTLY over Q with FLINT: an integer generating set of B_1(L) ∩ C_1(K)
    is rho d1^* W, with W an integer basis of ker(d1^* restricted to edges outside K); the kernel is the
    integer null space of [d0^*; (rho d1^* W)^T]. Only the final orthonormalization is floating point."""
    _, I1 = inclusion(K, L)
    outside = [k for k, e in enumerate(L.E) if e not in K.ei]
    rows = [_fm(K.B1)]
    nF = L.B2.shape[1]
    if nF:
        W = int_nullspace(L.B2[outside, :], nF) if outside else None
        B2K = _fm(I1.T @ L.B2)                              # rows of d1^* on edges of K
        Bup = B2K * W if W is not None else B2K             # integer generators of B_1(L) ∩ C_1(K)
        if Bup.ncols():
            rows.append(Bup.transpose())
    import flint
    M = rows[0]
    for R in rows[1:]:
        M = flint.fmpz_mat(M.tolist() + R.tolist())
    X = int_nullspace(M, K.nE)
    if check:
        assert (M * X).is_zero(), "exact kernel check failed"
        rB1 = exact_rank(K.B1)
        dim_bk = (exact_rank(L.B2) - (exact_rank(L.B2[outside, :]) if outside else 0)) if nF else 0
        assert X.ncols() == K.nE - rB1 - dim_bk, "kernel dimension differs from beta_1^{K,L}"
    return orthonormalize_exact_basis(X)


def effective_resistance(K, a, b):
    if not K.connected():
        raise ValueError("effective resistance requires a connected 1-skeleton")
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

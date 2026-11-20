"""Experiments and figures for 'Persistent Harmonic Circulation on Filtered Cell Complexes'.
Run: python3 experiments.py   (writes figures/*.pdf and results/*.json)
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gudhi
from scipy.linalg import expm
from phc import (Complex, inclusion, generator, exact_rate, Phi, Lambda,
                 persistent_laplacian_1, simulate_forward, adiabatic_prediction)

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6, "lines.linewidth": 1.1,
                     "font.family": "serif"})
BETA = 0.5
OUT = {}


def cycle(n, label=lambda k: k):
    V = [label(k) for k in range(n)]
    E = [(label(k), label((k + 1) % n)) if k < n - 1 else (label(0), label(n - 1)) for k in range(n)]
    return V, E


def exact_sector_gain(K, beta, tau):
    """m_K(tau): operator norm of z_ex(0) -> exact part of u(tau) on the exact sector."""
    A = generator(K, beta)
    n = K.nV
    PE = K.P_exact()
    # orthonormal basis of the exact sector {(x_perp, exact u)}
    Xp = np.eye(n) - np.ones((n, n)) / n
    from scipy.linalg import orth
    Q = orth(np.block([[Xp, np.zeros((n, K.nE))], [np.zeros((K.nE, n)), PE]]))
    T = expm(tau * A)
    Pout = np.c_[np.zeros((K.nE, n)), PE]
    return np.linalg.norm(Pout @ T @ Q, 2)


# ---------------------------------------------------------------------------
# Experiment 1: generation of circulation at a birth (chord on an 8-cycle)
# ---------------------------------------------------------------------------
V, E = cycle(8)
E = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (0, 7)]
K0 = Complex(V, E)
K1 = Complex(V, E + [(0, 4)])                       # chord: birth of a class across old vertices
K2 = Complex(V, E + [(0, 4)], [(0, 1, 2, 3, 4)])    # fill one half: death of a class
stagesA = [K0, K1, K2]
x0 = np.array([1.0, 0.6, 0.1, -0.4, -1.0, -0.5, 0.2, 0.7])
x0 -= x0.mean()
u0 = K0.P_exact() @ np.random.default_rng(3).normal(size=K0.nE)   # no harmonic content
assert np.linalg.norm(K0.P_harm() @ u0) < 1e-12
alpha0, crit0 = exact_rate(K0, BETA)
taus = np.linspace(0.25, 30, 240)
res = []
for t in taus:
    _, uinf, _ = simulate_forward(stagesA, BETA, [t, 200.0], x0, u0)
    res.append(np.linalg.norm(uinf))
res = np.array(res)
gain = np.array([exact_sector_gain(K0, BETA, t) for t in taus])
z0 = np.linalg.norm(np.r_[x0, u0])
lam01 = np.linalg.norm(Lambda(K0, K1), 2)
OUT["exp1"] = dict(betti=[K.betti1() for K in stagesA], alpha_ex=alpha0, critical=crit0,
                   Lambda_norm=lam01, nu_face=None,
                   max_ratio_res_over_bound=float(np.max(res / (lam01 * gain * z0))))
# without chord (only filling): no generation
K1b = Complex(V, E, [(0, 1, 2, 3, 4, 5, 6, 7)])
_, ub, _ = simulate_forward([K0, K1b], BETA, [3.0], x0, u0)
OUT["exp1"]["no_birth_residual"] = float(np.linalg.norm(ub))

# ---------------------------------------------------------------------------
# Experiment 2: dwell-time convergence on random 2-complex filtrations
# ---------------------------------------------------------------------------
from test_dynamics import random_filtration  # noqa: E402  (module guarded below)

rng = np.random.default_rng(11)
tau_grid = np.array([0.5, 1, 2, 4, 8, 12, 16, 24, 32, 48])
curves = []
for trial in range(6):
    Ks = random_filtration(rng=rng)
    x = rng.normal(size=Ks[0].nV)
    u = rng.normal(size=Ks[0].nE)
    xi = [None] + [rng.normal(size=Ks[i].nV - Ks[i - 1].nV) for i in range(1, len(Ks))]
    ze = [None] + [rng.normal(size=Ks[i].nE - Ks[i - 1].nE) for i in range(1, len(Ks))]
    pred = adiabatic_prediction(Ks, u, ze)
    B2 = np.sum((x - x.mean()) ** 2) + np.sum(u ** 2)
    mean = x.mean()
    for i in range(1, len(Ks)):
        B2 += np.sum(ze[i] ** 2) + np.sum((xi[i] - mean) ** 2)
        mean = (mean * Ks[i - 1].nV + xi[i].sum()) / Ks[i].nV
    errs, bnds = [], []
    lams = [np.linalg.norm(Lambda(Ks[i], Ks[i + 1]), 2) for i in range(len(Ks) - 1)]
    for t in tau_grid:
        _, uinf, _ = simulate_forward(Ks, BETA, [t] * (len(Ks) - 1), x, u, xi, ze)
        errs.append(np.linalg.norm(uinf - pred))
        bnds.append(np.sqrt(B2) * sum(lams[i] * exact_sector_gain(Ks[i], BETA, t)
                                       for i in range(len(Ks) - 1)))
    curves.append((np.array(errs), np.array(bnds), [K.betti1() for K in Ks]))
OUT["exp2"] = dict(max_err_over_bound=float(max(np.max(e / b) for e, b, _ in curves)),
                   betti=[c[2] for c in curves])

# ---------------------------------------------------------------------------
# Experiment 3: Rips filtration of a two-hole point cloud; residual and barcode
# ---------------------------------------------------------------------------
rng = np.random.default_rng(5)


def ring(n, R, w, c):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(-.05, .05, n)
    r = R + rng.uniform(-w, w, n)
    return np.c_[c[0] + r * np.cos(th), c[1] + r * np.sin(th)]


pts = np.r_[ring(36, 1.0, 0.06, (0, 0)), ring(14, 0.45, 0.03, (1.72, 0))]
st = gudhi.RipsComplex(points=pts, max_edge_length=4.0).create_simplex_tree(max_dimension=2)
st.compute_persistence(persistence_dim_max=True)
bars = [(b, d) for dim, (b, d) in st.persistence() if dim == 1]


def rips_stage(r):
    Vs, Es, Fs = set(), [], []
    for s, f in st.get_filtration():
        if f > r:
            continue
        if len(s) == 1:
            Vs.add(s[0])
        elif len(s) == 2:
            Es.append(tuple(sorted(s)))
        elif len(s) == 3:
            Fs.append(tuple(sorted(s)))
    return Complex(sorted(Vs), Es, Fs)


r_conn = 0.29          # 1-skeleton connected (merge scale 0.2523) and all three classes born
r_end = 1.8
radii = np.r_[np.linspace(r_conn, 0.4, 4), np.linspace(0.5, r_end, 12)]
Ks = [rips_stage(r) for r in radii]
betti = [K.betti1() for K in Ks]
x = rng.normal(size=Ks[0].nV)
u = rng.normal(size=Ks[0].nE)
ze = [None] + [np.zeros(Ks[i].nE - Ks[i - 1].nE) for i in range(1, len(Ks))]  # no injection
a = Ks[0].P_harm() @ u
harm_norm = [np.linalg.norm(a)]
for i in range(1, len(Ks)):
    _, I1 = inclusion(Ks[i - 1], Ks[i])
    a = Ks[i].P_harm() @ (I1 @ a)
    harm_norm.append(np.linalg.norm(a))
# persistent Laplacian check: ||u_inf^ad|| <= ||P_ker Delta^{K0,Kj} a0|| and nonvanishing equivalence
a0 = Ks[0].P_harm() @ u
plap = []
for j in range(len(Ks)):
    D = persistent_laplacian_1(Ks[0], Ks[j])
    w, Vv = np.linalg.eigh(D)
    Kr = Vv[:, w < 1e-8]
    plap.append(dict(r=float(radii[j]), dim_ker=int(Kr.shape[1]),
                     pers_betti=int(st.persistent_betti_numbers(radii[0], radii[j])[1]),
                     norm_proj=float(np.linalg.norm(Kr.T @ a0)), norm_transport=float(harm_norm[j])))
sim = {}
jlast = max(j for j in range(len(Ks)) if betti[j] > 0)      # last stage with a live class
pred_tr = adiabatic_prediction(Ks[:jlast + 1], u, ze[:jlast + 1])
for t in [8.0, 128.0, 512.0, 2048.0]:
    _, uinf, _ = simulate_forward(Ks[:jlast + 1], BETA, [t] * jlast, x, u, None, ze[:jlast + 1])
    sim[t] = dict(norm=float(np.linalg.norm(uinf)), err=float(np.linalg.norm(uinf - pred_tr)))
sim["pred_norm"] = float(np.linalg.norm(pred_tr)); sim["alpha_ex"] = [float(exact_rate(K, BETA)[0]) for K in Ks[:jlast]]; sim["r_last"] = float(radii[jlast])
OUT["exp3"] = dict(r_conn=float(r_conn), radii=radii.tolist(), betti=betti, harm_norm=harm_norm,
                   plap=plap, sim_residual_norm=sim, bars=[(float(b), float(d)) for b, d in bars])

# ---------------------------------------------------------------------------
# Experiment 4: zigzag theta graph and release of coexact flow by cell removal
# ---------------------------------------------------------------------------
# theta: endpoints s,t; three internally disjoint paths A,B,C of length 3
def path(name, s="s", t="t"):
    vs = [s, name + "1", name + "2", t]
    return vs[1:3], [(vs[k], vs[k + 1]) for k in range(3)]

vA, eA = path("a"); vB, eB = path("b"); vC, eC = path("c")
Z0 = Complex(["s", "t"] + vA + vB, eA + eB)
Z1 = Complex(["s", "t"] + vA + vB + vC, eA + eB + eC)
Z2 = Complex(["s", "t"] + vB + vC, eB + eC)
h0 = Z0.harm_basis()[:, 0]
_, I01 = inclusion(Z0, Z1)
_, I21 = inclusion(Z2, Z1)
f = Z1.P_harm() @ I01 @ Z0.P_harm()
g = Z1.P_harm() @ I21 @ Z2.P_harm()
pred = Z2.P_harm() @ I21.T @ (f @ h0)        # Phi_{0->1} then adjoint transport to Z2
# simulate: forward on Z0 (tau), switch to Z1, run tau, restrict to Z2, run to infinity
def run(K, z, t):
    return expm(t * generator(K, BETA)) @ z
xz = np.zeros(Z0.nV)
z = run(Z0, np.r_[xz, h0], 5.0)
I0, I1 = inclusion(Z0, Z1)
z = np.r_[I0 @ z[:Z0.nV], I1 @ z[Z0.nV:]]
z = run(Z1, z, 5.0)
I0b, I1b = inclusion(Z2, Z1)
z = np.r_[I0b.T @ z[:Z1.nV], I1b.T @ z[Z1.nV:]]
uZ = Z2.P_harm() @ z[Z2.nV:]
rank_fg = np.linalg.matrix_rank(np.c_[f @ Z0.harm_basis(), g @ Z2.harm_basis()], tol=1e-9)
full_bars = (np.linalg.matrix_rank(f, 1e-9) + np.linalg.matrix_rank(g, 1e-9) - rank_fg)
OUT["exp4"] = dict(residual_norm=float(np.linalg.norm(uZ)), predicted_norm=float(np.linalg.norm(pred)),
                   err=float(np.linalg.norm(uZ - pred)), full_bars=int(full_bars),
                   betti=[Z0.betti1(), Z1.betti1(), Z2.betti1()])
# release: filled octagon, coexact initial flow, remove the 2-cell at time tau
V8, E8 = list(range(8)), E
Kf = Complex(V8, E8, [tuple(range(8))])
Ku = Complex(V8, E8)
w = Kf.P_coexact() @ np.ones(8)
w = Kf.B2[:, 0] / np.linalg.norm(Kf.B2[:, 0])
nu = float(np.linalg.eigvalsh(Kf.B2.T @ Kf.B2)[0])
rel = []
tgrid = np.linspace(0, 1.2, 25)
for t in tgrid:
    zz = run(Kf, np.r_[np.zeros(8), w], t)
    rel.append(np.linalg.norm(Ku.P_harm() @ zz[8:]))
rel = np.array(rel)
OUT["exp4"]["release_max_err"] = float(np.max(np.abs(rel - np.exp(-BETA * nu * tgrid))))
OUT["exp4"]["nu"] = nu

with open("results/summary.json", "w") as fh:
    json.dump(OUT, fh, indent=1, default=float)
print(json.dumps(OUT, indent=1, default=float))

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
ax[0].semilogy(taus, res, color="k", label=r"$\|u_\infty\|$ (simulated)")
ax[0].semilogy(taus, lam01 * gain * z0, color="0.55", ls="--",
               label=r"$\|\Lambda_{0}\|\,m_{0}(\tau)\,\|z_0\|$")
ax[0].semilogy(taus, lam01 * z0 * np.exp(-alpha0 * taus) * 3, color="0.55", ls=":",
               label=rf"slope $-\alpha_{{\mathrm{{ex}}}}={-alpha0:.3f}$")
ax[0].set_xlabel(r"dwell time $\tau_0$ on $K_0$")
ax[0].set_ylim(1e-6, 3)
ax[0].legend(frameon=False, fontsize=7, loc="lower left")
ax[0].set_title("(a) circulation generated at a birth", fontsize=9)
for e, b, _ in curves:
    ax[1].loglog(tau_grid, e, color="k", alpha=0.7)
    ax[1].loglog(tau_grid, b, color="0.6", ls="--", alpha=0.7)
ax[1].plot([], [], color="k", label=r"$\|u_\infty-u_\infty^{\mathrm{ad}}\|$")
ax[1].plot([], [], color="0.6", ls="--", label="bound of Thm. 3")
ax[1].set_xlabel(r"common dwell time $\tau$")
ax[1].legend(frameon=False, fontsize=7, loc="lower left")
ax[1].set_title("(b) random filtrations, $m=4$", fontsize=9)
fig.tight_layout()
fig.savefig("figures/fig1_generation_and_dwell.pdf")

fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
bars_sorted = sorted(bars, key=lambda p: p[0])
for k, (b, d) in enumerate(bars_sorted):
    d_ = min(d, 2.4)
    ax[0].plot([b, d_], [k, k], color="k", lw=1.6)
ax[0].axvspan(radii[0], radii[-1], color="0.92", zorder=0)
ax[0].set_xlabel("Rips scale $r$")
ax[0].set_yticks([])
ax[0].set_title("(a) $H_1$ barcode; shaded: filtration window", fontsize=9)
ax[1].plot(radii, harm_norm, "k-o", ms=3, label=r"$\|\Phi_{0\to j}P_0u_0\|$")
ax[1].plot(radii, [p["norm_proj"] for p in plap], color="0.5", ls="--", marker="s", ms=3,
           label=r"$\|P_{\ker\Delta_1^{K_0,K_j}}P_0u_0\|$")
ax2 = ax[1].twinx()
ax2.step(radii, betti, where="post", color="0.7", lw=0.8)
ax2.set_ylabel(r"$\beta_1(K_j)$", color="0.5")
ax[1].set_xlabel("Rips scale $r_j$")
ax[1].legend(frameon=False, fontsize=7)
ax[1].set_title("(b) transported harmonic content", fontsize=9)
fig.tight_layout()
fig.savefig("figures/fig2_rips.pdf")
print("figures written")

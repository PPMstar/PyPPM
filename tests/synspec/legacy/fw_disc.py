"""Disc integration of the per-point FASTWIND profiles (shared functions).

Model of the disc-integrated line profile for an observer in direction n (unit vector):
    F(lambda) / F_c = sum_i w_i f_i(lambda_i') / sum_i w_i,   over the visible points (mu_i = r_i . n > 0),
    w_i = mu_i [1 - u_LD (1 - mu_i)] F_c,i,   lambda_i' = lambda (1 - v_i / c),  v_i = u_i . n
where f_i is the continuum-normalised flux profile of point i's FASTWIND model (at rest), F_c,i its
continuum flux, and v_i > 0 (towards the observer) gives a blueshift. The grid is equal-area, so every point
has the same dA. u_LD = 0 (default) is the simplest intensity law I(mu) = const: the only angle dependence is
the projected area mu dA (Lambert's cosine law). Using flux profiles as the local profiles is the usual first
approximation (e.g. CoMBiSpeC; SPAMMS uses FASTWIND I(mu) instead, Abdul-Masih et al. 2020).

All profiles are handled on a uniform velocity grid y = c ln(lambda / LREF) (1 km/s steps). Because every
model differs only in T_eff', the 1.24 million models form a profile library in T_eff' (library()): mean
profile and continuum flux in 10 K bins. With it, the disc integral is, per T_eff' bin, the rest profile
convolved with the (weighted) distribution of line-of-sight velocities (integrate_lib()); any dump's
T_eff' and velocities (sphere_sample.py) can be used. integrate_exact() sums the individual models of the run
itself, for validation.
"""
import os

import numpy as np
from scipy import fft as sfft
from scipy.signal import fftconvolve
from scipy.special import erf
from scipy import sparse

C_KMS = 299792.458
LINES = ["HEI4026", "HEII4200", "HEI4922"]
LABELS = [r"He I+II $\lambda$4026", r"He II $\lambda$4200", r"He I $\lambda$4922"]
LREF = np.array([4026.22, 4199.90, 4921.93])      # A (air), velocity zero points (CLAUDE.md line list)
DV = 1.0                                          # km/s, velocity grid step
VY = 2700.0                                       # km/s, half-width of the velocity grid (inside every model band)
Y = np.arange(-VY, VY + DV / 2, DV)
VSHIFT = 400                                      # km/s, largest |line-of-sight velocity| handled by the library method
RUN = "/scratch/ppathak/fastwind_sphere/d3200_r4050_N1236544"
SAMPLES = "/scratch/ppathak/fastwind_sphere/samples_r4050_N1236544"


def interp_rows(L, F, g):
    """Linear interpolation of every row (L[i], F[i]) onto g; rows of L increasing, constant beyond the ends."""
    n, k = L.shape
    off = (np.arange(n) * 1.0e5)[:, None]                  # separates the rows (span < 1e5)
    Lf = (L.astype(np.float64) + off).ravel()
    q = (g[None, :] + off).ravel()
    i = np.searchsorted(Lf, q)
    base = np.repeat(np.arange(n) * k, g.size)
    i = np.clip(i, base + 1, base + k - 1)
    x0, x1 = Lf[i - 1], Lf[i]
    y0, y1 = F.ravel()[i - 1].astype(np.float64), F.ravel()[i].astype(np.float64)
    w = np.clip((q - x0) / np.where(x1 > x0, x1 - x0, 1.0), 0.0, 1.0)
    return (y0 + w * (y1 - y0)).reshape(n, g.size)


def lam_of_y(j, y=Y):
    return LREF[j] * np.exp(y / C_KMS)


def load_run(run=RUN):
    """Per-point arrays of the per-point run: coordinates, T_eff', u_r, and the profiles (lam, fnorm, fcont)."""
    p = np.load(os.path.join(run, "profiles.npz"))
    return p


def library(run=RUN, dT=10.0, chunk=20_000):
    """Mean rest-frame profile (on Y) and continuum flux per T_eff' bin of width dT; cached in RUN/library_dT<dT>.npz."""
    cache = os.path.join(run, f"library_dT{dT:g}.npz")
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(os.path.join(run, "profiles.npz")):
        c = np.load(cache)
        return {k: c[k] for k in c.files}
    p = load_run(run)
    teff, lam, fn, fc = p["teff"], p["lam"], p["fnorm"], p["fcont"][:, :, 0]
    edges = np.arange(np.floor(teff.min() / dT) * dT, teff.max() + dT, dT)
    b = np.clip(np.digitize(teff, edges) - 1, 0, edges.size - 2)
    nb = edges.size - 1
    cnt = np.bincount(b, minlength=nb).astype(float)
    tsum = np.bincount(b, weights=teff, minlength=nb)
    prof = np.zeros((nb, 3, Y.size))
    fcs = np.zeros((nb, 3))
    for j in range(3):
        fcs[:, j] = np.bincount(b, weights=fc[:, j], minlength=nb)
        yl = C_KMS * np.log(lam[:, j].astype(np.float64) / LREF[j])
        for i0 in range(0, teff.size, chunk):
            f = interp_rows(yl[i0:i0 + chunk], fn[i0:i0 + chunk, j], Y)
            bb = b[i0:i0 + chunk]
            S = sparse.csr_matrix((np.ones(bb.size), (bb, np.arange(bb.size))), shape=(nb, bb.size))
            prof[:, j] += S @ f
    ok = cnt > 0
    prof[ok] /= cnt[ok, None, None]
    fcs[ok] /= cnt[ok, None]
    lib = dict(edges=edges, tmean=np.where(ok, tsum / np.maximum(cnt, 1), 0.5 * (edges[:-1] + edges[1:])),
               count=cnt, prof=prof.astype(np.float32), fc=fcs, dT=dT)
    # empty bins (sparse tails): take the nearest filled bin
    if not ok.all():
        filled = np.where(ok)[0]
        for i in np.where(~ok)[0]:
            k = filled[np.argmin(np.abs(filled - i))]
            lib["prof"][i], lib["fc"][i] = lib["prof"][k], lib["fc"][k]
    tmp = cache[:-4] + ".tmp.npz"
    np.savez(tmp, **lib)
    os.replace(tmp, cache)
    return lib


def unit_vectors(theta, phi):
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    rhat = np.stack([st * cp, st * sp, ct], axis=-1)
    that = np.stack([ct * cp, ct * sp, -st], axis=-1)
    phat = np.stack([-sp, cp, np.zeros_like(phi)], axis=-1)
    return rhat, that, phat


def directions(n):
    """n observer directions spread evenly over the sphere (Fibonacci), as unit vectors (n, 3)."""
    k = np.arange(n) + 0.5
    th = np.arccos(1.0 - 2.0 * k / n)
    ph = np.pi * (1.0 + 5 ** 0.5) * k
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=-1)


def los8():
    """The 8 lines of sight of Thompson et al. (2024, PPMstar M107; arXiv:2303.06125, Sect. 2.2), in the simulation
    frame: los1 = (1,1,1), los2 = los1 x (0,0,1), los3 = los1 x los2, los4 = los1 + los2 + los3 (each normalised
    before it is used for the next), los5..8 = -los1..-los4. Not aligned with the grid. -> (8, 3) unit vectors
    pointing from the star towards the observer."""
    n = lambda v: v / np.linalg.norm(v)
    l1 = n(np.array([1.0, 1.0, 1.0]))
    l2 = n(np.cross(l1, [0.0, 0.0, 1.0]))
    l3 = n(np.cross(l1, l2))
    l4 = n(l1 + l2 + l3)
    first = np.array([l1, l2, l3, l4])
    return np.concatenate([first, -first])


def mu_vlos(nvec, rhat, that, phat, ur, uth, uph):
    """mu = r.n and line-of-sight velocity v = u.n (km/s, > 0 towards the observer) of every point."""
    mu = rhat @ nvec
    v = ur * mu + uth * (that @ nvec) + uph * (phat @ nvec)
    return mu, v


def weights(mu, fc, uld=0.0):
    """Disc-integration weight mu [1 - uld (1 - mu)] F_c for visible points (0 elsewhere)."""
    return np.where(mu > 0, mu * (1.0 - uld * (1.0 - mu)), 0.0) * fc


def integrate_lib(lib, teff, v, w):
    """Normalised disc-integrated profiles (3, len(Y)) from the library: per T_eff' bin, the rest absorption
    depth convolved with the weighted histogram of line-of-sight velocities."""
    vis = w[..., 0] > 0 if w.ndim == 2 else w > 0
    b = np.clip(np.digitize(teff[vis], lib["edges"]) - 1, 0, lib["edges"].size - 2)
    s = np.clip(np.rint(-C_KMS * np.log(1.0 - v[vis] / C_KMS) / DV).astype(int), -VSHIFT, VSHIFT) + VSHIFT
    nb, nv = lib["edges"].size - 1, 2 * VSHIFT + 1
    out = np.zeros((3, Y.size))
    for j in range(3):
        wj = (w[vis, j] if w.ndim == 2 else w[vis])
        H = np.bincount(b * nv + s, weights=wj, minlength=nb * nv).reshape(nb, nv)
        use = np.where(H.sum(axis=1) > 0)[0]
        d = 1.0 - lib["prof"][use, j].astype(np.float64)                   # absorption depth, 0 at the grid ends
        conv = fftconvolve(d, H[use, ::-1], mode="full", axes=1)           # D(y) = sum_s H(s) d(y + s)
        D = conv[:, VSHIFT:VSHIFT + Y.size].sum(axis=0)
        out[j] = 1.0 - D / H.sum()
    return out


def integrate_exact(p, v, w, idx=None, chunk=20_000, lines=(0, 1, 2)):
    """Normalised disc-integrated profiles (3, len(Y)) summing the individual models (validation); only `lines` computed."""
    vis = np.where((w[:, 0] if w.ndim == 2 else w) > 0)[0] if idx is None else idx
    out = np.zeros((3, Y.size))
    for j in lines:
        wj = (w[:, j] if w.ndim == 2 else w)
        num = np.zeros(Y.size)
        for i0 in range(0, vis.size, chunk):
            ii = vis[i0:i0 + chunk]
            yl = C_KMS * np.log(p["lam"][ii, j].astype(np.float64) / LREF[j]) + C_KMS * np.log(1.0 - v[ii, None] / C_KMS)
            num += wj[ii] @ interp_rows(yl, p["fnorm"][ii, j], Y)
        out[j] = num / wj[vis].sum()
    return out


def diagnostics(F, j, vwin=400.0):
    """EW (A, whole grid; d lambda = lambda dy / c), and within |v| < vwin: centroid <v>, width sigma (km/s) of the
    absorption depth; FWHM (km/s) and central depth of the profile. vwin >= VY uses the whole grid: then <v> follows a
    Doppler shift exactly, while the default 400 km/s cuts the Stark wings of lambda4026/4200 (<v> responds only
    0.80/0.72 to a shift; used for the dump-3200 figures). Before 2026-09-29 the EW omitted the factor lambda/LREF
    (+9e-5 A for lambda4026, <= 1e-5 A for the others)."""
    d = 1.0 - F
    ew = np.trapz(d * np.exp(Y / C_KMS), Y) * LREF[j] / C_KMS
    m = np.abs(Y) <= vwin
    m0 = np.trapz(d[m], Y[m])
    v1 = np.trapz(Y[m] * d[m], Y[m]) / m0
    sig = np.sqrt(np.trapz((Y[m] - v1) ** 2 * d[m], Y[m]) / m0)
    k = np.argmax(d)
    half = d[k] / 2
    lo = k - np.argmax(d[k::-1] < half)
    hi = k + np.argmax(d[k:] < half)
    return dict(ew=ew, v1=v1, sigma=sig, fwhm=(hi - lo) * DV, depth=d[k])


# ---- broadening kernels on the velocity grid (unit area; Gray 2005) ----
def k_rot(vsini, eps=0.6):
    if vsini <= 0:
        return np.array([1.0])
    x = np.arange(-np.floor(vsini / DV), np.floor(vsini / DV) + 1) * DV / vsini
    g = 2 * (1 - eps) * np.sqrt(1 - x ** 2) + 0.5 * np.pi * eps * (1 - x ** 2)
    return g / g.sum()


def k_gauss(vmac, nsig=4.0):
    if vmac <= 0:
        return np.array([1.0])
    v = np.arange(-np.ceil(nsig * vmac / DV), np.ceil(nsig * vmac / DV) + 1) * DV
    g = np.exp(-(v / vmac) ** 2)
    return g / g.sum()


def k_rt(zeta, nsig=4.0):
    """Radial-tangential macroturbulence with equal radial and tangential parts (Gray 1975)."""
    if zeta <= 0:
        return np.array([1.0])
    x = np.abs(np.arange(-np.ceil(nsig * zeta / DV), np.ceil(nsig * zeta / DV) + 1) * DV / zeta)
    g = np.exp(-x ** 2) - np.sqrt(np.pi) * x * (1 - erf(x))
    return g / g.sum()


def broaden(F, *kernels):
    d = 1.0 - F
    for k in kernels:
        d = np.convolve(d, k, mode="same")
    return 1.0 - d


# ---- emergent intensities I(lambda, mu) (modified pformalsol, FW_10.6.4.1/v10.6_HHe_imu) ----
def read_imu(path):
    """OUT_IMU.<line>_VTV010 -> (lam (nk,), p (nray,), Ic (nk, nray), Il (nk, nray)): emergent continuum and line
    intensity of every ray of FASTWIND's formal solution, p = impact parameter in units of the inner-boundary radius."""
    with open(path) as f:
        f.readline()
        p = np.array(f.readline().split()[2:], dtype=float)
    d = np.loadtxt(path, comments="#")
    n = p.size
    return d[:, 1], p, d[:, 2:2 + n], d[:, 2 + n:2 + 2 * n]


def r_outer(p, Ic, frac=1e-3):
    """Outer radius of the emitting atmosphere: the smallest ray p beyond which the continuum intensity (at every
    wavelength point) stays below frac times its disc-centre value. Rays with p <= r_outer are mapped to the disc."""
    rel = (Ic / Ic[:, :1]).max(axis=0)
    beyond = np.where(rel < frac)[0]
    return float(p[beyond[0]]) if beyond.size else float(p[-1])


def flux_from_p(p, I):
    """Flux-like integral int I 2p dp over all rays (what FASTWIND does), trapezoidal. I: (nk, nray)."""
    return np.trapz(I * 2.0 * p, p, axis=1)


def imu_from_rays(p, I, rmax):
    """Intensity vs mu for the rays with p <= rmax: mu = sqrt(1 - (p/rmax)^2) (increasing order; mu = 0 at p = rmax).
    With this mapping int I 2mu dmu = int_0^rmax I 2p dp / rmax^2, so the flux of the 1D model is kept exactly.
    Returns (mu, I_mu (nk, nmu))."""
    inside = p <= rmax
    mu = np.sqrt(np.clip(1.0 - (p[inside] / rmax) ** 2, 0.0, 1.0))[::-1]
    return mu, I[:, inside][:, ::-1]


# ---- all dumps (fw_disc_dumps.py): T_eff' library interpolation with each dump's own velocities ----
# The per-point models differ only in T_eff', so the disc integral of any dump follows from the dump-3200 models: each
# point's rest profile is interpolated linearly in T_eff' between library nodes, and the dump's own velocities shift it.
DISC_DUMPS = "/scratch/ppathak/fastwind_sphere/disc_dumps_r4050_N1236544"
IMU_LIB = "/scratch/ppathak/fastwind_imu/imu_library_dT10.npz"


def lam_corrections(lib, runs="/scratch/ppathak/fastwind_imu/runs", imu_lib=IMU_LIB):
    """Per 10 K bin of the flux library: correction (nb, 3, ny) for the 0.01 A rounding of the wavelengths in FASTWIND's
    OUT files (from which profiles.npz and the library were built). For the bin's representative model (the intensity
    library's, rerun with the modified pformalsol) the difference between its OUT profile placed at the precise OUT_IMU
    wavelengths and at the rounded OUT wavelengths; the frequency grid is the same for all models of a bin (the core points
    of all models agree to <= 0.001 A), so every model of the bin shares this error. Cached in DISC_DUMPS/lamfix_dT10.npz."""
    cache = os.path.join(DISC_DUMPS, "lamfix_dT10.npz")
    if os.path.exists(cache):
        return np.load(cache)["corr"]
    L = np.load(imu_lib)
    assert np.allclose(L["edges"], lib["edges"])
    corr = np.zeros((lib["count"].size, 3, Y.size))
    for b in np.where(lib["count"] > 0)[0]:
        d = os.path.join(runs, f"P{int(L['idx_rep'][b]):06d}", f"P{int(L['idx_rep'][b]):06d}")
        for j, ln in enumerate(LINES):
            lr, f = np.genfromtxt(os.path.join(d, f"OUT.{ln}_VTV010"), usecols=[2, 4], max_rows=161).T
            lp = read_imu(os.path.join(d, f"OUT_IMU.{ln}_VTV010"))[0]
            assert np.abs(lp - lr).max() <= 0.0051, (b, ln)
            yr, yp = (C_KMS * np.log(x / LREF[j]) for x in (lr, lp))
            corr[b, j] = np.interp(Y, yp, f) - np.interp(Y, yr, f)
    os.makedirs(DISC_DUMPS, exist_ok=True)
    np.savez(cache[:-4] + ".tmp.npz", corr=corr)
    os.replace(cache[:-4] + ".tmp.npz", cache)
    return corr


def lib_nodes(lib, nmin=20, smooth=0.0, lamfix=False):
    """T_eff' interpolation nodes from the 10 K flux library (library()): consecutive filled bins are merged until a
    node holds >= nmin models (only the sparse tails are affected; a leftover at the hot end joins the last node).
    Node T_eff' = mean T_eff' of its models; node profile and F_c = means over its models.
    Variants (systematics of the FASTWIND outputs; see fw_disc_systematics.py):
      lamfix  add lam_corrections() per bin first (precise instead of 0.01 A-rounded wavelengths);
      smooth  W > 0: replace every node's profile and F_c by a local-linear fit in T_eff' over the nodes within +-W/2
              (weights = model counts); W = 335 K (one period of the EW(T_eff') sawtooth of the continuum sampling,
              Sect. fw_contfix) removes the sawtooth while keeping linear trends.
    -> dict(t (nn,), count (nn,), prof (nn, 3, ny) float64, fc (nn, 3))"""
    cnt = lib["count"]
    bprof = lib["prof"].astype(np.float64)
    if lamfix:
        bprof = bprof + lam_corrections(lib)
    groups, cur, c = [], [], 0.0
    for b in np.where(cnt > 0)[0]:
        cur.append(b)
        c += cnt[b]
        if c >= nmin:
            groups.append(cur)
            cur, c = [], 0.0
    if cur:
        if groups:
            groups[-1] = groups[-1] + cur
        else:
            groups.append(cur)
    nn = len(groups)
    t, n, fc = np.zeros(nn), np.zeros(nn), np.zeros((nn, 3))
    prof = np.zeros((nn, 3, Y.size))
    for i, g in enumerate(groups):
        w = cnt[g]
        n[i] = w.sum()
        t[i] = np.sum(w * lib["tmean"][g]) / n[i]
        prof[i] = np.tensordot(w, bprof[g], axes=1) / n[i]
        fc[i] = w @ lib["fc"][g] / n[i]
    if smooth > 0:
        ps, fs_ = np.empty_like(prof), np.empty_like(fc)
        for i in range(nn):
            k = np.where(np.abs(t - t[i]) <= smooth / 2)[0]
            w, dt = n[k], t[k] - t[i]
            S0, S1, S2 = w.sum(), w @ dt, w @ dt ** 2
            det = S0 * S2 - S1 ** 2
            if k.size < 3 or det <= 1e-12 * S0 * S2:
                ps[i], fs_[i] = prof[i], fc[i]
                continue
            c0, c1 = S2 / det, -S1 / det                         # local-linear value at t_i: sum_k (c0 + c1 dt_k) w_k y_k
            wk = w * (c0 + c1 * dt)
            ps[i] = np.tensordot(wk, prof[k], axes=1)
            fs_[i] = wk @ fc[k]
        prof, fc = ps, fs_
    return dict(t=t, count=n, prof=prof, fc=fc)


def node_pairs(tn, teff):
    """Linear interpolation in T_eff' between nodes tn (increasing): lower and upper node k0, k1 and the weight a of k1.
    Beyond the end nodes the end node is used (clamped)."""
    k0 = np.clip(np.searchsorted(tn, teff, side="right") - 1, 0, tn.size - 2)
    a = np.clip((teff - tn[k0]) / (tn[k0 + 1] - tn[k0]), 0.0, 1.0)
    return k0, k0 + 1, a


def shift_steps(v, vshift=VSHIFT):
    """Doppler shift of v (km/s, > 0 towards the observer) in grid steps, rounded, and whether |shift| > vshift (clipped)."""
    s = np.rint(-C_KMS * np.log(1.0 - v / C_KMS) / DV).astype(np.int64)
    return np.clip(s, -vshift, vshift), np.abs(s) > vshift


class DiscFlux:
    """Flux method with T_eff' interpolation: point i contributes mu_i F_line,i(lambda (1 - v_i/c)), with the line flux
    F_line = F_c f interpolated linearly in T_eff' between library nodes k0, k1 (weights 1 - a, a), and F_c likewise:
        F / F_c = 1 - sum_k F_c,k sum_s H_k(s) d_k(y + s) / sum_k F_c,k H0_k,   d = 1 - f,
    H_k(s) = sum of mu (1 - a) resp. mu a over the visible points of node k with Doppler shift s (1 km/s steps),
    H0_k = sum_s H_k(s). The convolutions use FFTs of the (fixed) library, computed once."""

    def __init__(self, nodes, vshift=VSHIFT):
        self.t, self.fc = nodes["t"], nodes["fc"]
        self.nn, self.vs, self.nv = self.t.size, vshift, 2 * vshift + 1
        self.L = sfft.next_fast_len(Y.size + self.nv - 1, real=True)
        self.P = np.ascontiguousarray(np.transpose(self.fc[:, :, None] * nodes["prof"], (1, 0, 2)))      # (3, nn, ny)
        d = np.transpose(self.fc[:, :, None] * (1.0 - nodes["prof"]), (1, 0, 2))
        self.Dhat = sfft.rfft(d, n=self.L, axis=-1)                                                     # (3, nn, nf)

    def pairs(self, teff):
        return node_pairs(self.t, teff)

    def __call__(self, mu, v, k0, k1, a, novel=True):
        """-> F, F0 (3, ny) with and without Doppler shifts, continuum-weighted mean and rms of v (3,), clipped points."""
        vis = mu > 0
        m, v, k0, k1, a = mu[vis], v[vis], k0[vis], k1[vis], a[vis]
        s, clip = shift_steps(v, self.vs)
        H = np.bincount(np.concatenate([k0, k1]) * self.nv + np.concatenate([s, s]) + self.vs,
                        weights=np.concatenate([m * (1.0 - a), m * a]), minlength=self.nn * self.nv).reshape(self.nn, self.nv)
        H0 = H.sum(axis=1)
        den = self.fc.T @ H0                                                                             # (3,)
        F0 = np.einsum("k,jky->jy", H0, self.P) / den[:, None] if novel else None
        Hhat = sfft.rfft(H[:, ::-1], n=self.L, axis=1)
        D = sfft.irfft(np.einsum("jkf,kf->jf", self.Dhat, Hhat), n=self.L, axis=-1)[:, self.vs:self.vs + Y.size]
        F = 1.0 - D / den[:, None]
        w = m[:, None] * ((1.0 - a)[:, None] * self.fc[k0] + a[:, None] * self.fc[k1])                 # (nvis, 3)
        vm = (w * v[:, None]).sum(axis=0) / w.sum(axis=0)
        sd = np.sqrt((w * (v[:, None] - vm) ** 2).sum(axis=0) / w.sum(axis=0))
        return F, F0, vm, sd, int(clip.sum())


class DiscImu:
    """Intensity method (SPAMMS approach, as fw_disc_imu.py) with T_eff' interpolation: point i contributes
    mu_i I(lambda (1 - v_i/c), mu_i) of its T_eff', interpolated linearly in T_eff' between the representative models of
    the intensity library (fw_imu_library.py) and linearly in s = sqrt(1 - mu^2) between their rays:
        F / F_c = sum mu I_l(shifted) / sum mu I_c(shifted).
    Rows of the velocity histogram are (T_eff' node, ray node) pairs; intensities are edge-padded before the FFT
    convolution (as fw_disc_imu.py), so the grid ends are handled like there. float64 throughout (~8 GB of library
    FFTs, shared by forked workers)."""
    CHUNK = 128                                                 # histogram rows per FFT block

    def __init__(self, path=IMU_LIB, vshift=VSHIFT):
        lib = np.load(path)
        u = np.unique(lib["src"])                               # bins with their own representative model
        self.t = lib["teff_rep"][u]
        assert np.all(np.diff(self.t) > 0)
        self.nn, self.K = u.size, lib["s"].shape[2]
        self.vs, self.nv = vshift, 2 * vshift + 1
        S = lib["s"][u].copy()                                  # (nn, 3, K), NaN beyond nnode
        self.nnode = lib["nnode"][u]
        pad = 1.0 + (1.0 + np.arange(self.K)) / (self.K + 1)   # in (1, 2): beyond s <= 1, inside the row's band of 10
        S = np.where(np.isnan(S), pad[None, None, :], S)
        self.Sflat = [(S[:, j, :] + 10.0 * np.arange(self.nn)[:, None]).ravel() for j in range(3)]
        assert all(np.all(np.diff(x) > 0) for x in self.Sflat), "ray nodes not increasing"
        ny = Y.size
        self.L = sfft.next_fast_len(ny + 2 * vshift + self.nv - 1, real=True)
        self.I0, self.Ihat = [], []
        for j in range(3):
            Il = lib["Il"][u, j].reshape(self.nn * self.K, ny).astype(np.float64)
            Ic = lib["Ic"][u, j].reshape(self.nn * self.K, ny).astype(np.float64)
            self.I0.append((Il, Ic))
            self.Ihat.append(tuple(sfft.rfft(np.pad(A, ((0, 0), (vshift, vshift)), mode="edge"), n=self.L, axis=1)
                                   for A in (Il, Ic)))
        self.ic0 = [self.I0[j][1][:, ny // 2] for j in range(3)]

    def pairs(self, teff):
        return node_pairs(self.t, teff)

    def _rays(self, i, s, j):
        """Lower ray node kk and fraction tt (linear in s) of points with T_eff' node i."""
        sf = self.Sflat[j]
        g = np.searchsorted(sf, s + 10.0 * i, side="right") - 1 - i * self.K
        kk = np.clip(g, 0, self.nnode[i, j] - 2)
        x0, x1 = sf[i * self.K + kk] - 10.0 * i, sf[i * self.K + kk + 1] - 10.0 * i
        return kk, np.clip((s - x0) / (x1 - x0), 0.0, 1.0)

    def __call__(self, mu, v, k0, k1, a, novel=True, lines=(0, 1, 2)):
        """-> F, F0 (3, ny), mean and rms of v (weight mu I_c at the line centre) (3,), clipped points."""
        vis = mu > 0
        m, v, k0, k1, a = mu[vis], v[vis], k0[vis], k1[vis], a[vis]
        s = np.sqrt(np.clip(1.0 - m ** 2, 0.0, 1.0))
        sh, clip = shift_steps(v, self.vs)
        ny, nr = Y.size, self.nn * self.K
        F, F0 = np.full((3, ny), np.nan), np.full((3, ny), np.nan)
        vm, sd = np.full(3, np.nan), np.full(3, np.nan)
        vv = np.tile(v, 4)
        for j in lines:
            kA, tA = self._rays(k0, s, j)
            kB, tB = self._rays(k1, s, j)
            rows = np.concatenate([k0 * self.K + kA, k0 * self.K + kA + 1, k1 * self.K + kB, k1 * self.K + kB + 1])
            wts = np.concatenate([m * (1 - a) * (1 - tA), m * (1 - a) * tA, m * a * (1 - tB), m * a * tB])
            H = np.bincount(rows * self.nv + np.tile(sh, 4) + self.vs, weights=wts, minlength=nr * self.nv).reshape(nr, self.nv)
            H0 = H.sum(axis=1)
            Il, Ic = self.I0[j]
            if novel:
                F0[j] = (H0 @ Il) / (H0 @ Ic)
            nf = self.L // 2 + 1
            acc_l, acc_c = np.zeros(nf, complex), np.zeros(nf, complex)
            used = np.where(H0 > 0)[0]
            for r0 in range(0, used.size, self.CHUNK):               # small blocks: no huge temporaries (page faults)
                rr = used[r0:r0 + self.CHUNK]
                Hhat = sfft.rfft(H[rr, ::-1], n=self.L, axis=1)
                acc_l += np.einsum("kf,kf->f", self.Ihat[j][0][rr], Hhat)
                acc_c += np.einsum("kf,kf->f", self.Ihat[j][1][rr], Hhat)
            num = sfft.irfft(acc_l, n=self.L)[2 * self.vs:2 * self.vs + ny]
            den = sfft.irfft(acc_c, n=self.L)[2 * self.vs:2 * self.vs + ny]
            F[j] = num / den
            wp = wts * self.ic0[j][rows]                        # weight of v: mu I_c(mu) at the line centre
            vm[j] = np.sum(wp * vv) / wp.sum()
            sd[j] = np.sqrt(np.sum(wp * (vv - vm[j]) ** 2) / wp.sum())
        return F, F0, vm, sd, int(clip.sum())

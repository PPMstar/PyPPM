"""Disc-integrated line profiles of the dump-3200 per-point run for the 8 lines of sight of Thompson et al. (2024),
plus the T_eff' profile library.

Every point's own FASTWIND model is used (no library, no interpolation between models). Per line, one weight matrix M
(sparse, rows x points) is built first, with rows for
  * each line of sight n (fw_disc.los8) and each 1 km/s group of line-of-sight velocities v = u.n (sphere_sample.py):
    the visible points (mu = r.n > 0) with weight mu F_c (Lambert's cosine law, I(mu) = const);
  * each line of sight without velocities (same weights; the template for the broadening analysis);
  * each 10 K bin of T_eff' (weight 1: the library);
  * the 1 km/s groups of a random subset of los1 (check a).
The rest-frame profiles are then streamed: worker processes interpolate blocks of profiles onto the common velocity
grid fw_disc.Y and add M[:, block] @ P_block (float64) to their partial sums, so the profiles are never all held in
memory (a few GB in total; runs on the login node with 4 workers). Finally each velocity group's summed profile is
shifted once (lambda_obs = lambda (1 - v/c), v > 0 towards the observer = blueshift) and the groups are added.
Checks: (a) 1 km/s rounding vs the continuous calculation (fw_disc.integrate_exact) on the subset; (b) the library
method (fw_disc.integrate_lib) vs this exact sum, for los1.
Output: RUN/disc_los8.npz (Y, LREF, los, F, F0 (8, 3, ny), diag_F, diag_F0, vmean_w, sigma_w, checks) and
RUN/library_dT10.npz (the fw_disc.library cache).

Usage:  ./run.sh fw_disc_los.py --nproc 4                      (login node, ~20 min)
        ./run.sh fw_disc_los.py --nproc 4 --nmax 20000 --nsub 2000   smoke test (writes *_test.npz)
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy import sparse

import fw_disc as fd

ap = argparse.ArgumentParser()
ap.add_argument("--nproc", type=int, default=4)
ap.add_argument("--dT", type=float, default=10.0)
ap.add_argument("--nsub", type=int, default=20000)
ap.add_argument("--block", type=int, default=5000)
ap.add_argument("--nmax", type=int, default=0, help="test: use only the first NMAX points (outputs *_test.npz)")
ap.add_argument("--shm", default=None, help="ignored (kept for the old job scripts)")
a = ap.parse_args()
SUFFIX = "_test" if a.nmax else ""
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f} s] {msg}", flush=True)


# ---- data (module level, shared with the forked workers) ----
p = np.load(os.path.join(fd.RUN, "profiles.npz"))
NS = slice(0, a.nmax) if a.nmax else slice(None)
teff, theta, phi = p["teff"][NS], p["theta"][NS], p["phi"][NS]
N = teff.size
s = {k: v[NS] for k, v in np.load(os.path.join(fd.SAMPLES, "d3200.npz")).items() if np.ndim(v)}
lam_all, fn_all, fc_all = p["lam"][NS], p["fnorm"][NS], p["fcont"][NS, :, 0].astype(np.float64)
rhat, that, phat = fd.unit_vectors(theta, phi)
los = fd.los8()
MU, V = np.zeros((8, N)), np.zeros((8, N))
for k in range(8):
    MU[k], V[k] = fd.mu_vlos(los[k], rhat, that, phat, s["ur"], s["uth"], s["uph"])
del rhat, that, phat
SH = np.rint(-fd.C_KMS * np.log(1.0 - V / fd.C_KMS) / fd.DV).astype(np.int64)     # shift in grid steps
edges = np.arange(np.floor(teff.min() / a.dT) * a.dT, teff.max() + a.dT, a.dT)
bins = np.clip(np.digitize(teff, edges) - 1, 0, edges.size - 2)
nb = edges.size - 1
cnt = np.bincount(bins, minlength=nb).astype(float)
sub = np.sort(np.random.default_rng(1).choice(np.where(MU[0] > 0)[0], min(a.nsub, int((MU[0] > 0).sum())), replace=False))
ny = fd.Y.size
MC = None                     # the weight matrix of the current line (CSC), set before the pool is forked
JLINE = 0


def stream(worker):
    """Partial M @ P over this worker's blocks of points (P = rest profiles on fd.Y, computed block by block)."""
    A = np.zeros((MC.shape[0], ny))
    for i0 in range(worker * a.block, N, a.nproc * a.block):
        i1 = min(i0 + a.block, N)
        blk = MC[:, i0:i1]
        if blk.nnz == 0:
            continue
        yl = fd.C_KMS * np.log(lam_all[i0:i1, JLINE].astype(np.float64) / fd.LREF[JLINE])
        A += blk @ fd.interp_rows(yl, fn_all[i0:i1, JLINE], fd.Y)
    return A


def shift_add(Arows, Wrows, shifts):
    """sum over velocity groups of the weighted absorption depth, each shifted by its group velocity."""
    depth = np.zeros(ny)
    for g, sh in enumerate(shifts):
        dg = Wrows[g] - Arows[g]
        if sh >= 0:
            depth[:ny - sh] += dg[sh:]
        else:
            depth[-sh:] += dg[:ny + sh]
    return depth


log(f"{N} points, 8 lines of sight, {nb} library bins, {sub.size} check points; v_los {V.min():.1f} .. {V.max():.1f} km/s; "
    f"{a.nproc} workers")
F, F0 = np.zeros((8, 3, ny)), np.zeros((8, 3, ny))
lib_prof, lib_fc = np.zeros((nb, 3, ny)), np.zeros((nb, 3))
checks = {}
for j in range(3):
    JLINE = j
    rows, cols, vals, groups = [], [], [], []
    r0 = 0
    for k in range(8):                                   # velocity groups per line of sight
        w = fd.weights(MU[k], fc_all[:, j])
        vis = np.where(w > 0)[0]
        sv, gi = np.unique(SH[k, vis], return_inverse=True)
        rows.append(r0 + gi); cols.append(vis); vals.append(w[vis])
        groups.append((r0, sv))
        r0 += sv.size
    r_novel = r0
    for k in range(8):                                   # no velocities
        w = fd.weights(MU[k], fc_all[:, j])
        vis = np.where(w > 0)[0]
        rows.append(np.full(vis.size, r0 + k)); cols.append(vis); vals.append(w[vis])
    r0 += 8
    r_lib = r0
    rows.append(r0 + bins); cols.append(np.arange(N)); vals.append(np.ones(N))
    r0 += nb
    r_sub = r0                                           # check (a): velocity groups of the los1 subset
    w1 = fd.weights(MU[0], fc_all[:, j])
    sv_sub, gi_sub = np.unique(SH[0, sub], return_inverse=True)
    rows.append(r0 + gi_sub); cols.append(sub); vals.append(w1[sub])
    r0 += sv_sub.size
    M = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r0, N))
    W = np.asarray(M.sum(axis=1)).ravel()
    MC = M.tocsc()
    with Pool(a.nproc) as pool:                          # forked after MC/JLINE are set
        A = sum(pool.map(stream, range(a.nproc)))
    log(f"{fd.LINES[j]}: streamed {N} profiles through {r0} weight rows")
    for k in range(8):
        g0, sv = groups[k]
        F[k, j] = 1.0 - shift_add(A[g0:g0 + sv.size], W[g0:g0 + sv.size], sv) / W[g0:g0 + sv.size].sum()
        F0[k, j] = A[r_novel + k] / W[r_novel + k]
    ok = cnt > 0
    lib_prof[ok, j] = A[r_lib:r_lib + nb][ok] / cnt[ok, None]
    lib_fc[:, j] = np.bincount(bins, weights=fc_all[:, j], minlength=nb) / np.maximum(cnt, 1)
    Fr_sub = 1.0 - shift_add(A[r_sub:], W[r_sub:], sv_sub) / W[r_sub:].sum()
    Fc_sub = fd.integrate_exact({"lam": lam_all, "fnorm": fn_all}, V[0], np.repeat(w1[:, None], 3, axis=1), idx=sub, lines=(j,))[j]
    checks[f"round_{fd.LINES[j]}"] = float(np.abs(Fr_sub - Fc_sub).max())
    log(f"{fd.LINES[j]} check (a): 1 km/s rounding vs continuous, {sub.size} points of los1: max |dF| {checks[f'round_{fd.LINES[j]}']:.1e}")
    del A, M, MC

# library (same format as the fw_disc.library cache); empty bins take the nearest filled bin
ok = cnt > 0
filled = np.where(ok)[0]
for i in np.where(~ok)[0]:
    kk = filled[np.argmin(np.abs(filled - i))]
    lib_prof[i], lib_fc[i] = lib_prof[kk], lib_fc[kk]
tsum = np.bincount(bins, weights=teff, minlength=nb)
lib = dict(edges=edges, tmean=np.where(ok, tsum / np.maximum(cnt, 1), 0.5 * (edges[:-1] + edges[1:])),
           count=cnt, prof=lib_prof.astype(np.float32), fc=lib_fc, dT=a.dT)
libpath = os.path.join(fd.RUN, f"library_dT{a.dT:g}{SUFFIX}.npz")
np.savez(libpath[:-4] + ".tmp.npz", **lib)
os.replace(libpath[:-4] + ".tmp.npz", libpath)
wl = fd.weights(MU[0], 1.0)[:, None] * lib_fc[bins]
Fl = fd.integrate_lib(lib, teff, V[0], wl)
for j in range(3):
    checks[f"lib_{fd.LINES[j]}"] = float(np.abs(Fl[j] - F[0, j]).max())
    checks[f"lib_dEW_{fd.LINES[j]}"] = float(fd.diagnostics(Fl[j], j)["ew"] - fd.diagnostics(F[0, j], j)["ew"])
    log(f"{fd.LINES[j]} check (b): library vs exact, los1: max |dF| {checks[f'lib_{fd.LINES[j]}']:.1e}, "
        f"dEW {checks[f'lib_dEW_{fd.LINES[j]}']:+.1e} A")

keys = ["ew", "v1", "sigma", "fwhm", "depth"]
with np.errstate(invalid="ignore", divide="ignore"):
    dF = np.array([[[fd.diagnostics(F[k, j], j)[q] for q in keys] for j in range(3)] for k in range(8)])
    dF0 = np.array([[[fd.diagnostics(F0[k, j], j)[q] for q in keys] for j in range(3)] for k in range(8)])
vmean, vsig = np.zeros((8, 3)), np.zeros((8, 3))
for k in range(8):
    for j in range(3):
        w = fd.weights(MU[k], fc_all[:, j])
        if w.sum() > 0:
            vmean[k, j] = np.sum(w * V[k]) / w.sum()
            vsig[k, j] = np.sqrt(np.sum(w * (V[k] - vmean[k, j]) ** 2) / w.sum())
out = os.path.join(fd.RUN, f"disc_los8{SUFFIX}.npz")
np.savez(out[:-4] + ".tmp.npz", Y=fd.Y, LREF=fd.LREF, los=los, F=F, F0=F0, diag_keys=np.array(keys), diag_F=dF,
         diag_F0=dF0, vmean_w=vmean, sigma_w=vsig, check_keys=np.array(list(checks)), check_vals=np.array(list(checks.values())))
os.replace(out[:-4] + ".tmp.npz", out)
for j in range(3):
    log(f"{fd.LINES[j]}: EW {dF[:, j, 0].mean():.4f} (+-{dF[:, j, 0].std():.4f}) A [no velocities {dF0[:, j, 0].mean():.4f}]; "
        f"<v> {dF[:, j, 1].mean():+.2f}; FWHM {dF[:, j, 3].mean():.0f} vs {dF0[:, j, 3].mean():.0f} km/s; "
        f"depth {dF[:, j, 4].mean():.3f} vs {dF0[:, j, 4].mean():.3f}; disc rms v {vsig[:, j].mean():.1f} km/s")
log(f"wrote {out} and {libpath}")

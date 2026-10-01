"""Hold-out test of the T_eff'-library method (fw_disc_dumps.py) on dump 3200 (validation V2).

For later dumps the library predicts profiles for T_eff' values whose own FASTWIND models were never computed. This test
mimics that: the dump-3200 points are split at random into halves A and B. For each half X, the exact disc-integrated
profiles of X alone (X's own models, dump-3200 velocities, the 8 lines of sight, weight mu F_c; as fw_disc_los.py) are
compared with the prediction from the library built from the OTHER half only (10 K bins, nodes with >= 20 models,
linear T_eff' interpolation: fw_disc.DiscFlux applied to X's points). The in-sample prediction (library from all points)
is shown for reference. Profiles are streamed as in fw_disc_los.py (login node, --nproc 20, ~4 min, ~15 GB).
Output: DISC_DUMPS/holdout.npz.

Usage:  ./run.sh fw_disc_holdout.py --nproc 20
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
ap.add_argument("--block", type=int, default=5000)
ap.add_argument("--nmin", type=int, default=20)
ap.add_argument("--seed", type=int, default=11)
a = ap.parse_args()
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f} s] {msg}", flush=True)


p = np.load(os.path.join(fd.RUN, "profiles.npz"))
teff, theta, phi = p["teff"], p["theta"], p["phi"]
N = teff.size
smp = np.load(os.path.join(fd.SAMPLES, "d3200.npz"))
lam_all, fn_all, fc_all = p["lam"], p["fnorm"], p["fcont"][:, :, 0].astype(np.float64)
rhat, that, phat = fd.unit_vectors(theta, phi)
LOS = fd.los8()
MU, V = np.zeros((8, N)), np.zeros((8, N))
for k in range(8):
    MU[k], V[k] = fd.mu_vlos(LOS[k], rhat, that, phat, smp["ur"].astype(np.float64), smp["uth"].astype(np.float64),
                             smp["uph"].astype(np.float64))
del rhat, that, phat
SH = fd.shift_steps(V)[0]
lib_all = np.load(os.path.join(fd.RUN, "library_dT10.npz"))
edges = lib_all["edges"]
nb = edges.size - 1
bins = np.clip(np.digitize(teff, edges) - 1, 0, nb - 1)
half = np.random.default_rng(a.seed).random(N) < 0.5          # True: half A
HALVES = {"A": half, "B": ~half}
ny = fd.Y.size
MC, JLINE = None, 0


def stream(worker):
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
    depth = np.zeros(ny)
    for g, sh in enumerate(shifts):
        dg = Wrows[g] - Arows[g]
        if sh >= 0:
            depth[:ny - sh] += dg[sh:]
        else:
            depth[-sh:] += dg[:ny + sh]
    return depth


exact = {X: (np.zeros((8, 3, ny)), np.zeros((8, 3, ny))) for X in HALVES}
libs = {X: dict(edges=edges, prof=np.zeros((nb, 3, ny)), fc=np.zeros((nb, 3)), count=np.bincount(bins[m], minlength=nb).astype(float),
                tmean=np.bincount(bins[m], weights=teff[m], minlength=nb) / np.maximum(np.bincount(bins[m], minlength=nb), 1))
        for X, m in HALVES.items()}
for j in range(3):
    JLINE = j
    rows, cols, vals, layout = [], [], [], {}
    r0 = 0
    for X, m in HALVES.items():
        for k in range(8):
            w = fd.weights(MU[k], fc_all[:, j]) * m
            vis = np.where(w > 0)[0]
            sv, gi = np.unique(SH[k, vis], return_inverse=True)
            rows.append(r0 + gi); cols.append(vis); vals.append(w[vis])
            layout[X, k, "v"] = (r0, sv)
            r0 += sv.size
            rows.append(np.full(vis.size, r0)); cols.append(vis); vals.append(w[vis])
            layout[X, k, "0"] = r0
            r0 += 1
        idx = np.where(m)[0]
        rows.append(r0 + bins[idx]); cols.append(idx); vals.append(np.ones(idx.size))
        layout[X, "lib"] = r0
        r0 += nb
    M = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r0, N))
    W = np.asarray(M.sum(axis=1)).ravel()
    MC = M.tocsc()
    with Pool(a.nproc) as pool:
        A = sum(pool.map(stream, range(a.nproc)))
    log(f"{fd.LINES[j]}: streamed {N} profiles through {r0} rows")
    for X, m in HALVES.items():
        for k in range(8):
            g0, sv = layout[X, k, "v"]
            exact[X][0][k, j] = 1.0 - shift_add(A[g0:g0 + sv.size], W[g0:g0 + sv.size], sv) / W[g0:g0 + sv.size].sum()
            r = layout[X, k, "0"]
            exact[X][1][k, j] = A[r] / W[r]
        L = libs[X]
        ok = L["count"] > 0
        L["prof"][ok, j] = A[layout[X, "lib"]:layout[X, "lib"] + nb][ok] / L["count"][ok, None]
        L["fc"][:, j] = np.bincount(bins[m], weights=fc_all[m, j], minlength=nb) / np.maximum(L["count"], 1)
    del A, M, MC

R = {}
fx_all = fd.DiscFlux(fd.lib_nodes(lib_all, nmin=a.nmin))
for X, m in HALVES.items():
    other = "B" if X == "A" else "A"
    fx_other = fd.DiscFlux(fd.lib_nodes(libs[other], nmin=a.nmin))
    for tag, fx in (("holdout", fx_other), ("insample", fx_all)):
        k0, k1, w = fx.pairs(teff)
        F, F0 = np.zeros((8, 3, ny)), np.zeros((8, 3, ny))
        for k in range(8):
            F[k], F0[k] = fx(np.where(m, MU[k], 0.0), V[k], k0, k1, w)[:2]
        R[f"{tag}_{X}_F"] = np.abs(F - exact[X][0]).max(axis=(0, 2))
        R[f"{tag}_{X}_F0"] = np.abs(F0 - exact[X][1]).max(axis=(0, 2))
        R[f"{tag}_{X}_dEW"] = np.array([np.abs([fd.diagnostics(F[k, j], j)["ew"] - fd.diagnostics(exact[X][0][k, j], j)["ew"]
                                                for k in range(8)]).max() for j in range(3)])
        log(f"half {X} ({int(m.sum())} points), library from {'half ' + other if tag == 'holdout' else 'all points'}: max|dF| "
            + " ".join(f"{x:.1e}" for x in R[f'{tag}_{X}_F']) + " (no velocities " + " ".join(f"{x:.1e}" for x in R[f'{tag}_{X}_F0'])
            + "); max|dEW| " + " ".join(f"{x:.1e}" for x in R[f'{tag}_{X}_dEW']) + " A")
    R[f"exact_{X}_F"] = exact[X][0].astype(np.float32)
out = os.path.join(fd.DISC_DUMPS, "holdout.npz")
np.savez(out, lines=np.array(fd.LINES), seed=a.seed, nmin=a.nmin, **R)
log(f"wrote {out}")

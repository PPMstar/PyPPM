"""Disc-integrated line profiles of every moms dump (3200-4800) for the 8 lines of sight, from the dump-3200 FASTWIND
models (no new FASTWIND runs) and each dump's own T_eff' and velocities.

All per-point models differ only in T_eff' (every other input is fixed), so the 1.24 million dump-3200 models form a
library in T_eff'. For a dump, each point's rest-frame profile is interpolated linearly in T_eff' between library nodes
(fw_disc.lib_nodes: the 10 K bins of library_dT10.npz, tail bins merged to >= --nmin models; points beyond the end
nodes take the end node, counted in n_lo/n_hi), shifted by that dump's line-of-sight velocity v = u.n
(lambda_obs = lambda (1 - v/c); v > 0 towards the observer = blueshift) and summed over the visible hemisphere with
weight mu F_c (--method flux: I(mu) = const, as fw_disc_los.py) or with the emergent intensities I(lambda, mu)
(--method imu: as fw_disc_imu.py). T_eff' and u_r, u_theta, u_phi come from sphere_sample.py (SAMPLES/d<dump>.npz).

Library variants of the flux method (systematics of the FASTWIND outputs, fw_disc.lib_nodes): --lamfix (precise instead of
0.01 A-rounded wavelengths), --smooth 335 (nodes smoothed over one period of the EW(T_eff') sawtooth); they go to
OUT/flux_lamfix, OUT/flux_sm335, ... (NAME below).

Output per dump: OUT/<NAME>/d<dump>.npz with F, F0 (8, 3, ny) float32 (with / without Doppler shifts), diag_F,
diag_F0 (8, 3, 5: ew, v1, sigma, fwhm, depth; moments over the whole velocity grid, diag_vwin), vmean_w, sigma_w (8, 3:
weighted mean and rms of v), n_lo, n_hi
(points below / above the node range), wout (8,: visible weight fraction of those points), n_clip (8,: |v| > VSHIFT),
teff_mean, teff_std, teff_min, teff_max, node_range, t_s. Written atomically; existing files are skipped (restartable). fw_disc_collect.py assembles
the time series.

Usage:  ./run.sh fw_disc_dumps.py --method flux --nproc 20                (all dumps, login node)
        ./run.sh fw_disc_dumps.py --method imu --d0 3200 --d1 3209 --nproc 4
        (fw_disc_dumps.sbatch for a compute node; --rank/--nranks split the dumps over nodes)
"""
import argparse
import os
import time
import multiprocessing as mp
from multiprocessing import Pool

import numpy as np

import fw_disc as fd

ap = argparse.ArgumentParser()
ap.add_argument("--method", choices=["flux", "imu"], default="flux")
ap.add_argument("--d0", type=int, default=3200)
ap.add_argument("--d1", type=int, default=4800)
ap.add_argument("--dumps", default=None, help="comma-separated dump list (overrides --d0/--d1)")
ap.add_argument("--nproc", type=int, default=4)
ap.add_argument("--nmin", type=int, default=20, help="minimum number of models per flux-library node")
ap.add_argument("--out", default=fd.DISC_DUMPS)
ap.add_argument("--rank", type=int, default=0)
ap.add_argument("--nranks", type=int, default=1)
ap.add_argument("--maxtasks", type=int, default=20, help="dumps per worker process before it is replaced (CPU-time limit)")
ap.add_argument("--overwrite", action="store_true")
ap.add_argument("--lamfix", action="store_true", help="flux: precise wavelengths (fw_disc.lam_corrections)")
ap.add_argument("--smooth", type=float, default=0.0, help="flux: smooth the nodes over this T_eff' width (K)")
ap.add_argument("--timeout", type=float, default=900.0, help="abort if no dump finishes for this long (s): a killed worker")
a = ap.parse_args()
T0 = time.time()
assert a.method == "flux" or not (a.lamfix or a.smooth), "library variants are for the flux method"
NAME = a.method + ("_lamfix" if a.lamfix else "") + (f"_sm{a.smooth:g}" if a.smooth else "")
OUT = os.path.join(a.out, NAME)
os.makedirs(OUT, exist_ok=True)


def log(msg):
    print(f"[{time.time() - T0:7.1f} s] {msg}", flush=True)


# ---- module level: shared (copy-on-write) with the forked workers ----
if a.method == "flux":
    NODES = fd.lib_nodes(np.load(os.path.join(fd.RUN, "library_dT10.npz")), nmin=a.nmin, smooth=a.smooth, lamfix=a.lamfix)
    INT = fd.DiscFlux(NODES)
else:
    INT = fd.DiscImu()
pts = np.load(os.path.join(fd.RUN, "points.npz"))
rhat, that, phat = fd.unit_vectors(pts["theta"], pts["phi"])
LOS = fd.los8()
MU, TN, PN = rhat @ LOS.T, that @ LOS.T, phat @ LOS.T          # (N, 8): mu, theta.n, phi.n
MU, TN, PN = (np.ascontiguousarray(x.T) for x in (MU, TN, PN))
del rhat, that, phat, pts
KEYS = ["ew", "v1", "sigma", "fwhm", "depth"]


def process(d):
    path = os.path.join(OUT, f"d{d:04d}.npz")
    if os.path.exists(path) and not a.overwrite:
        return None
    t0 = time.time()
    smp = np.load(os.path.join(fd.SAMPLES, f"d{d:04d}.npz"))
    teff = smp["teff"].astype(np.float64)
    ur, uth, uph = (smp[k].astype(np.float64) for k in ("ur", "uth", "uph"))
    k0, k1, w1 = INT.pairs(teff)
    lo, hi = teff < INT.t[0], teff > INT.t[-1]
    ny = fd.Y.size
    F, F0 = np.zeros((8, 3, ny)), np.zeros((8, 3, ny))
    vm, sd = np.zeros((8, 3)), np.zeros((8, 3))
    ncl, wout = np.zeros(8, int), np.zeros(8)
    for k in range(8):
        v = ur * MU[k] + uth * TN[k] + uph * PN[k]
        F[k], F0[k], vm[k], sd[k], ncl[k] = INT(MU[k], v, k0, k1, w1)
        vis = MU[k] > 0
        wout[k] = MU[k][vis & (lo | hi)].sum() / MU[k][vis].sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        dF = np.array([[[fd.diagnostics(F[k, j], j, vwin=fd.VY)[q] for q in KEYS] for j in range(3)] for k in range(8)])
        dF0 = np.array([[[fd.diagnostics(F0[k, j], j, vwin=fd.VY)[q] for q in KEYS] for j in range(3)] for k in range(8)])
    tmp = path[:-4] + f".tmp{os.getpid()}.npz"
    np.savez(tmp, dump=d, t_s=float(smp["t_s"]), method=a.method, name=NAME, lamfix=a.lamfix, smooth=a.smooth,
             F=F.astype(np.float32), F0=F0.astype(np.float32), diag_keys=np.array(KEYS), diag_vwin=fd.VY, diag_F=dF, diag_F0=dF0, vmean_w=vm, sigma_w=sd, n_lo=int(lo.sum()),
             n_hi=int(hi.sum()), wout=wout, n_clip=ncl, teff_mean=teff.mean(), teff_std=teff.std(), teff_min=teff.min(), teff_max=teff.max(),
             nmin=a.nmin if a.method == "flux" else 0, node_range=np.array([INT.t[0], INT.t[-1]]))
    os.replace(tmp, path)
    return (f"dump {d}: EW {dF[:, 0, 0].mean():.4f} {dF[:, 1, 0].mean():.4f} {dF[:, 2, 0].mean():.4f} A, FWHM "
            f"{dF[:, 0, 3].mean():.0f} {dF[:, 1, 3].mean():.0f} {dF[:, 2, 3].mean():.0f} km/s, sigma_v {sd[:, 0].mean():.1f} km/s, "
            f"out of range {int(lo.sum())}/{int(hi.sum())} (vis. weight <= {wout.max():.1e}), clipped {int(ncl.sum())}; "
            f"{time.time() - t0:.1f} s")


if __name__ == "__main__":
    if a.dumps is not None and not a.dumps.strip():
        raise SystemExit("--dumps given but empty")
    dumps = [int(x) for x in a.dumps.split(",")] if a.dumps is not None else list(range(a.d0, a.d1 + 1))
    dumps = [d for i, d in enumerate(dumps) if i % a.nranks == a.rank]
    todo = [d for d in dumps if a.overwrite or not os.path.exists(os.path.join(OUT, f"d{d:04d}.npz"))]
    log(f"{NAME}: {len(dumps)} dumps (rank {a.rank}/{a.nranks}), {len(todo)} to do, {a.nproc} workers -> {OUT}; "
        f"T_eff' nodes {INT.t.size} ({INT.t[0]:.0f}-{INT.t[-1]:.0f} K)")
    n = 0
    t1 = time.time()
    with Pool(a.nproc, maxtasksperchild=a.maxtasks) as pool:
        it = pool.imap_unordered(process, todo)
        while True:
            try:
                msg = it.next(timeout=a.timeout)       # a worker killed by a signal (OOM, CPU limit) never returns
            except StopIteration:
                break
            except mp.TimeoutError:
                left = [d for d in todo if not os.path.exists(os.path.join(OUT, f"d{d:04d}.npz"))]
                log(f"ABORT: no dump finished for {a.timeout:.0f} s (worker killed?); {len(left)} dumps missing, e.g. {left[:10]}; "
                    f"resubmit to finish (existing outputs are kept)")
                raise SystemExit(3)
            n += 1
            if msg:
                log(f"[{n}/{len(todo)}] {msg}")
    dt = time.time() - t1
    log(f"done: {n} dumps in {dt:.0f} s ({dt / max(n, 1):.2f} s per dump wall, {a.nproc} workers)")

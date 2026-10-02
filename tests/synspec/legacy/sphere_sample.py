"""Sample moms dumps at the points of the per-point FASTWIND sphere (for disc integration and time series).

At the same points and with the same interpolation as fw_sphere_extract.py (ppmpy's constant-area grid on
the sphere r = --radius, --npoints points, trilinear interpolation of the moms cell values) this computes,
for every dump:
    relT = (T9 - <T9>)/<T9>,  teff = --teff0 (1 + relT)        (the model's T_eff' for that dump)
    ur, uth, uph   spherical velocity components [km/s]: ppmpy get_spherical_components on the moms grid
                   (physics convention: theta from +z, phi from +x), each then interpolated to the points,
                   exactly as u_r in fw_sphere_extract.py (so ur here = ur_kms of points.npz for that dump).
The line-of-sight velocity towards an observer in direction n is u.n = ur (r.n) + uth (theta.n) + uph (phi.n).
Velocities are the moms values (not density-corrected), as in the per-point run.

Output: OUTDIR/d<dump>.npz (float32, points in idx order): relT, teff, ur, uth, uph; plus T9_mean, t_s.
Written atomically; existing files are skipped, so a job can be resubmitted. --check RUN_DIR compares
relT, teff and ur with RUN_DIR/points.npz (must be identical for the dump of that run).

Usage:  ./run.sh sphere_sample.py --d0 3200 --d1 3200 [--check RUN_DIR]
        parallel: --worker i --nworkers n   (dump d goes to worker d % n; see sphere_sample.sbatch)
"""
import argparse
import os
import time

import numpy as np

import figstyle as fs

ap = argparse.ArgumentParser()
ap.add_argument("--d0", type=int, default=fs.MOMS_DUMPS[0])
ap.add_argument("--d1", type=int, default=fs.MOMS_DUMPS[1])
ap.add_argument("--radius", type=float, default=4050.0)
ap.add_argument("--npoints", type=int, default=1_236_544)
ap.add_argument("--teff0", type=float, default=38230.0)
ap.add_argument("--outdir", default=None)
ap.add_argument("--worker", type=int, default=0)
ap.add_argument("--nworkers", type=int, default=1)
ap.add_argument("--check", default=None)
a = ap.parse_args()

out = a.outdir or f"/scratch/ppathak/fastwind_sphere/samples_r{a.radius:.0f}_N{a.npoints}"
os.makedirs(out, exist_ok=True)
mine = [d for d in range(a.d0, a.d1 + 1) if d % a.nworkers == a.worker]
todo = [d for d in mine if not os.path.exists(os.path.join(out, f"d{d:04d}.npz"))]
print(f"worker {a.worker}/{a.nworkers}: {len(mine)} dumps, {len(todo)} to do -> {out}", flush=True)
if not todo:
    raise SystemExit(0)

m = fs.moms(todo[0])
iu = [fs.MOMS_VARS.index(v) for v in ("ux", "uy", "uz")]
for d in todo:
    t0 = time.time()
    T9 = m.get_spherical_interpolation(fs.MOMS_VARS.index("T9"), a.radius, fname=d, npoints=a.npoints)
    ux, uy, uz = (m.get(i, fname=d) for i in iu)
    comps = m.get_spherical_components(ux, uy, uz)
    ur, uth, uph = (m.get_spherical_interpolation(c, a.radius, fname=d, npoints=a.npoints) * 1e3 for c in comps)
    del ux, uy, uz, comps
    relT = (T9 - T9.mean()) / T9.mean()
    teff = a.teff0 * (1.0 + relT)
    path = os.path.join(out, f"d{d:04d}.npz")
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, relT=relT.astype(np.float32), teff=teff.astype(np.float32), ur=ur.astype(np.float32),
             uth=uth.astype(np.float32), uph=uph.astype(np.float32), T9_mean=float(T9.mean()), t_s=fs.time_s(d))
    os.replace(tmp, path)
    print(f"dump {d}: Teff' {teff.min():.0f}-{teff.max():.0f} K, u_r {ur.min():.1f}..{ur.max():.1f}, "
          f"|u_t| rms {np.sqrt(np.mean(uth ** 2 + uph ** 2)):.1f} km/s  {time.time() - t0:.1f} s", flush=True)
    if a.check:
        p = np.load(os.path.join(a.check, "points.npz"))
        for k, v in (("relT", relT), ("teff", teff), ("ur_kms", ur)):
            print(f"  check vs points.npz {k}: max |diff| {np.abs(p[k] - v).max():.3e}", flush=True)

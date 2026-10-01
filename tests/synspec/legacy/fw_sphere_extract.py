"""Stage 1 of the per-point FASTWIND run: sample one moms dump on a sphere, once.

For every point of ppmpy's constant-area (Fibonacci) grid on the sphere r = --radius
(dump --dump, --npoints points) compute
    relT  = (T9 - <T9>) / <T9>                  (moms slot 7, <.> = sphere mean)
    teff  = --teff0 * (1 + relT)                (input Teff of that point's FASTWIND model)
    ur    = U . r_hat  [km/s]                   (from ux, uy, uz; the Doppler shift applied later)
as in the 3x3 sensitivity test (fastwind_var_setup.py). This is the only step that reads the
moms data (one process, ~11 GB). It writes, atomically (temporary name + rename, so readers
never see a partial file), into --outdir:
    points.npz   idx; coordinates r (Mm), theta, phi (physics convention, rad: theta from +z,
                 phi from +x in the x-y plane) and x, y, z (Mm) of the moms grid; relT, teff, ur_kms, T9
    points.txt   "idx teff" per line (%.3f K), read once per node by fw_sphere_node.sbatch
    meta.json    run parameters and summary statistics

Usage:  ./run.sh fw_sphere_extract.py [--dump 4800] [--radius 4050] [--npoints 618272]
"""
import argparse
import json
import os

import numpy as np

import figstyle as fs

ap = argparse.ArgumentParser()
ap.add_argument("--dump", type=int, default=4800)
ap.add_argument("--radius", type=float, default=4050.0)
ap.add_argument("--npoints", type=int, default=618_272)       # 8 (lmax+1)^2 with lmax = 277 ~ one point per moms cell
ap.add_argument("--teff0", type=float, default=38230.0)       # MESA model 3700 photospheric Teff (rounded)
ap.add_argument("--outdir", default=None)
a = ap.parse_args()

out = a.outdir or f"/scratch/ppathak/fastwind_sphere/d{a.dump}_r{a.radius:.0f}_N{a.npoints}"
os.makedirs(out, exist_ok=True)

m = fs.moms(a.dump)
T9 = m.get_spherical_interpolation(fs.MOMS_VARS.index("T9"), a.radius, fname=a.dump, npoints=a.npoints)
ux, uy, uz = (m.get(fs.MOMS_VARS.index(v), fname=a.dump) for v in ("ux", "uy", "uz"))
ur, _, _ = m.get_spherical_components(ux, uy, uz)
ur_kms = m.get_spherical_interpolation(ur, a.radius, fname=a.dump, npoints=a.npoints) * 1e3

# physics (theta, phi) of ppmpy's constant-area grid (same formula as MomsDataSet._constantArea_spherical_grid)
ind = np.arange(a.npoints) + 0.5
theta = np.arccos(1.0 - 2.0 * ind / a.npoints)
g = np.pi * (1.0 + 5 ** 0.5) * ind
phi = g - 2.0 * np.pi * np.floor(g / (2.0 * np.pi))

# the grid ppmpy interpolated on, from ppmpy itself (x, y, z in its [z, y, x] igrid order), as a check
igrid, th_p, ph_p = m._constantArea_spherical_grid(np.array([a.radius]), a.npoints)
assert np.allclose(th_p, theta) and np.allclose(ph_p, phi), "theta/phi differ from ppmpy's grid"
r = np.full(a.npoints, a.radius)
x, y, z = r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)
ig = np.asarray(igrid).reshape(-1, 3)
assert np.allclose(ig[:, 2], x, atol=1e-6) and np.allclose(ig[:, 1], y, atol=1e-6) and np.allclose(ig[:, 0], z, atol=1e-6), \
    "x, y, z differ from ppmpy's interpolation grid"
print(f"coordinates verified against ppmpy's interpolation grid ({a.npoints} points, r = {a.radius} Mm)")

relT = (T9 - T9.mean()) / T9.mean()
teff = a.teff0 * (1.0 + relT)
idx = np.arange(a.npoints, dtype=np.int32)

tmp = os.path.join(out, "points.tmp.npz")
np.savez(tmp, idx=idx, r=r, theta=theta, phi=phi, x=x, y=y, z=z, relT=relT, teff=teff, ur_kms=ur_kms, T9=T9)
os.replace(tmp, os.path.join(out, "points.npz"))
tmp = os.path.join(out, "points.tmp.txt")
np.savetxt(tmp, np.column_stack([idx, teff]), fmt=["%d", "%.3f"])
os.replace(tmp, os.path.join(out, "points.txt"))

meta = dict(dump=a.dump, radius_Mm=a.radius, npoints=a.npoints, teff0=a.teff0, t_days=fs.time_s(a.dump) / 86400,
            T9_mean=float(T9.mean()), relT_min=float(relT.min()), relT_max=float(relT.max()), relT_std=float(relT.std()),
            teff_min=float(teff.min()), teff_max=float(teff.max()), ur_min=float(ur_kms.min()), ur_max=float(ur_kms.max()),
            ur_std=float(ur_kms.std()), note="ur from moms (not density-corrected); Doppler shift applied after FASTWIND")
tmp = os.path.join(out, "meta.tmp.json")
json.dump(meta, open(tmp, "w"), indent=1)
os.replace(tmp, os.path.join(out, "meta.json"))
print(json.dumps(meta, indent=1))
print("wrote", out)

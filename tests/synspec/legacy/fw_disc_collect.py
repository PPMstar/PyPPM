"""Assemble the per-dump disc-integrated profiles of fw_disc_dumps.py into one time series per method.

Reads DISC_DUMPS/<NAME>/d<dump>.npz for dumps D0..D1 (NAME = method or a library variant such as flux_sm335) and writes
DISC_DUMPS/<NAME>_timeseries.npz with
dumps, t_s, Y, LREF, los, F, F0 (ndump, 8, 3, ny) float32, diag_keys, diag_F, diag_F0 (ndump, 8, 3, 5), vmean_w, sigma_w
(ndump, 8, 3), n_lo, n_hi (ndump,), wout, n_clip (ndump, 8), teff_mean, teff_std, teff_min, teff_max (ndump,), node_range. Refuses to write if a dump is
missing (use --allow-missing to collect what exists).

Usage:  ./run.sh fw_disc_collect.py --name flux [--d0 3200 --d1 4800]
"""
import argparse
import os

import numpy as np

import fw_disc as fd

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="flux", help="flux, imu, or a variant directory such as flux_sm335")
ap.add_argument("--d0", type=int, default=3200)
ap.add_argument("--d1", type=int, default=4800)
ap.add_argument("--out", default=fd.DISC_DUMPS)
ap.add_argument("--allow-missing", action="store_true")
a = ap.parse_args()

src = os.path.join(a.out, a.name)
want = list(range(a.d0, a.d1 + 1))
have = [d for d in want if os.path.exists(os.path.join(src, f"d{d:04d}.npz"))]
missing = sorted(set(want) - set(have))
print(f"{a.name}: {len(have)} of {len(want)} dumps present" + (f"; missing {missing[:20]}{' ...' if len(missing) > 20 else ''}" if missing else ""))
if missing and not a.allow_missing:
    raise SystemExit(1)
nd, ny = len(have), fd.Y.size
F, F0 = np.zeros((nd, 8, 3, ny), np.float32), np.zeros((nd, 8, 3, ny), np.float32)
dF, dF0 = np.zeros((nd, 8, 3, 5)), np.zeros((nd, 8, 3, 5))
vm, sd = np.zeros((nd, 8, 3)), np.zeros((nd, 8, 3))
wout, ncl = np.zeros((nd, 8)), np.zeros((nd, 8), int)
t_s, nlo, nhi, tm, ts, tmin, tmax = (np.zeros(nd) for _ in range(7))
for i, d in enumerate(have):
    r = np.load(os.path.join(src, f"d{d:04d}.npz"))
    assert int(r["dump"]) == d and str(r["name"]) == a.name
    if i == 0:
        method, vwin = str(r["method"]), float(r["diag_vwin"])
    F[i], F0[i], dF[i], dF0[i], vm[i], sd[i] = r["F"], r["F0"], r["diag_F"], r["diag_F0"], r["vmean_w"], r["sigma_w"]
    wout[i], ncl[i] = r["wout"], r["n_clip"]
    t_s[i], nlo[i], nhi[i], tm[i], ts[i] = r["t_s"], r["n_lo"], r["n_hi"], r["teff_mean"], r["teff_std"]
    tmin[i], tmax[i] = r["teff_min"], r["teff_max"]
    keys, nrange = r["diag_keys"], r["node_range"]
out = os.path.join(a.out, f"{a.name}_timeseries.npz")
np.savez(out[:-4] + ".tmp.npz", dumps=np.array(have), t_s=t_s, Y=fd.Y, LREF=fd.LREF, los=fd.los8(), method=method, name=a.name, diag_vwin=vwin,
         F=F, F0=F0, diag_keys=keys, diag_F=dF, diag_F0=dF0, vmean_w=vm, sigma_w=sd, n_lo=nlo, n_hi=nhi, wout=wout,
         n_clip=ncl, teff_mean=tm, teff_std=ts, teff_min=tmin, teff_max=tmax, node_range=nrange)
os.replace(out[:-4] + ".tmp.npz", out)
res = F - F.mean(axis=0, dtype=np.float64, keepdims=True)
for j in range(3):
    print(f"{fd.LINES[j]}: EW {dF[:, :, j, 0].mean():.4f} A (rms over dumps and LOS {dF[:, :, j, 0].std():.1e}); "
          f"<v> rms {dF[:, :, j, 1].std():.2f} km/s; FWHM {dF[:, :, j, 3].min():.0f}-{dF[:, :, j, 3].max():.0f} km/s; "
          f"residual F - <F>_t: rms {res[:, :, j].std():.1e}, max {np.abs(res[:, :, j]).max():.1e}")
print(f"out-of-range points per dump: max {int(nlo.max())} below / {int(nhi.max())} above, visible weight <= {wout.max():.1e}; "
      f"clipped |v| > {fd.VSHIFT} km/s: {int(ncl.sum())}")
print(f"wrote {out} ({os.path.getsize(out) / 1e9:.2f} GB)")

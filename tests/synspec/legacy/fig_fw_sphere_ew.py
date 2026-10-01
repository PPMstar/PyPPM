"""Per-point FASTWIND run: sanity checks of profiles.npz and equivalent widths vs Teff'.

Reads RUN_DIR/profiles.npz (fw_sphere_merge.py --combine) and
  * checks that every point of points.txt has a converged model and finite profiles;
  * checks whether the wavelength grid of each line is the same for all points;
  * prints iteration / run-time statistics and the total CPU time of the models;
  * computes the equivalent width EW = int (1 - F/F_cont) dlambda of each line at rest
    (no Doppler shift, no broadening) for every point, cached as RUN_DIR/ew.npz;
  * fits a smooth cubic EW(Teff') per line to the medians in 25 K bins (equal weight for every Teff',
    so the sparse tails do not distort it) and reports the scatter of the models about it (the
    branch-switching artefact of lambda 4026/4200, see the progress document) next to the real EW rms;
  * plots EW vs Teff' (point density) for the three lines, with the reference model
    (Teff 38 230 K) marked: figures/fw_sphere_ew_<tag>.pdf/png.

Usage:  ./run.sh fig_fw_sphere_ew.py [RUN_DIR]
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import figstyle as fs

RUN = sys.argv[1] if len(sys.argv) > 1 else "/scratch/ppathak/fastwind_sphere/d3200_r4050_N1236544"
TAG = os.path.basename(RUN.rstrip("/"))
LABELS = [r"He I+II $\lambda$4026", r"He II $\lambda$4200", r"He I $\lambda$4922"]
EW_REF = [1.035, 0.756, 0.422]          # reference model M424_T38230 (Mdot 1e-10), unbroadened
TEFF0 = 38230.0
CHUNK = 100_000

meta = json.load(open(os.path.join(RUN, "meta.json")))
p = np.load(os.path.join(RUN, "profiles.npz"))
idx, teff, status = p["idx"], p["teff"], p["status"]
n = meta["npoints"]
print(f"{TAG}: {idx.size} of {n} points in profiles.npz, unique {np.unique(idx).size}; "
      + ", ".join(f"{s} {c}" for s, c in zip(*np.unique(status, return_counts=True))))
assert idx.size == n and np.all(idx == np.arange(n)) and np.all(status == "ok")
nud = p["teff_nudge"]
print(f"Teff' {teff.min():.1f} - {teff.max():.1f} K; {int((nud != 0).sum())} points with a Teff nudge "
      f"(idx {idx[nud != 0].tolist()}, +{nud.max():g} K)")
niter, tp, tf = p["niter"], p["t_pnlte"], p["t_formal"]
print(f"iterations: median {np.median(niter):.0f}, mean {niter.mean():.1f}, range {niter.min()}-{niter.max()}; "
      f"pnlte time (192 per node): median {np.median(tp):.0f} s, mean {tp.mean():.0f} s, max {tp.max():.0f} s; "
      f"total model CPU time {(tp.sum() + tf.sum()) / 3600:.0f} core-h")

cache = os.path.join(RUN, "ew.npz")
if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(os.path.join(RUN, "profiles.npz")):
    ew = np.load(cache)["ew"]
else:
    lam0 = p["lam"][0]                                   # [3, 161]
    ew = np.empty((n, 3))
    dlam_max = np.zeros(3)
    nbad = 0
    for i0 in range(0, n, CHUNK):                        # chunks: lam, fnorm are 2.4 GB each
        lam = p["lam"][i0:i0 + CHUNK].astype(np.float64)
        fn = p["fnorm"][i0:i0 + CHUNK].astype(np.float64)
        nbad += int((~np.isfinite(fn)).any(axis=(1, 2)).sum())
        dlam_max = np.maximum(dlam_max, np.abs(lam - lam0).max(axis=(0, 2)))
        ew[i0:i0 + CHUNK] = np.trapz(1.0 - fn, lam, axis=2)
    print(f"points with non-finite profiles: {nbad}; max wavelength-grid deviation from point 0 per line "
          f"{', '.join(f'{d:.2e}' for d in dlam_max)} A")
    tmp = cache[:-4] + ".tmp.npz"
    np.savez(tmp, idx=idx, teff=teff, ew=ew, lines=p["lines"])
    os.replace(tmp, cache)
    print("wrote", cache)

x = (teff - TEFF0) / 1000.0
fs.set_style()
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
xs = np.linspace(x.min(), x.max(), 200)
bins = np.arange(x.min(), x.max() + 0.025, 0.025)             # 25 K
kb = np.digitize(x, bins)
use = [i for i in range(1, bins.size) if (kb == i).sum() >= 5]
xb = np.array([np.median(x[kb == i]) for i in use])
for j, ax in enumerate(axs):
    c = np.polyfit(xb, [np.median(ew[kb == i, j]) for i in use], 3)
    res = ew[:, j] - np.polyval(c, x)
    print(f"{p['lines'][j]}: EW {ew[:, j].min():.4f} - {ew[:, j].max():.4f} A (mean {ew[:, j].mean():.4f}, rms over the "
          f"sphere {ew[:, j].std():.4f}; sphere-mean / reference {ew[:, j].mean() / EW_REF[j]:.4f}); scatter about a "
          f"smooth cubic EW(Teff'): {res.std():.4f} A rms ({100 * res.std() / ew[:, j].mean():.2f} %), "
          f"max |dev| {np.abs(res).max():.4f} A")
    hb = ax.hexbin(teff / 1e3, ew[:, j], gridsize=70, norm=LogNorm(), cmap="magma_r", mincnt=1, linewidths=0)
    ax.plot(TEFF0 / 1e3 + xs, np.polyval(c, xs), color=fs.GREY, lw=0.8)
    ax.plot(TEFF0 / 1e3, EW_REF[j], "o", mfc="none", mec=fs.cbcolor(1), ms=5, mew=0.9)
    ax.set_xlabel(r"$T_{\rm eff}'$ (kK)")
    ax.set_title(LABELS[j])
axs[0].set_ylabel(r"EW (\AA)" if plt.rcParams["text.usetex"] else r"EW ($\mathrm{\AA}$)")
cbar = fig.colorbar(hb, ax=axs, pad=0.015, fraction=0.03)
cbar.set_label("points per bin")
fs.savefig(fig, f"fw_sphere_ew_{TAG}")

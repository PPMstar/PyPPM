"""Power spectrum of the central zero crossing A(t) of the residual spectra (fig_disc_zerocross.py), computed exactly as in
get_temporal_spectrum() of ~/ppmstar.repo/projects/H-core-M25/analysis_fullstar/Temporal_spectra-threaded.ipynb:
  x = A/<A> - 1 (instead of the notebook's quadratic detrending, as requested); x *= np.hanning(N);
  np.pad(x, (pad//2, pad//2), 'mean') with pad = 10 000 000 (as the notebook's cell 16); dft = np.fft.fft; frequencies np.fft.fftfreq(N2, dt), positive only, in muHz;
  power = sqrt(8/3) * (1e-6 dt / N1) |dft|^2, shown in ppm^2/muHz (x in ppm)   (N1 = unpadded length, so power per muHz is conserved; sqrt(8/3) Hann factor).
dt = spacing of dumps 3200/3201 (the notebook's time_spacing). Dumps without a crossing (41 in lambda4200) are filled by
linear interpolation in time.
Dashed vertical lines: eigenmode frequencies of M424 identified by Pathak et al. (2026) (EIGEN_MUHZ, muHz; list from the user).
Output: figures/fw_disc_zerocross_spectrum_<name>_los<k>.pdf/png; with --los 1,2,...,8 one row per line of sight
(..._los1-8) and the mean power over them (..._los1-8_mean).

Usage:  ./run.sh fig_disc_zerocross_spectrum.py [--name imu] [--los 1 | --los 1,2,3,4,5,6,7,8] [--pad 10000000] [--no-hann]
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs
import fw_disc as fd

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="imu")
ap.add_argument("--los", default="1", help="comma-separated lines of sight; several: stacked figure + mean over them")
ap.add_argument("--pad", type=int, default=10_000_000)
ap.add_argument("--no-hann", action="store_true")
ap.add_argument("--zdir", default=fd.DISC_DUMPS, help="directory of zerocross_<name>_los<k>.npz")
ap.add_argument("--stack-ylim", default="1e-1,1e3", help="y range of the per-LOS panels (ppm^2/muHz)")
ap.add_argument("--mean-ylim", default="4,200", help="y range of the mean-over-LOS figure (ppm^2/muHz)")
a = ap.parse_args()
EIGEN_MUHZ = np.array([10.95588503, 13.52081278, 85.22377711, 99.36265539, 128.59128465, 141.89232907,
                       149.26678274, 154.59501136, 175.97912585])      # M424 eigenmodes (Pathak et al. 2026)
z = np.load(os.path.join(a.zdir, f"zerocross_{a.name}_los{a.los.split(',')[0]}.npz"))
t = z["t_s"] - z["t_s"][0]
dt = z["t_s"][1] - z["t_s"][0]


def spectrum(x):
    N1 = x.size
    w = np.hanning(N1) if not a.no_hann else 1.0
    xf = w * x
    if a.pad:
        xf = np.pad(xf, (a.pad // 2, a.pad // 2), "mean")
    dft = np.fft.fft(xf)
    f = np.fft.fftfreq(xf.size, dt)
    pos = f > 0
    return f[pos] * 1e6, np.sqrt(8 / 3) * (1e-6 * dt / N1) * np.abs(dft[pos]) ** 2 * 1e12     # ppm^2 / muHz


def series(los, j):
    z = np.load(os.path.join(a.zdir, f"zerocross_{a.name}_los{los}.npz"))
    A = z["A"][:, j].copy()
    bad = ~np.isfinite(A)
    A[bad] = np.interp(t[bad], t[~bad], A[~bad])
    return A / A.mean() - 1, int(bad.sum())


def decorate(ax, fnyq):
    ax.set_xlim(1.0, fnyq)
    for fe in EIGEN_MUHZ:
        ax.axvline(fe, color=fs.cbcolor(1), ls="--", lw=0.6, zorder=0)


fnyq = 0.5e6 / dt
LOS = [int(x) for x in a.los.split(",")]
P = {}
for los in LOS:
    for j in range(3):
        x, nbad = series(los, j)
        f, p = spectrum(x)
        keep = (f >= 1.0) & (f <= fnyq)
        fk, P[los, j] = f[keep], p[keep]
        i = np.argmax(p)
        print(f"los{los} {fd.LINES[j]}: {nbad} dumps interpolated; highest power at {f[i]:.2f} muHz ({f[i] * 0.0864:.3f} d^-1); "
              f"total power {np.trapz(p, f):.3e} ppm^2, variance of x {x.var() * 1e12:.3e} ppm^2")

fs.set_style()
tag = f"los{LOS[0]}" if len(LOS) == 1 else f"los{LOS[0]}-{LOS[-1]}"
nr = len(LOS)
fig, axs = plt.subplots(nr, 3, figsize=(7.2, 2.4 + 0.9 * (nr - 1)), sharex=True, sharey=True, squeeze=False)
for r, los in enumerate(LOS):
    for j in range(3):
        ax = axs[r, j]
        ax.loglog(fk, P[los, j], color="k", lw=0.4, rasterized=True)
        decorate(ax, fnyq)
        if nr > 1:
            ax.set_ylim(*[float(v) for v in a.stack_ylim.split(",")])
            ax.set_yticks([1e0, 1e2])                    # no labels at the row boundaries (they would overlap)
        if r == 0:
            ax.set_title(fd.LABELS[j])
            sec = ax.secondary_xaxis("top", functions=(lambda x: x * 86400e-6, lambda x: x / 86400e-6))
            sec.set_xlabel(r"frequency (d$^{-1}$)", fontsize=7)
        if nr > 1:
            ax.text(0.03, 0.06, f"los{los}", transform=ax.transAxes, fontsize=6.5)
    axs[r, 0].set_ylabel(r"power (ppm$^2\,\mu$Hz$^{-1}$)" if nr == 1 else "power", fontsize=8 if nr > 1 else None)
for j in range(3):
    axs[-1, j].set_xlabel(r"frequency ($\mu$Hz)")
if nr > 1:
    fig.text(0.005, 0.5, r"power (ppm$^2\,\mu$Hz$^{-1}$)", rotation=90, va="center")
fig.subplots_adjust(wspace=0.08, hspace=0.08, top=1 - 0.5 / (2.4 + 0.9 * (nr - 1)))
fs.savefig(fig, f"fw_disc_zerocross_spectrum_{a.name}_{tag}")

if nr > 1:                                                     # mean power over the lines of sight
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
    for j in range(3):
        Pm = np.mean([P[los, j] for los in LOS], axis=0)
        axs[j].loglog(fk, Pm, color="k", lw=0.4, rasterized=True)
        decorate(axs[j], fnyq)
        axs[j].set_title(fd.LABELS[j])
        axs[j].set_xlabel(r"frequency ($\mu$Hz)")
        sec = axs[j].secondary_xaxis("top", functions=(lambda x: x * 86400e-6, lambda x: x / 86400e-6))
        sec.set_xlabel(r"frequency (d$^{-1}$)", fontsize=7)
        i = np.argmax(Pm)
        print(f"mean of {nr} LOS, {fd.LINES[j]}: highest power at {fk[i]:.2f} muHz ({fk[i] * 0.0864:.3f} d^-1)")
    axs[0].set_ylim(*[float(v) for v in a.mean_ylim.split(",")])
    axs[0].set_ylabel(r"$\langle$power$\rangle_{\rm los}$ (ppm$^2\,\mu$Hz$^{-1}$)")
    fig.subplots_adjust(wspace=0.08, top=0.78)
    fs.savefig(fig, f"fw_disc_zerocross_spectrum_{a.name}_{tag}_mean")

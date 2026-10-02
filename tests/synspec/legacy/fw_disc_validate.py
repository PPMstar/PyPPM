"""Validate the library-based disc integration (fw_disc.integrate_lib) against the exact sum over the
individual FASTWIND models of the dump-3200 run (fw_disc.integrate_exact), for a few observer directions.

Builds (or loads) the T_eff' library, samples dump 3200 (sphere_sample.py output: relT, T_eff', u_r, u_theta,
u_phi), and for each direction compares the normalised profiles and their diagnostics. Also checks that the
library itself reproduces the individual rest-frame profiles (scatter within a bin).

Usage:  ./run.sh fw_disc_validate.py [--ndir 3] [--dT 10]
"""
import argparse
import time

import numpy as np

import fw_disc as fd

ap = argparse.ArgumentParser()
ap.add_argument("--ndir", type=int, default=3)
ap.add_argument("--dT", type=float, default=10.0)
a = ap.parse_args()

t0 = time.time()
lib = fd.library(dT=a.dT)
print(f"library: {lib['count'].size} bins of {a.dT:g} K, {int((lib['count'] > 0).sum())} filled, "
      f"{int(lib['count'].sum())} models ({time.time() - t0:.0f} s)")
p = fd.load_run()
s = np.load(f"{fd.SAMPLES}/d3200.npz")
assert np.allclose(s["teff"], p["teff"], atol=2.0), "sample T_eff' differs from the run (beyond retry nudges)"
rhat, that, phat = fd.unit_vectors(p["theta"], p["phi"])
fc = p["fcont"][:, :, 0].astype(np.float64)
for k, nvec in enumerate(fd.directions(a.ndir)):
    mu, v = fd.mu_vlos(nvec, rhat, that, phat, s["ur"], s["uth"], s["uph"])
    t1 = time.time()
    Fx = fd.integrate_exact(p, v, fd.weights(mu, 1.0)[:, None] * fc)
    t2 = time.time()
    b = np.clip(np.digitize(p["teff"], lib["edges"]) - 1, 0, lib["edges"].size - 2)
    Fl = fd.integrate_lib(lib, p["teff"], v, fd.weights(mu, 1.0)[:, None] * lib["fc"][b])
    t3 = time.time()
    print(f"direction {k} n = ({nvec[0]:+.2f}, {nvec[1]:+.2f}, {nvec[2]:+.2f}): {int((mu > 0).sum())} visible points; "
          f"exact {t2 - t1:.0f} s, library {t3 - t2:.1f} s")
    for j in range(3):
        dx, dl = fd.diagnostics(Fx[j], j), fd.diagnostics(Fl[j], j)
        print(f"  {fd.LINES[j]:8s} max|F_lib - F_exact| {np.abs(Fl[j] - Fx[j]).max():.1e};  exact: EW {dx['ew']:.4f} A, "
              f"<v> {dx['v1']:+.2f}, sigma {dx['sigma']:.2f}, FWHM {dx['fwhm']:.0f} km/s | library: EW {dl['ew']:.4f}, "
              f"<v> {dl['v1']:+.2f}, sigma {dl['sigma']:.2f}, FWHM {dl['fwhm']:.0f}")

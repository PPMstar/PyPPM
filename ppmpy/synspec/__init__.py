"""
ppmpy.synspec: synthetic line profiles from PPMstar moms data and their variability.

PP 2026-10-01: started from the tested M424 -> FASTWIND -> disc-integration pipeline
(project stellar-atmosphere-KU-Leuven). The pipeline:

1. sample a moms dump on an equal-area sphere (T_eff' and velocities per point);
2. compute local NLTE line profiles with FASTWIND (external code, located by path);
3. build T_eff' libraries of flux and intensity profiles;
4. disc-integrate every dump for chosen lines of sight;
5. analyse the line-profile variability (residuals, zero crossings, temporal spectra).

Modules
-------
conventions  physical constants, sign conventions, unit helpers, line-of-sight sets
spectral     LineSet, VelocityGrid, row interpolation, Doppler steps
io           atomic writes, '_meta' provenance, zero-copy npz members
diagnostics  equivalent width, centroid, width, FWHM, depth; broadening kernels and fits
lpv          residual spectra, zero-crossing tracker, gap filling, lag correlation
spectrum     temporal power spectra (padded FFT or direct DFT)
plotting     symmetric-log norms, dynamic spectra, profile bundles, power spectra

Every module imports only numpy and scipy at load time (plotting imports matplotlib
inside its functions); nothing here imports ppmpy.ppm at module level.
"""
API_VERSION = 1

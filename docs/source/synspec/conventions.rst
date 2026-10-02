.. PP 2026-10-02: new (M8 documentation of ppmpy.synspec).

Conventions
===========

These conventions hold in every module of ``ppmpy.synspec``. The authoritative statements are in the module
docstrings of :mod:`ppmpy.synspec.conventions`, :mod:`ppmpy.synspec.spectral`, :mod:`ppmpy.synspec.sphere`,
:mod:`ppmpy.synspec.disc` and :mod:`ppmpy.synspec.spectrum`.

Units
-----

=====================  ================================================================================
quantity               unit
=====================  ================================================================================
wavelength             Angstrom, **air** (FASTWIND's OUT files); rest wavelengths of the lines in a
                       :class:`~ppmpy.synspec.spectral.LineSet` (M424: 4026.22, 4199.90, 4921.93)
velocity               km/s (``C_KMS`` = 299 792.458). PPMstar code velocities are Mm/s: the moms
                       samples are multiplied by ``velocity_scale`` = 1e3
lengths, radii         Mm (PPMstar code units; M424 sampling radius 4050 Mm)
temperature, T_eff'    K
time                   s (``t_s`` of the samples and products, from the rprof headers)
frequency              microHz; 1 d\ :sup:`-1` = 11.574 microHz (``MUHZ_PER_CPD``), 1 microHz =
                       0.0864 d\ :sup:`-1` (``CPD_PER_MUHZ``); quote both
equivalent width       Angstrom
profiles               continuum-normalised flux F / F_c (1 in the continuum); depth d = 1 - F
relative power         (relative fluctuation)\ :sup:`2` per microHz; times ``PPM2_PER_REL2`` = 1e12 gives
                       ppm\ :sup:`2` per microHz
=====================  ================================================================================

Velocity grid
-------------

All profiles are handled on a uniform grid in the logarithmic velocity

    y = c ln(lambda / lambda_ref)    [km/s],    lambda = lambda_ref exp(y / c),

one grid per line with that line's ``lref`` as zero point (:class:`ppmpy.synspec.spectral.VelocityGrid`,
:func:`~ppmpy.synspec.spectral.lam_of_y`, :func:`~ppmpy.synspec.spectral.y_of_lam`). The M424 grid (the default
``VelocityGrid()``) has dv = 1 km/s and abs(y) <= vmax = 2700 km/s (ny = 5401 points); Doppler shifts up to
vshift = 400 km/s are handled, larger ones are clipped and counted (``n_clip``).

* A Doppler shift is exact on this grid: the shift in grid steps is ``rint(-c ln(1 - v/c) / dv)``
  (:meth:`~ppmpy.synspec.spectral.VelocityGrid.shift_steps`); all-dump integrations round it to whole steps (the
  error of this rounding is check V6, :doc:`validation`).
* The FFT disc integration needs every rest profile to have zero depth at abs(y) > vmax - vshift
  (:meth:`~ppmpy.synspec.spectral.VelocityGrid.check_zero_padding`; DiscFlux checks it).
* Models are interpolated from their native FASTWIND wavelengths onto the grid by
  :func:`~ppmpy.synspec.spectral.interp_rows` (linear, constant beyond the model's band). FASTWIND prints lambda to
  0.01 Angstrom only, so a native wavelength row can contain the same value twice; compare models row by row.
* Equivalent width (:func:`ppmpy.synspec.diagnostics.line_diagnostics`):
  EW = (lref / c) int d exp(y / c) dy over the whole grid (the factor exp(y / c) = lambda / lref is the Jacobian
  d lambda / dy). The moments v1 (centroid) and sigma are taken over abs(y) <= vwin; the per-dump products use the whole
  grid, because the 400 km/s window of the early dump-3200 figures cuts the Stark wings of lambda 4026 / 4200.

Doppler sign
------------

The line-of-sight velocity is v = u . n with n the unit vector **from the star towards the observer**:

* v > 0 means motion towards the observer, a **blueshift**: lambda_obs = lambda (1 - v / c)
  (:func:`~ppmpy.synspec.spectral.doppler_lambda`);
* on the y grid the profile seen at y is the rest profile at y + s dv, s = the shift in steps; a centroid v1
  is therefore about -<v>;
* the mean line-of-sight velocity of the disc, weighted like the profile (``vmean_w``), and the centroid of the
  integrated line anti-correlate (M424 imu: correlation -0.996).

Frames, points and weights
--------------------------

* Simulation frame: x, y, z of the moms grid (the cube is indexed [z, y, x] as in ppmpy). Spherical coordinates
  follow the physics convention: theta from +z, phi from +x in the x-y plane, phi in [0, 2 pi).
* Local basis (:func:`ppmpy.synspec.sphere.sphere_basis`): r_hat, theta_hat (towards increasing theta, i.e.
  southwards), phi_hat. The samples ``ur``, ``uth``, ``uph`` are the components along them (ppmpy's
  ``get_spherical_components``).
* Points: the equal-area golden-spiral grid :func:`ppmpy.synspec.sphere.fibonacci_sphere`, the grid of ppmpy's
  ``MomsDataSet`` spherical interpolations; every point represents dA = 4 pi R\ :sup:`2` / N.
* For a line of sight n: mu = r_hat . n (> 0 on the visible hemisphere), and v = u_r mu + u_theta (theta_hat . n) +
  u_phi (phi_hat . n) (:func:`~ppmpy.synspec.sphere.project_los`, :func:`~ppmpy.synspec.sphere.los_velocity`).
* Flux method: weight mu F_c per visible point (I(mu) = const; :func:`~ppmpy.synspec.sphere.disc_weights`; a linear
  limb-darkening coefficient ``uld`` is available but not used for M424). Intensity method: the model's own I(y, mu)
  in direction mu, rays with impact parameter p <= R_max mapped to mu = sqrt(1 - (p / R_max)\ :sup:`2`) and
  interpolated linearly in s = p / R_max (this reproduces FASTWIND's own flux; linear in mu would bias the EW by up
  to 0.7 %).
* The M424 samples are the moms values: velocities are not density-corrected and moms temperatures are
  briquette-averaged (project notes).

Lines of sight
--------------

Lines of sight are given by name or as an array (:func:`ppmpy.synspec.conventions.los_array`); there is deliberately
no default set.

``'thompson2024'`` (:func:`~ppmpy.synspec.conventions.los_thompson2024`), used for all M424 line-profile products: the
8 directions of Thompson et al. (2024, Sect. 2.2): los1 = (1, 1, 1), los2 = los1 x (0, 0, 1), los3 = los1 x los2,
los4 = los1 + los2 + los3, each normalised before it is used for the next, and los5..8 = -los1..-los4::

    los1  ( 0.5774,  0.5774,  0.5774)      los5 = -los1
    los2  ( 0.7071, -0.7071,  0     )      los6 = -los2
    los3  ( 0.4082,  0.4082, -0.8165)      los7 = -los3
    los4  ( 0.9773,  0.1608, -0.1381)      los8 = -los4

``'ppmstar_fortran'`` (:func:`~ppmpy.synspec.conventions.los_ppmstar_fortran`): the 8 directions of the PPMstar
Fortran code behind the rprof luminosities ``lum1`` .. ``lum8`` (``ppm.make_los_vectors_fortran``; primary
direction (0.3, 0.3, 0.3) from the flags file). **It is not the same set:**

=====  ================================  =============================
LOS    PPMstar Fortran vector            relation to ``thompson2024``
=====  ================================  =============================
1, 5   (+-0.5774, +-0.5774, +-0.5774)    same
2, 6   (-+0.7071, +-0.7071, 0)           reversed (cos = -1)
3, 7   (-+0.4082, -+0.4082, +-0.8165)    reversed (cos = -1)
4, 8   (-+0.3106, +-0.5059, +-0.8047)    different (cos = -1/3)
=====  ================================  =============================

So a LOS-by-LOS comparison of line profiles with the rprof ``lum_k`` pairs different directions except for k = 1
and 5 (frequencies are unaffected). ``ppm.make_los_vectors`` agrees with ``thompson2024`` except for LOS 4 (13
degrees apart: it does not normalise before summing). ``'fibonacci:N'`` gives N directions spread evenly over the
sphere.

Products and dtypes
-------------------

Per-dump and time-series products keep the legacy layout: F, F0 float32 with shape (nlos, nline, ny) per dump and
(ndump, nlos, nline, ny) in ``<name>_timeseries.npz``; ``diag_F`` / ``diag_F0`` float64 with the last axis
``diag_keys`` = (ew, v1, sigma, fwhm, depth). Products are uncompressed .npz files without pickles, written
atomically (temporary name, then rename); new products may carry a ``_meta`` provenance member
(:mod:`ppmpy.synspec.io`), which the readers do not require. Statistics on float32 profiles are computed in float64.

Temporal spectra and the 'ppmstar' normalisation
------------------------------------------------

:func:`ppmpy.synspec.spectrum.temporal_power_spectrum` takes a uniformly sampled series (``dt`` from
:func:`~ppmpy.synspec.spectrum.sample_spacing`; M424 dumps 3200-4800: dt_mean = 2834.54 s, spacings 2826-2843 s),
a required ``detrend`` (None, 'mean', 'ratio' = x / mean(x) - 1, or a polynomial), a window (default Hann) and
mean-padding (default 1e7 samples, as the PPMstar notebooks).

* ``norm='ppmstar'`` (default, the legacy PPMstar pipeline ``get_temporal_spectrum`` /
  ``ppmpy.spectra.lums_temporal_spectrum``): P(f) = sqrt(8/3) (1e-6 dt / N) abs(Z(f))\ :sup:`2`, N = number of
  samples (not the padded length), Z = DFT of the windowed, mean-padded series. The Hann power correction 8/3 enters
  as its square root and the one-sided factor 2 is missing, so for a stationary series with the Hann window the
  power integrates to sqrt(3/32) = 0.306 of the variance (0.816 without a window). Peak positions and shapes are
  unaffected; absolute levels are a factor 16 / (3 sqrt(8/3)) = 3.27 below a variance-conserving PSD (Hann).
* ``norm='psd'``: the one-sided, window-corrected PSD 2 (1e-6 dt) abs(Z)\ :sup:`2` / sum(w\ :sup:`2`), which
  conserves the variance (Parseval).
* ``method='fft'`` (legacy, bit-identical) or ``method='dft'`` (the same values on any frequency grid, cheaper for
  a band).

For M424, quote frequencies in d\ :sup:`-1` and microHz; with 1601 dumps the resolution is 0.22 microHz and the Nyquist
frequency 176.4 microHz.

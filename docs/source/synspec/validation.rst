.. PP 2026-10-02: new (M8 documentation of ppmpy.synspec).

Validation and tests
====================

Two questions are kept apart:

* **Is the code right?** The test suite (``tests/synspec``): the ported code against frozen copies of the original
  project scripts and against the stored M424 products, analytic cases, synthetic runs with injected faults.
* **Is a run's method good enough?** The validation checks of :mod:`ppmpy.synspec.validate` (V1-V6, hold-out, brute
  force, EW conservation), which measure what the library and the rounding of a run cost, compared with the run's
  own line-profile variability (LPV).

The bit-identity policy
-----------------------

Every default of ``ppmpy.synspec`` reproduces the M424 production bit for bit (byte for byte for the .npz files
without ``_meta``). New options (other start methods, low-memory modes, ``_meta`` records, extra fields) are
opt-in and leave the defaults unchanged. Where a port reorganised a computation (blockwise sums, slab reads of the
moms cubes, streamed merges), the tests show that the bits do not depend on the block size, the number of workers,
'fork' or 'spawn', or the BLAS thread count. Every change is checked on a few dumps (the subset 3200, 3334, 4000,
4169, 4391, 4800 and 10 random dumps); the full reproduction of all 1601 dumps (job 2483431, 2026-10-02: sphere
samples, per-dump files and time series of the flux, flux_lamfix, flux_sm335 and imu runs) was identical to the
production.

**CPU caveat.** The bitwise agreement holds for numpy 1.26 (the Python 3.9 container) on the AVX512 Trillium nodes
(AMD EPYC 9655), where the products were made. Elsewhere the last bits of transcendental functions differ:

* numpy 1.26 evaluates ``np.log`` with AVX512_SKX SVML routines: without AVX512, 3 % of the grid values
  y = c ln(lambda / lref) change by 1 ulp; ``arccos`` (theta of the sphere grid) differs from libm ``acos`` in the
  last bit for ~30 % of the points; sin and cos come from the platform's libm;
* numpy >= 2 rewrote ``np.exp`` vectorisation and ``np.fft``: EW factors, broadening kernels and Fourier
  amplitudes differ at the 1e-16 level;
* the projections mu = r_hat . n go through BLAS (``method='matmul'`` = one dgemm as the all-dump production,
  ``'matvec'`` = dgemv as the dump-3200 exact sums); another BLAS may differ by 1 ulp. With 1 km/s grid steps a
  Doppler shift crosses a rounding boundary with probability ~1e-14 per point: for dump 3200 the rounded shifts and
  hence the F profiles are identical for all three methods; only vmean_w, sigma_w change in the last bits;
* node profiles with smoothing or a correction are BLAS sums whose last bits depend on the thread count: the
  integrator factories and :func:`~ppmpy.synspec.dumps.run_disc_dumps` limit the loaded BLAS to 1 thread.

Off Trillium the products then agree to ~1e-16 (float64) or 1 float32 ulp, far below any validation tolerance. For
bitwise work elsewhere read theta and phi from ``points.npz`` instead of recomputing the grid.
:meth:`~ppmpy.synspec.library.FluxLibrary.save` records the relevant CPU features in ``_meta``.

Test tiers
----------

============  =================================================================================================
marker        meaning
============  =================================================================================================
(none)        fast tests on synthetic data, analytic cases and the frozen legacy sources; run anywhere
``m424``      regressions against the M424 production products; skipped when the products are absent
``slow``      takes more than ~30 s (most M424 products at full size)
``fastwind``  runs the real FASTWIND; skipped without the install (``PPMPY_FASTWIND_ROOT``) or ``/cvmfs``
============  =================================================================================================

About 880 tests (2026-10-02), of which ~50 are slow (the 10 ``fastwind`` tests among them) and ~110 use M424
products. ``tests/synspec/legacy/`` holds frozen copies of the original project scripts (``README.txt`` lists which
test runs which lines); the tests run them next to the port and compare the outputs bit for bit, so they must not
read the live project files.

Running the tests (from the PyPPM checkout; ``tests/synspec/README.txt`` has the details)::

    SIF=/project/rrg-fherwig-ad/fherwig/Apptainers/python__3.9-env.sif
    apptainer exec --bind /home,/scratch,/project $SIF python -m pytest tests/synspec -q -m "not slow"
    apptainer exec --bind /home,/scratch,/project $SIF python -m pytest tests/synspec/test_disc.py -q
    apptainer exec --bind /home,/scratch,/cvmfs $SIF python -m pytest tests/synspec -q -m fastwind

The fast tests took 7.8 min at milestone M4 (626 tests) on a Trillium login node; run heavy selections one at a
time there.

Self-test (no data, no FASTWIND)
--------------------------------

``python -m ppmpy.synspec.testing`` (:func:`ppmpy.synspec.testing.selftest`) builds a toy run with the production
functions (analytic pseudo-Voigt lines whose depth, centre and width vary with T_eff', 20 000 points with smooth
random T_eff' and velocity fields of M424-like amplitude, 40 library nodes, 3 dumps, the 8 Thompson lines of sight)
and validates it: V1-V6, the hold-out test, the brute force against the integrator and the stored products, EW
conservation, the LPV comparison and analytic anchors (Lambert's law, the blueshift sign for a uniform outflow, the
hidden hemisphere). It takes ~10 s serially (0.6 GB); ``--nproc 2`` also checks that serial, 'fork' and 'spawn' runs
are identical; ``--full`` uses the M424 grid (39 s). ``--mutate NAME`` injects one of the faults of
``testing.MUTATIONS`` (library offset, dropped node, flipped v, swapped lines of sight, permuted points, ...), which
must then fail. Run it once per installation.

The validation checks
---------------------

All checks give max abs(dF) over lines of sight and the velocity grid, per line, in units of the continuum (dEW in
Angstrom). :func:`ppmpy.synspec.validate.run_validation` runs those the given inputs allow and returns a
:class:`~ppmpy.synspec.validate.ValidationReport` (printable table, ``to_json``, ``to_npz``).

=================  ====================================================  ===================  ===============
check              compares                                              M424 recorded        default tol.
=================  ====================================================  ===================  ===============
V1 (flux, imu)     library dump through the pipeline vs exact            3.2e-6, 2.0e-6;      5e-6; 2e-5 A
                   per-point sums (every point's own model)              dEW 7.3e-6 A
V2 hold-out        exact sums of one random half vs the library of the   8.4e-6 (in-sample    1e-5; 5e-5 A
                   other half (the situation of every later dump)        7.1e-6); dEW 2.3e-5
V3                 points beyond the node range: clamped vs              1.6e-6               5e-6
                   extrapolated (leave-out informational: 1.0e-4)
V4                 linear T_eff' interpolation vs nearest 10 K bin       9.0e-7               2e-6
V5                 node merging nmin 1 / 5 / 100 vs 20                   2.6e-7               1e-6
V6                 shifts rounded to 1 km/s vs continuous                3.3e-5               5e-5
                   (20 000-point subsets)
brute              direct per-point shift-and-add vs the integrator      ~1e-15; 3.0e-8       1e-10; 2.5e-7
                   (float64) and vs the stored float32 products
brute_imu          the same for the intensity method                     2.5e-14 (subset);
                                                                         3.0e-8 vs stored
EW conservation    abs(EW(F) - EW(F0)) / EW (whole-step shifts           1.6e-7               1e-6
                   conserve the EW)
=================  ====================================================  ===================  ===============

(:data:`ppmpy.synspec.validate.RECORDED_M424` and ``RECORDED_M424_ARRAYS`` hold the full values per line,
:data:`~ppmpy.synspec.validate.DEFAULT_TOLERANCES` the tolerances.) The tolerances are M424-specific (grid,
library density, velocity field): set them per run. What matters is the ratio to the run's LPV
(:func:`~ppmpy.synspec.validate.compare_lpv`): M424 residual rms 3.0e-4 / 9.0e-5 / 2.9e-4 (imu, abs(v) <= 600 km/s),
EW rms 1.3e-4 / 2.3e-4 / 2.6e-4 A. The profile checks stay within 3 % of the residual rms (V6 aside, a subset
effect: 3-11 % on 20 000 points, 1.4-2.7 % with all ~618 000 visible points); the lambda 4026 EW checks reach
6-18 % of the EW rms because of the EW(T_eff') sawtooth (:doc:`fastwind`), so compare_lpv warns on V6 and the dEW
checks.

What the checks see: V1 and V2 compare with sums over the per-point models themselves; V3-V6 compare the integrator
with variants of itself (they quantify method choices); the brute force sees faults of the integration (shifts,
weights, FFT), and against stored products wrong inputs of a rerun. The table of which injected fault each check
detects is in the :mod:`ppmpy.synspec.validate` docstring and asserted by ``test_validate.py``. The references
share the projections, the line-of-sight velocity and the shift rounding with the pipeline, so sign and
convention errors there are caught by the analytic anchors (``testing.convention_check``), not by V1-V6.

Library mode adds V8 (:func:`ppmpy.synspec.libmode.sparse_library_test`): libraries from a sparse subset of existing
per-point models against the full library, to choose the node spacing (M424: dT = 10 K, one model per node, keeps
the time-variable deviation at <= 1.4 % of the LPV rms).

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`ppmpy` is a Python package for analysing and visualising output from PPMstar
stellar hydrodynamics simulations. It is interactive-first: the API is designed
to be driven from IPython / Jupyter sessions (the canonical environment is the
NuGrid `astrohub` server, which ships the data and dependencies). There is no
test suite, CI, or lint config — development is exploratory and validated against
real simulation data.

## Commands

```bash
# Install in editable mode (dependencies: numpy, scipy, matplotlib, nugridpy, setuptools)
pip install -e .
# Install from GitHub (as end users do)
pip install git+https://github.com/PPMstar/PyPPM

# Build the Sphinx docs (run from docs/)
cd docs && make html        # output in docs/build/html
cd docs && make gh-pages     # WARNING: commits + pushes local repo to remote master, switches branches
```

Most of ppmpy has no tests; sanity-check changes by importing the package and
exercising it against a data directory, e.g. `python -c "from ppmpy import ppm; ..."`.
`ppmpy.synspec` has a pytest suite (`tests/synspec`, run from the checkout; the
conftest puts the checkout first on `sys.path`):

```bash
apptainer exec --bind /home <python-3.9 SIF> python -m pytest tests/synspec -q -p no:cacheprovider
```

Tests marked `m424` compare with the M424 production products on Trillium scratch
and skip when those are absent; `tests/synspec/legacy/` holds frozen copies of the
original project sources used as the reference.

## Architecture

Almost all code lives in `ppmpy/ppm.py` (~15k lines). The package exposes several
data-reader classes plus a large library of `compute_*` physics routines and
plotting methods.

### Two simulation-output formats, two reader hierarchies

PPMstar has produced two generations of output, and the code reflects both:

- **YProfile files** (legacy, PPMstar 1.0): `YProfile-01-xxxx.bobaaa` text files
  with a quirky header/column format (parsing rules documented in
  `docs/source/installation.rst`). Read by **`yprofile(DataPlot, PPMtools)`**.
- **.rprof / .hstry files** (PPMstar 2.0): read by **`RprofSet(PPMtools, DerivedMixin)`**
  (a run = a set of dumps), **`Rprof`** (one dump), and **`RprofHistory`** (`.hstry` files).
- **Moments datacubes** (.aaa, 3D briquette-averaged): **`MomsDataSet(DerivedMixin)`**
  / `MomsData`, with half-briquette-resolution variants `MomsDataSet2X` / `MomsData2X`.

`RprofSet` is the primary modern entry point. `initialize_cases(...)` (module-level)
builds a dict of cases, each wiring up an `RprofSet`, its history, a reference dump,
luminosity factors, and grid metadata.

### PPMtools — shared physics layer

`PPMtools` is a mix-in inherited by both `yprofile` and `RprofSet`, so the same
`compute_*` API works across both output formats. Key points:

- `compute(quantity, fname, ...)` dispatches to `compute_<quantity>` methods.
  `get_computable_quantities()` lists them. There are dozens: thermodynamics
  (`compute_Hp`, `compute_Gamma1`, `compute_csound`, `compute_nabla_T*`,
  `compute_N2`), opacity/diffusion (`compute_kappa`, `compute_Krad`, `compute_Drad`),
  luminosities (`compute_lum_conv/kin/rad/tot`), nuclear burning
  (`compute_enuc_C12pg`, `compute_rhodot_C12pg`, `compute_T9`), and mass/geometry
  (`compute_m`, `compute_g`, `compute_mu`).
- Many compute methods branch on `self.isRprofSet()` and on geometry
  (`self.get_geometry()` returning `'spherical'` vs `'cartesian'`) because the
  data layout and physical definitions differ. When adding or editing a compute
  method, handle both the yprofile and RprofSet paths and both geometries.
- `num_type` (`'ndump'` vs `'t'`/time) selects how `fname` is interpreted; nearly
  every reader/compute method threads this argument through.

### Derived-quantity registry — `ppmpy/derived.py`

A decoupled, decorator-based registry for quantities that combine raw variables
(e.g. velocity components → `ur`, `utot`). `@derived_quantity('name')` registers a
function taking the host object as its first arg; `DerivedMixin._get_derived()`
(inherited by `RprofSet` and `MomsDataSet`) looks it up when a requested variable
isn't in the raw data. Prefer adding new combination formulas here rather than as
`compute_*` methods when they apply to the Rprof/Moms readers. Note the precedence
rule (see `ppm.py` header comment): if a variable is directly available in the
RProf data, the reader uses it instead of computing it.

### Supporting modules

- `ppmpy/rprofile.py` — low-level binary `.rprof` reader (`rprofile_set`,
  `rprofile`), imported into `ppm.py` as `bprof`.
- `ppmpy/ppmsetup.py` — setup/EOS/unit-conversion helpers (`UnitConvert`, `eosPS`,
  `EOSgasrad`, MESA-profile burning-coefficient extraction).
- `ppmpy/spectra.py` — line-of-sight `lums` spectra (disk-integrated light-curve
  power spectra): `lums_temporal_spectrum` (polynomial detrend, Hann window,
  mean-pad, FFT, sqrt(8/3)(1e-6 dt/N)|X|^2), `lums_spectra_rprof(rprofset, ...)`
  for rprof `lum1..lum8`, `lums_spectra_moms(moms, ...)` and
  `compare_lums_with_rprof(moms, ...)` for the moms-derived side, plus plots.
  Kept separate from `ppm.py`: it imports only numpy/scipy at load time and
  imports `ppm`, matplotlib and tqdm inside the functions that need them.
- `ppmpy/synspec/` — synthetic line profiles from moms data and their variability
  (M424 -> FASTWIND -> disc integration pipeline, being moved here in milestones):
  `conventions` (c, Doppler and frame conventions, line-of-sight sets), `spectral`
  (LineSet, VelocityGrid y = c ln(lambda/lambda_ref)), `io` (atomic npz, '_meta'
  provenance, npz memmaps), `diagnostics` (EW, moments, FWHM, broadening kernels and
  fits), `lpv` (residual spectra, zero-crossing tracker, systematics), `spectrum`
  (temporal power spectra, FFT or direct DFT), `plotting` (matplotlib imported inside
  functions). M2: `fwresults` (OUT readers, merge/streaming combine,
  ProfileStore), `sphere`, `parallel` (fork/spawn pools, watchdog), `library` (flux
  library, nodes), `disc` (DiscFlux, exact sums), `dumps` (all-dump driver, collect). M3: `validate`
  (V1-V6, hold-out, brute force, EW conservation, `run_validation`), `testing` (toy
  runs; `python -m ppmpy.synspec.testing` runs `selftest()`, < 1 min, no data). M4: intensity method:
  `library.build_imu_library`/`ImuLibrary`, `disc.DiscImu` (default = legacy, 9 GB;
  `fft='lazy'` ~2.8 GB, `dtype='float32'`), `disc.integrate_imu_nearest`,
  `dumps.imu_integrator`, `fwresults.locate_points`/`extract_points`. M5: `moms` (block reader,
  `sample_moms_sphere` with backend 'slab' (~6 s, 0.3 GB per dump, bit-identical to
  the 'momsdataset' backend: ~50 s, 18 GB), `write_points_table`, `sample_moms_dumps`). M6: `fastwind/`
  (stdlib only, runs under the host python without numpy; FASTWIND itself is never
  shipped): `FastwindInstall` (check, stage), `Indat` (byte-identical to the old awk
  edit), `FormalInput`, log/CONVERG/MAXTCORR convergence, `run_model`, `rerun_formal`,
  `batch.run_models` (task k of K, packing child, SIGUSR1 stop/restart), `archive`, CLI
  `python3 -m ppmpy.synspec.fastwind`, and a fake FASTWIND for tests (`fake`). Rules: Python 3.9 syntax; only numpy/scipy imported at module level;
  defaults reproduce the M424 products bit for bit, verified by the tests.
- `Messenger` (in `ppm.py`) — centralised verbosity-controlled output; reader
  classes take a `verbose` int (0=silent … 3=all) and route messages through it.

### Conventions

- `ppm.py` does `from numpy import *` and `from nugridpy.data_plot import *` at
  module top, so the namespace is broad — be mindful of shadowing.
- Data are in **PPM code units**; conversions go through `ppmsetup.UnitConvert`
  and `nugridpy.constants`.
- Docstrings are **NumPy-style** and feed the Sphinx autosummary docs — match that
  format when adding public methods.
- Provenance for non-obvious changes is recorded as dated inline comments with
  author initials (e.g. `# PP 2026-03-14: ...`); follow that style for tricky fixes.

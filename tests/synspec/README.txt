Tests of ppmpy.synspec (PP 2026-10-02)

Run them from the PyPPM checkout. conftest.py puts the checkout first on sys.path (the Python 3.9 container has its
own, older /PyPPM). The numpy tests run in the container; the FASTWIND runner tests need only the standard library
but run the real FASTWIND binaries only where their ELF interpreter (/cvmfs) exists.

    cd /home/ppathak/PyPPM
    SIF=/project/rrg-fherwig-ad/fherwig/Apptainers/python__3.9-env.sif
    apptainer exec --bind /home,/scratch,/project $SIF python -m pytest tests/synspec -q -m "not slow"   # fast
    apptainer exec --bind /home,/scratch,/project $SIF python -m pytest tests/synspec -q                  # all
    apptainer exec --bind /home,/scratch,/project $SIF python -m pytest tests/synspec/test_discimu.py -q   # one module
    apptainer exec --bind /home,/scratch,/cvmfs $SIF python -m pytest tests/synspec -q -m fastwind   # FASTWIND

Markers (conftest.py)
    (none)     fast: synthetic data, analytic cases, frozen legacy sources; run anywhere
    m424       regression against the M424 production products; skipped when they are absent
    slow       takes more than ~30 s
    fastwind   runs the real FASTWIND; skipped without the install (PPMPY_FASTWIND_ROOT, default
               /scratch/ppathak/FW_10.6.4.1) or /cvmfs
Combine them with -m, e.g. -m "m424 and not slow". About 880 tests (2026-10-02), ~50 of them slow (all fastwind
tests are slow), ~110 m424. The fast tests took 7.8 min at M4 (626 tests) on a Trillium login node.

Login node: run one heavy selection at a time (the slow M424 tests use several GB and up to 8 workers each); the
3600 s CPU limit per process applies to a pytest process too. Compute nodes cannot write /home: pass
-p no:cacheprovider there.

Data locations (environment variables; defaults are the Trillium paths of the M424 project)
    PPMPY_SYNSPEC_M424_RUN       /scratch/ppathak/fastwind_sphere/d3200_r4050_N1236544 (points, profiles, library)
    PPMPY_SYNSPEC_M424_DISC      /scratch/ppathak/fastwind_sphere/disc_dumps_r4050_N1236544 (per-dump products)
    PPMPY_SYNSPEC_M424_SAMPLES   /scratch/ppathak/fastwind_sphere/samples_r4050_N1236544 (per-dump sphere samples)
    PPMPY_SYNSPEC_M424_PROJECT   tests/synspec/legacy (frozen project sources; another copy of them)
    PPMPY_SYNSPEC_M424_IMU_RUNS, _IMU_RAW, _FWRUNS, _ITMORE, _PPMRUN, _R3, ...   further M424 inputs (see the tests)
    PPMPY_FASTWIND_ROOT          FASTWIND install for the fastwind tests
    PPMPY_SYNSPEC_TMP            disk directory for large temporary files (default $SCRATCH/synspec_tmp; the slow
                                 intensity-library build writes 1.9 GB there, not into a RAM /tmp)
    PPMPY_SYNSPEC_START_METHOD   start method of worker pools ('spawn' to test macOS / Windows behaviour)
Note: PPMPY_SYNSPEC_M424_IMU means different things in different test files (a directory, the library file, one
model directory): leave it unset.

Without the M424 data (another machine) the m424 tests skip and the rest must pass; bitwise comparisons with stored
M424 products assume numpy 1.26 on AVX512 CPUs (docs/source/synspec/validation.rst).

No data and no pytest needed: python -m ppmpy.synspec.testing (self-test of the whole pipeline on a toy run, ~10 s).

legacy/ holds frozen copies of the original project scripts that the tests run next to the port; legacy/README.txt
lists which test uses which lines. Never edit them.

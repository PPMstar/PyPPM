"""
ppmpy.synspec.fastwind: running FASTWIND (1D NLTE model atmospheres and line profiles) for many sphere points.

This subpackage imports only the Python standard library: it runs under the host python of a cluster (Trillium:
/usr/bin/python3 3.9, no numpy), where the FASTWIND binaries can run (they need the host's /cvmfs ELF interpreter,
absent in the Python container). FASTWIND itself is not part of ppmpy; the user gives its location
(:class:`FastwindInstall`, or the environment variables PPMPY_FASTWIND_ROOT / _BUILD / _FORMAL_BUILD / _LAUNCHER).

Modules
-------
install  FastwindInstall (layout, check(), has_imu_patch(), fingerprint(), stage() -> StagedRoot), ELF interpreter
indat    Indat: format-preserving INDAT.DAT editor in nlte.f90's READ order (byte-identical to the legacy awk edit)
formal   FormalInput (line list), formal_suffix / out_name (formalsol.f90's output names), formalsol_stdin,
         out_problem (is an OUT / OUT_IMU file complete?)
logs     pnlte.log digest, CONVERG / MAXTCORR.dat, convergence verdict (nlte.f90 criteria), legacy status + flags
model    run_model (one point: fw_sphere_point.sh in Python, process groups, timeouts), rerun_formal, stop_all
         (signal-handler safe), extras 'full' / 'digest' / none
fake     a fake FASTWIND (stdlib-python executables with the same file contract) for tests; selftest()

Outputs are byte-compatible with the legacy per-point run (meta.txt, result directory layout, kept model files), so
``ppmpy.synspec.fwresults`` (numpy side) reads them unchanged.

PP 2026-10-02: new (M6), ported from the project scripts fastwind_run.sh, fw_sphere_point.sh, fw_sphere_task.sh and
fw_imu_run.sh (stellar-atmosphere-KU-Leuven).
"""
from .formal import OUT_NROW, FormalInput, FormalLine, formal_suffix, formalsol_stdin, out_name, out_names, out_problem
from .indat import INDAT_SCHEMA, Indat
from .install import FastwindInstall, StagedRoot, elf_interpreter
from .logs import classify, convergence, parse_converg, parse_maxtcorr, parse_pnlte_log
from .model import ModelBusy, ModelInterrupted, parse_digest, rerun_formal, run_model, stop_all

__all__ = ["FastwindInstall", "StagedRoot", "elf_interpreter", "Indat", "INDAT_SCHEMA", "FormalInput",
           "FormalLine", "formal_suffix", "formalsol_stdin", "out_name", "out_names", "out_problem", "OUT_NROW",
           "parse_pnlte_log", "parse_converg", "parse_maxtcorr", "convergence", "classify", "run_model",
           "rerun_formal", "stop_all", "parse_digest", "ModelInterrupted", "ModelBusy"]

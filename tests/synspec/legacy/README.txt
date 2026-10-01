Frozen legacy sources and tables of the M424 project (stellar-atmosphere-KU-Leuven, commit 67e042f, 2026-10-01),
used by the synspec regression tests as a fixed reference: the project scripts are being rewritten on top of
ppmpy.synspec, so the tests must not read the live project files.

fw_disc.py                          shared functions of the original pipeline (diagnostics, kernels, DiscFlux, ...)
fig_disc_zerocross_spectrum.py      original spectrum()/series() of the zero-crossing power spectra
figures/fw_disc_{vmac,profiles}{,_imu}_d3200_r4050_N1236544.csv   tables printed by the original fig_disc_vmac.py
                                    and fig_disc_profiles.py

PPMPY_SYNSPEC_M424_PROJECT can point the tests at another copy.

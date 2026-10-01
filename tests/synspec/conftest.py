"""pytest configuration for ppmpy.synspec.

Run from the PyPPM checkout, e.g. in the Python 3.9 container:
    apptainer exec --bind /home,/scratch,/project SIF python -m pytest tests/synspec -q
The checkout is put first on sys.path (the container has its own, older /PyPPM).

Markers
-------
m424      regression against the M424 production products; skipped when they are absent.
          Locations default to the Trillium paths and can be changed with PPMPY_SYNSPEC_M424_RUN,
          PPMPY_SYNSPEC_M424_DISC (see M424 below).
slow      takes more than ~30 s.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.path[0] != ROOT:
    sys.path.insert(0, ROOT)

M424 = dict(
    run=os.environ.get("PPMPY_SYNSPEC_M424_RUN", "/scratch/ppathak/fastwind_sphere/d3200_r4050_N1236544"),
    disc=os.environ.get("PPMPY_SYNSPEC_M424_DISC", "/scratch/ppathak/fastwind_sphere/disc_dumps_r4050_N1236544"),
)


def pytest_configure(config):
    config.addinivalue_line("markers", "m424: regression against the M424 production products (skipped if absent)")
    config.addinivalue_line("markers", "slow: takes more than ~30 s")


def m424_path(kind, *parts):
    """Path of an M424 product; skips the test when it does not exist."""
    p = os.path.join(M424[kind], *parts)
    if not os.path.exists(p):
        pytest.skip("M424 product not available: {}".format(p))
    return p

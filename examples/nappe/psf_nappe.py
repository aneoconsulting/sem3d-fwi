"""Point-spread functions (../fathi/psf.py) of the water-table case: does the surface acquisition see
lambda below the water table, and does it separate it from mu? Results in $SEMFWI_RUNS/nappe/psf_<pulse>.

    python psf_nappe.py [--pulse p20] [--depths 3 8 12 15 25]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fathi"))
from psf import run  # noqa: E402
from run_nappe import CASE  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pulse", default="p20")
    ap.add_argument("--depths", type=float, nargs="+", default=[3., 8., 12., 15., 25.])
    ap.add_argument("--width", type=float, default=1.)
    ap.add_argument("--dx", type=float, default=1e-3)
    run(ap.parse_args(), CASE)

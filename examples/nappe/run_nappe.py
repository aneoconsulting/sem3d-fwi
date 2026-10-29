"""Inversion of the water-table case (nappe.py) with the driver of Fathi's cases (run_fathi2.invert):
homogeneous 80 MPa initial model, Total Variation, T = 0.45 s for p20 and 0.4 s for p30 and p40 (as
in Fathi's case 3). Runs in $SEMFWI_RUNS/nappe/<name>.

    python run_nappe.py both --stages p20:20 --bias 10      # Fathi's strategy
    python run_nappe.py both --stages p20:20                # without the bias
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fathi"))
import nappe  # noqa: E402
from run_fathi2 import SIM_TIME, Case, invert, parser  # noqa: E402
from semfwi.config import RUNS_DIR  # noqa: E402

CASE = Case(os.path.join(RUNS_DIR, "nappe"), nappe.true_model, SIM_TIME[3], geometry=3, reg="tv")

if __name__ == "__main__":
    invert(parser().parse_args(), CASE)

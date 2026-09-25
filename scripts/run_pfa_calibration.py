#!/usr/bin/env python3
"""CLI runner for detector Pfa Monte Carlo calibration experiment."""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ew_core.evaluation.detector_pfa_calibration import main

if __name__ == "__main__":
    main()

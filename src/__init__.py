# Compatibility alias for absolute 'src' imports used throughout the project.
# This file ensures that statements like "from src.contracts import ..." work when the
# package is executed as a module (e.g., `python -m cognitive_ew_smart_scan.src...`).
import sys
from pathlib import Path

# Resolve the actual source directory relative to the repository root.
_real_src = Path(__file__).resolve().parents[1] / "cognitive_ew_smart_scan" / "src"
# Prepend it to sys.path if not already present.
if str(_real_src) not in sys.path:
    sys.path.insert(0, str(_real_src))
# Expose as a namespace package so `import src.xxx` works
__path__ = [str(_real_src)]

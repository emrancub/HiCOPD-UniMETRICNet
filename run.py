"""HiCOPD-UniMETRICNet command entry point.

Runtime variables are set *before* importing any project module so CUDA/cuBLAS
reproducibility and CPU thread limits are established before PyTorch/NumPy load.
"""
from pathlib import Path
import os
import sys

# Safe defaults for Windows + Conda scientific stacks.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")
# Never silently enable Intel's unsupported duplicate-runtime bypass.
os.environ.pop("KMP_DUPLICATE_LIB_OK", None)

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hicopd_unimetricnet.cli import main

if __name__ == "__main__":
    main()

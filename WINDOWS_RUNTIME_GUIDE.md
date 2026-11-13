# Windows Runtime Guide

CPU audio/scientific-stack operations and PyTorch GPU operations are kept in separate fresh processes to avoid duplicate Intel OpenMP runtime crashes in Conda/Windows environments.

Launchers set:

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
MKL_THREADING_LAYER=SEQUENTIAL
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

The project does **not** enable `KMP_DUPLICATE_LIB_OK=TRUE`.

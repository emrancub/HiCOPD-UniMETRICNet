from __future__ import annotations
import warnings
warnings.filterwarnings('ignore', message='.*adaptive_max_pool2d_backward_cuda.*')
warnings.filterwarnings('ignore', message='.*Memory Efficient attention.*')
warnings.filterwarnings('ignore', message='.*Non-deterministic.*')
import hashlib, json, random
from pathlib import Path
import numpy as np


class DataIntegrityError(RuntimeError):
    pass


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(chunk), b''): h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str: return hashlib.sha256(data).hexdigest()


def ensure_dir(path: str | Path) -> Path:
    p=Path(path); p.mkdir(parents=True, exist_ok=True); return p


def write_json(path: str | Path, obj) -> None:
    p=Path(path); ensure_dir(p.parent); p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8')


def set_seed(seed: int) -> None:
    # Lazy PyTorch import keeps CPU audit/feature-extraction processes free of
    # the PyTorch OpenMP runtime on Windows.
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, 'enable_math_sdp'):
            torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)


def project_root() -> Path: return Path(__file__).resolve().parents[2]

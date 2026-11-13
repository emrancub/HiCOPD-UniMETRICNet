from pathlib import Path


def test_utils_has_no_top_level_torch_import():
    root=Path(__file__).resolve().parents[1]
    text=(root/'src/hicopd_unimetricnet/utils.py').read_text(encoding='utf-8')
    prefix=text.split('def set_seed',1)[0]
    assert 'import torch' not in prefix


def test_cli_has_no_top_level_torch_import():
    root=Path(__file__).resolve().parents[1]
    text=(root/'src/hicopd_unimetricnet/cli.py').read_text(encoding='utf-8')
    prefix=text.split('def _torch_info',1)[0]
    assert 'import torch' not in prefix


def test_features_lazily_imports_audio_stack():
    root=Path(__file__).resolve().parents[1]
    text=(root/'src/hicopd_unimetricnet/features.py').read_text(encoding='utf-8')
    prefix=text.split('def precompute',1)[0]
    assert 'from .audio import' not in prefix

from pathlib import Path


def root():
    return Path(__file__).resolve().parents[1]


def test_training_process_does_not_run_numpy_blas_domain_probe():
    text=(root()/"src/hicopd_unimetricnet/train.py").read_text(encoding="utf-8")
    assert "domain_probe(" not in text
    assert "pairwise_domain_geometry(" not in text
    assert "pending_cpu_postprocess" in text


def test_embedding_analysis_is_torch_free():
    text=(root()/"src/hicopd_unimetricnet/embedding_analysis.py").read_text(encoding="utf-8")
    assert "import torch" not in text
    assert "domain_probe" in text


def test_run_all_sets_cublas_before_python_and_uses_cpu_postprocess():
    text=(root()/"RUN_ALL.bat").read_text(encoding="utf-8")
    assert 'CUBLAS_WORKSPACE_CONFIG=:4096:8' in text
    assert 'run.py embedding-analysis' in text
    assert text.index('run.py embedding-analysis') < text.index('run.py summarize')
    assert 'KMP_DUPLICATE_LIB_OK=TRUE' not in text.upper()


def test_python_entrypoint_sets_runtime_before_cli_import():
    text=(root()/"run.py").read_text(encoding="utf-8")
    assert text.index('CUBLAS_WORKSPACE_CONFIG') < text.index('from hicopd_unimetricnet.cli import main')


def test_windows_test_suite_uses_subprocess_isolation_for_heavy_cpu_gpu_checks():
    cpu=(root()/"tests/test_cpu_audio.py").read_text(encoding="utf-8")
    gpu=(root()/"tests/test_gpu_core.py").read_text(encoding="utf-8")
    assert "subprocess.run" in cpu and "runtime_cpu_check.py" in cpu
    assert "subprocess.run" in gpu and "runtime_gpu_check.py" in gpu


def test_audio_has_mixed_runtime_blas_free_fallback():
    text=(root()/"src/hicopd_unimetricnet/audio.py").read_text(encoding="utf-8")
    assert "'torch' in sys.modules" in text
    assert "np.sum(sp32 * fb32" in text


def test_launchers_request_sequential_mkl_on_windows():
    for name in ["TEST_PROJECT.bat","VERIFY_RUNTIME_ISOLATION.bat","RUN_QUICK.bat","RUN_ALL.bat"]:
        text=(root()/name).read_text(encoding="utf-8")
        assert "MKL_THREADING_LAYER=SEQUENTIAL" in text

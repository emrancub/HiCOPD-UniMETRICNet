from pathlib import Path
import re


def test_launchers_and_entrypoints_exist():
    root=Path(__file__).resolve().parents[1]
    for x in ['FIX_ENVIRONMENT.bat','GPU_CHECK.bat','TEST_PROJECT.bat','CHECK_DATA.bat','RUN_QUICK.bat','RUN_QUICK_FIXED.bat','RUN_ALL.bat','RUN_PUBLICATION.bat','run.py','run_quick.py','run_all.py']:
        assert (root/x).exists(),x


def test_batch_files_do_not_reference_missing_python_scripts():
    root=Path(__file__).resolve().parents[1]
    pattern=re.compile(r"(?i)(?:^|[\s\"'])([A-Za-z0-9_\-]+\.py)(?=$|[\s\"'])")
    for bat in root.glob('*.bat'):
        text=bat.read_text(encoding='utf-8',errors='ignore')
        for m in pattern.finditer(text):
            name=m.group(1)
            assert (root/name).exists(),f'{bat.name} references missing {name}'


def test_quick_launcher_calls_run_py_not_nonexistent_script():
    root=Path(__file__).resolve().parents[1]; text=(root/'RUN_QUICK.bat').read_text(encoding='utf-8')
    assert 'run.py' in text

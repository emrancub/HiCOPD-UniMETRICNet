from pathlib import Path
from hicopd_unimetricnet.config import load_config
from hicopd_unimetricnet.dataset import taxonomy_id


def root(): return Path(__file__).resolve().parents[1]


def test_quick_and_full_outputs_are_separate():
    full=load_config('configs/main.yaml',False); quick=load_config('configs/main.yaml',True)
    assert full['project']['output_dir']!=quick['project']['output_dir']
    assert str(quick['project']['output_dir']).endswith('_quick')


def test_merit_strategy_is_taxonomy_aware_and_no_target_adaptation():
    cfg=load_config('configs/main.yaml',False)
    assert cfg['model']['taxonomy_classes']==5
    assert cfg['evaluation']['safe_stacking']['enabled'] is False
    assert cfg['evaluation']['style_threshold_adaptation']['enabled'] is False
    assert cfg['evaluation']['label_free_prior_adaptation']['enabled'] is False
    assert taxonomy_id('COPD')==0 and taxonomy_id('Healthy')==1
    assert taxonomy_id('Asthma')==2 and taxonomy_id('Pneumonia')==3 and taxonomy_id('ILD')==4


def test_full_launcher_has_completeness_before_summary():
    text=(root()/'RUN_ALL.bat').read_text(encoding='utf-8')
    assert 'run.py validate-results' in text
    assert text.index('run.py validate-results') < text.index('run.py summarize')


def test_interpretability_commands_exist():
    cli=(root()/'src/hicopd_unimetricnet/cli.py').read_text(encoding='utf-8')
    assert 'explainability' in cli and 'embedding-analysis' in cli

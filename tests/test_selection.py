from hicopd_unimetricnet.parsers import parse_kauh_labels

def test_kauh_selected_label_names_are_canonical():
    assert parse_kauh_labels('Heart Failure')==['Heart_Failure']
    assert parse_kauh_labels('N')==['Healthy']
    assert parse_kauh_labels('COPD')==['COPD']

def test_multi_diagnosis_is_preserved_not_silently_collapsed():
    assert parse_kauh_labels('COPD + Heart Failure')==['COPD','Heart_Failure']

from pathlib import Path
from hicopd_unimetricnet.audit import collect_kauh


def test_kauh_primary_prefers_D_and_falls_back_without_pseudoreplication(tmp_path):
    audio=tmp_path/'Audio Files'; audio.mkdir()
    # Patient 1 has B/D/E; patient 2 has B/E but no D.
    for name in [
        'BP1_N, 50, M.wav','DP1_N, 50, M.wav','EP1_N, 50, M.wav',
        'BP2_COPD, 60, F.wav','EP2_COPD, 60, F.wav',
    ]:
        (audio/name).touch()
    cfg={'data':{'kauh_root':str(tmp_path),'kauh_audio_dir':'Audio Files','kauh_keep':['Healthy','COPD'],'kauh_main_filter':'D'}}
    rows=collect_kauh(cfg)
    primary=[r for r in rows if r['main_role']==1]
    assert len(primary)==2
    by={r['person_id']:r for r in primary}
    assert by['KAUH_P001']['filter_mode']=='D'
    assert by['KAUH_P002']['filter_mode']=='B'
    assert 'fallback' in by['KAUH_P002']['selection_reason']
    # All selected-class exports remain available for the intervention experiment.
    assert sum(r['filter_intervention_role'] for r in rows)==5

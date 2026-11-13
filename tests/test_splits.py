import pandas as pd
from hicopd_unimetricnet.splits import patient_cv,assert_disjoint

def test_patient_cv_has_no_overlap():
    rows=[]
    for ds in ['ICBHI','HYBRID','BRACETS']:
        for i in range(20): rows.append(dict(person_id=f'{ds}{i}',logical_dataset=ds,copd_label=i%2,main_role=1))
    f=pd.DataFrame(rows); ss=patient_cv(f,5,42)
    for s in ss: assert not set(s['train_people'])&set(s['test_people'])

def test_disjoint_accepts_clean_groups(): assert_disjoint(['a'],['b'],['c'])

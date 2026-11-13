
import numpy as np
import pandas as pd
from hicopd_unimetricnet.controlled_validation import analyze_controlled_predictions

def test_controlled_validation_repeatability_and_locked_metrics():
    rows=[]
    for i in range(12):
        y=1 if i>=6 else 0
        base=.75 if y else .25
        for j,c in enumerate(["A","B","C"]):
            rows.append(dict(participant_id=f"P{i}",recording_id=f"P{i}_{c}",
                             measurement_condition=c,probability=base+(j-1)*.01,reference_label=y))
    df=pd.DataFrame(rows)
    _,per,cond,summary,overall=analyze_controlled_predictions(df,.5,50,3)
    assert len(per)==12
    assert summary["mean_within_participant_range"] < .05
    assert np.isfinite(summary["repeatability_coefficient_95"])
    assert overall["auroc"] > .99
    assert overall["sensitivity"] == 1.0
    assert overall["specificity"] == 1.0
    assert len(cond)==3

def test_controlled_validation_rejects_duplicate_recording_ids():
    df=pd.DataFrame([
        dict(participant_id="P1",recording_id="x",measurement_condition="A",probability=.2),
        dict(participant_id="P1",recording_id="x",measurement_condition="B",probability=.3),
    ])
    try:
        analyze_controlled_predictions(df,.5,10,1)
        assert False
    except ValueError as e:
        assert "unique" in str(e)

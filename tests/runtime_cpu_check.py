"""CPU-only numerical regression checks.

This script intentionally never imports torch.  It is launched as a child
process by pytest on Windows to keep NumPy/MKL/SciPy in a process that cannot
collide with PyTorch's OpenMP runtime.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src'
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
import numpy as np
import pandas as pd

from hicopd_unimetricnet.metrics import aggregate_prediction_views, robust_domain_threshold
from hicopd_unimetricnet.parsers import parse_kauh_filename, norm_bracets_id
from hicopd_unimetricnet.audio import (
    spectral_views, segment_quality_features, trim_silence_edges,
    spectral_gate_denoise, mel_filterbank,
)


def cfg():
    return {'audio':{'sample_rate':4000,'n_fft':512,'win_length':256,'hop_length':128,
        'n_mels':64,'fmin':60.0,'fmax':1800.0,'normalize_waveform':'robust',
        'normalize_logmel':'zscore','risn':{'enabled':True,'mad_floor':.15,'clip':5.0},
        'pcen':{'enabled':True,'alpha':.96,'delta':2.0,'root':2.0,'smooth':.04,'eps':1e-6},
        'quality':{'clipping_threshold':.98},
        'silence_trim':{'enabled':True,'frame_ms':100,'hop_ms':50,'relative_db':-38,
                        'max_trim_fraction_each_side':.15,'min_remaining_seconds':2},
        'denoise':{'enabled':True,'noise_quantile':.15,'reduction_strength':.5,'spectral_floor':.1}}}


def main():
    rows=[dict(person_id='one',subject_id='one',logical_dataset='BRACETS',origin_dataset='BRACETS',
               category='COPD',y=1,probability=p,age=60,sex='M') for p in [.2,.4,.9]]
    r=aggregate_prediction_views(pd.DataFrame(rows)); assert np.isclose(r.probability.iloc[0],.5)
    a=parse_kauh_filename('BP1_COPD,I E N,65,M.wav'); b=parse_kauh_filename('DP1_COPD,I E N,65,M.wav')
    assert a['subject_id']==b['subject_id']=='KAUH_P001'; assert a['filter_mode']=='B' and b['filter_mode']=='D'
    assert norm_bracets_id('1')=='01'
    c=cfg(); ac=c['audio']; rng=np.random.default_rng(1); x=rng.normal(0,.05,16000).astype(np.float32)
    fb=mel_filterbank(4000,512,64,60,1800); v=spectral_views(x,ac,fb); q=segment_quality_features(x,ac)
    assert v.shape[0]==3 and v.shape[1]==64 and q.shape==(8,) and np.isfinite(v).all() and np.isfinite(q).all()
    sr=4000; t=np.arange(sr*6)/sr
    sig=np.r_[np.zeros(sr//2),.1*np.sin(2*np.pi*250*t[:sr*5]),np.zeros(sr//2)].astype(np.float32)
    tr=trim_silence_edges(sig,sr,ac); dn=spectral_gate_denoise(tr,sr,ac)
    assert len(tr)>=sr*2 and len(dn)==len(tr) and np.isfinite(dn).all()
    f=pd.DataFrame({'y':[0,0,1,1,0,0,1,1],'probability':[.1,.2,.7,.8,.2,.3,.75,.9],
                    'origin_dataset':['A']*4+['B']*4})
    th=robust_domain_threshold(f,domain_col='origin_dataset',grid_points=31); assert 0.2<th<0.8
    import sys
    assert 'torch' not in sys.modules, 'CPU runtime check accidentally loaded torch'
    print('CPU-only UniMETRICNet runtime checks passed; torch loaded:', 'torch' in sys.modules)

if __name__=='__main__': main()

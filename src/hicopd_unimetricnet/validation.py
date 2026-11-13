from __future__ import annotations
from pathlib import Path
import json


def validate_results(cfg, strict=True):
    out=Path(cfg['project']['output_dir']); missing=[]; malformed=[]
    models=list(cfg['experiments']['core_models']); seeds=[int(x) for x in cfg['training']['seeds']]; targets=list(cfg['experiments']['lodo_targets'])
    for target in targets:
        for model in models:
            for seed in seeds:
                rd=out/'runs'/'lodo'/target/model/f'seed_{seed}'
                for f in ['metrics.json','test_predictions.csv','calibration_predictions.csv','test_embeddings.npz','calibration_embeddings.npz','best_model.pt']:
                    if not (rd/f).exists(): missing.append(str(rd/f))
                mp=rd/'metrics.json'
                if mp.exists():
                    try:
                        m=json.loads(mp.read_text(encoding='utf-8'))
                        if m.get('embedding_analysis_status')!='complete_cpu_isolated': malformed.append(f'{mp}: embedding analysis not complete')
                    except Exception as e: malformed.append(f'{mp}: {e}')
    # Secondary locked external must exist for every primary model/seed.
    ext=cfg['experiments']['locked_external_target']
    for model in models:
        for seed in seeds:
            mp=out/'runs'/'external'/ext/model/f'seed_{seed}'/'metrics.json'
            if not mp.exists(): missing.append(str(mp))
    report={'complete':not missing and not malformed,'missing_count':len(missing),'malformed_count':len(malformed),'missing':missing,'malformed':malformed,'targets':targets,'models':models,'seeds':seeds}
    (out/'result_completeness.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if report['complete']:
        print(f"Result completeness PASSED: {len(targets)} targets x {len(models)} models x {len(seeds)} seeds are present.")
    else:
        msg='Result completeness FAILED. This prevents a partial quick run from being mistaken for a publication run.\n'
        msg+=f'Missing={len(missing)}, malformed={len(malformed)}\n'
        if missing: msg+='First missing files:\n'+'\n'.join(missing[:20])+'\n'
        if malformed: msg+='First malformed runs:\n'+'\n'.join(malformed[:20])+'\n'
        if strict: raise RuntimeError(msg)
        print(msg)
    return report

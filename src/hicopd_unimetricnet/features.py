from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from .utils import ensure_dir, sha256_file, write_json


def precompute(cfg, condition='clean'):
    # CPU-only signal processing process: intentionally no torch import.
    from .audio import (read_mono, bandpass, apply_condition, frame_signal,
                        mel_filterbank, spectral_views, physiological_features,
                        segment_quality_features, trim_silence_edges,
                        spectral_gate_denoise)
    cache = Path(cfg['project']['cache_dir'])
    all_path = cache / 'recordings_all.csv'
    if not all_path.exists():
        raise RuntimeError('Run audit first')
    df = pd.read_csv(all_path, keep_default_na=False)
    use = ((df.main_role.astype(int).eq(1) | df.filter_intervention_role.astype(int).eq(1))
           & df.integrity_exclusion_reason.astype(str).eq(''))
    df = df[use].copy()
    acfg = cfg['audio']; sr = int(acfg['sample_rate'])
    length = int(round(float(acfg['segment_seconds']) * sr))
    stride = int(round(float(acfg['stride_seconds']) * sr))
    fb = mel_filterbank(sr, int(acfg['n_fft']), int(acfg['n_mels']), float(acfg['fmin']), float(acfg['fmax']))
    outdir = ensure_dir(cache / 'features' / condition)
    rows, skipped = [], []

    for r in tqdm(df.itertuples(index=False), total=len(df), desc=f'features:{condition}'):
        try:
            key = str(r.raw_sha256)[:16] if hasattr(r, 'raw_sha256') else sha256_file(r.audio_path)[:16]
            dest = outdir / f'{r.recording_id}_{key}.npz'
            if not dest.exists():
                x = read_mono(r.audio_path, sr)
                x = bandpass(x, sr, float(acfg['highpass_hz']), float(acfg['lowpass_hz']))
                x = apply_condition(x, sr, condition, r.recording_id)
                x = trim_silence_edges(x, sr, acfg)
                x = spectral_gate_denoise(x, sr, acfg)
                frames = frame_signal(x, length, stride, bool(acfg.get('drop_incomplete', True)))
                if not frames:
                    skipped.append(dict(recording_id=r.recording_id, audio_path=r.audio_path,
                                        reason='no_complete_segment_after_quality_preprocessing'))
                    continue
                views = np.stack([spectral_views(seg, acfg, fb) for seg in frames]).astype(np.float32)
                phys = np.stack([physiological_features(seg, acfg) for seg in frames]).astype(np.float32)
                quality = np.stack([segment_quality_features(seg, acfg) for seg in frames]).astype(np.float32)
                np.savez_compressed(dest, specs=views, phys=phys, quality=quality)
            with np.load(dest, allow_pickle=False) as z:
                n = int(len(z['specs']))
                pdim = int(z['phys'].shape[1])
                qdim = int(z['quality'].shape[1])
                channels = int(z['specs'].shape[1])
            d = r._asdict()
            d.update(feature_path=str(dest.resolve()), n_segments=n, phys_dim=pdim,
                     quality_dim=qdim, spectral_channels=channels, condition=condition,
                     feature_sha256=sha256_file(dest))
            rows.append(d)
        except Exception as exc:
            skipped.append(dict(recording_id=getattr(r, 'recording_id', ''),
                                audio_path=getattr(r, 'audio_path', ''),
                                reason=f'{type(exc).__name__}: {exc}'))
            raise
    out = pd.DataFrame(rows)
    path = cache / f'features_{condition}.csv'
    out.to_csv(path, index=False)
    pd.DataFrame(skipped).to_csv(cache / f'features_{condition}_skipped.csv', index=False)
    write_json(cache / f'features_{condition}.json', dict(
        condition=condition, rows=len(out), people=int(out.person_id.nunique()) if len(out) else 0,
        audio=acfg, phys_dim=int(out.phys_dim.iloc[0]) if len(out) else 0,
        quality_dim=int(out.quality_dim.iloc[0]) if len(out) else 0,
        spectral_channels=int(out.spectral_channels.iloc[0]) if len(out) else 0,
        skipped=len(skipped)))
    print(f'Saved {path} with {len(out)} recordings and {out.person_id.nunique() if len(out) else 0} people')
    return out


def load_feature_frame(cfg, condition='clean'):
    path = Path(cfg['project']['cache_dir']) / f'features_{condition}.csv'
    if not path.exists():
        raise RuntimeError(f'Missing {path}; run features --condition {condition}')
    df = pd.read_csv(path, keep_default_na=False)
    for p in df.feature_path.unique():
        if not Path(p).exists():
            raise RuntimeError(f'Missing feature cache: {p}')
    return df

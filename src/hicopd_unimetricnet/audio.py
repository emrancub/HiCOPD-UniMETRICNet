from __future__ import annotations
import math, hashlib, os, sys
from pathlib import Path
import numpy as np
try:
    import soundfile as sf
except ImportError:
    sf = None
from scipy.signal import butter, sosfiltfilt, sosfilt, resample_poly, hilbert, stft, istft
from scipy.stats import skew, kurtosis


def read_mono(path: str | Path, target_sr: int) -> np.ndarray:
    x, sr = sf.read(path, always_2d=True, dtype='float32')
    x = x.mean(axis=1)
    if sr != target_sr:
        g = math.gcd(int(sr), int(target_sr))
        x = resample_poly(x, target_sr // g, sr // g).astype(np.float32)
    return np.nan_to_num(x.astype(np.float32))


def bandpass(x: np.ndarray, sr: int, hp: float, lp: float) -> np.ndarray:
    ny = sr / 2
    hp = max(1e-3, min(hp, ny * 0.95))
    lp = max(hp + 1, min(lp, ny * 0.99))
    sos = butter(4, [hp / ny, lp / ny], btype='bandpass', output='sos')
    try:
        return sosfiltfilt(sos, x).astype(np.float32)
    except ValueError:
        return sosfilt(sos, x).astype(np.float32)


def robust_normalize_waveform(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)) * 1.4826)
    if mad < 1e-5:
        mad = float(np.std(x) + 1e-6)
    y = (x - med) / max(mad, 1e-6)
    return np.clip(y, -8.0, 8.0).astype(np.float32)


def trim_silence_edges(x: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    scfg = cfg.get('silence_trim', {})
    if not scfg.get('enabled', True) or len(x) < sr:
        return x.astype(np.float32)
    fl = max(8, int(round(float(scfg.get('frame_ms', 100)) * sr / 1000)))
    hop = max(4, int(round(float(scfg.get('hop_ms', 50)) * sr / 1000)))
    if len(x) < fl:
        return x.astype(np.float32)
    starts = np.arange(0, len(x) - fl + 1, hop, dtype=int)
    rms = np.asarray([np.sqrt(np.mean(x[s:s + fl] ** 2) + 1e-12) for s in starts])
    ref = float(np.quantile(rms, 0.90) + 1e-12)
    db = 20 * np.log10((rms + 1e-12) / ref)
    active = db >= float(scfg.get('relative_db', -38.0))
    if not active.any():
        return x.astype(np.float32)
    first = int(starts[np.argmax(active)])
    last_idx = len(active) - 1 - int(np.argmax(active[::-1]))
    last = int(min(len(x), starts[last_idx] + fl))
    max_trim = int(float(scfg.get('max_trim_fraction_each_side', 0.15)) * len(x))
    first = min(first, max_trim)
    last = max(last, len(x) - max_trim)
    min_len = int(float(scfg.get('min_remaining_seconds', 5.0)) * sr)
    if last - first < min_len:
        return x.astype(np.float32)
    return x[first:last].astype(np.float32)


def spectral_gate_denoise(x: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """Conservative minimum-statistics spectral gate.

    Only stationary background components estimated from low-energy time frames are
    attenuated. The operation is deliberately soft to avoid erasing wheeze/crackle energy.
    """
    dcfg = cfg.get('denoise', {})
    if not dcfg.get('enabled', True) or len(x) < int(0.5 * sr):
        return x.astype(np.float32)
    nper = min(256, max(64, int(sr * 0.064)))
    nover = nper // 2
    _, _, Z = stft(x.astype(np.float64), fs=sr, nperseg=nper, noverlap=nover,
                   boundary='zeros', padded=True)
    mag = np.abs(Z)
    power = mag ** 2
    frame_energy = power.mean(axis=0)
    q = float(np.clip(dcfg.get('noise_quantile', 0.15), 0.02, 0.40))
    k = max(1, int(np.ceil(q * len(frame_energy))))
    low = np.argsort(frame_energy)[:k]
    noise = np.median(power[:, low], axis=1, keepdims=True)
    strength = float(np.clip(dcfg.get('reduction_strength', 0.55), 0.0, 1.0))
    floor = float(np.clip(dcfg.get('spectral_floor', 0.10), 0.01, 0.5))
    gain = 1.0 - strength * noise / (power + noise + 1e-12)
    gain = np.clip(gain, floor, 1.0)
    Z2 = Z * gain
    _, y = istft(Z2, fs=sr, nperseg=nper, noverlap=nover, input_onesided=True)
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return np.asarray(y[:len(x)], np.float32)


def deterministic_rng(key: str) -> np.random.Generator:
    seed = int(hashlib.sha256(key.encode('utf-8')).hexdigest()[:16], 16) % (2**32 - 1)
    return np.random.default_rng(seed)


def apply_condition(x: np.ndarray, sr: int, condition: str, key: str) -> np.ndarray:
    if condition == 'clean':
        return x
    rng = deterministic_rng(key + '|' + condition)
    if condition.startswith('noise_'):
        db = float(condition.split('_')[1].replace('db', ''))
        power = float(np.mean(x * x) + 1e-12)
        noise_power = power / (10 ** (db / 10))
        return (x + rng.normal(0, np.sqrt(noise_power), len(x))).astype(np.float32)
    if condition == 'bandlimit_800':
        ny = sr / 2
        sos = butter(6, min(800 / ny, .99), btype='lowpass', output='sos')
        try:
            return sosfiltfilt(sos, x).astype(np.float32)
        except ValueError:
            return sosfilt(sos, x).astype(np.float32)
    if condition == 'clipping_050':
        peak = float(np.max(np.abs(x)) + 1e-12)
        limit = .50 * peak
        return np.clip(x, -limit, limit).astype(np.float32)
    raise ValueError(f'Unknown condition: {condition}')


def frame_signal(x: np.ndarray, length: int, stride: int, drop_incomplete: bool = True):
    if len(x) < length:
        if drop_incomplete:
            return []
        return [np.pad(x, (0, length - len(x)))]
    starts = list(range(0, len(x) - length + 1, stride))
    frames = [x[s:s + length].copy() for s in starts]
    if not drop_incomplete and starts and starts[-1] + length < len(x):
        tail = x[starts[-1] + stride:]
        if len(tail):
            frames.append(np.pad(tail, (0, length - len(tail))))
    return frames


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    m = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(m)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        l, c, r = bins[i:i + 3]
        if c <= l:
            c = min(l + 1, n_fft // 2)
        if r <= c:
            r = min(c + 1, n_fft // 2)
        if c > l:
            fb[i, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            fb[i, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb


def _stft_power(x, sr, n_fft, win, hop):
    window = np.hanning(win).astype(np.float32)
    n = 1 + max(0, (len(x) - win) // hop)
    if n <= 0:
        x = np.pad(x, (0, max(0, win - len(x))))
        n = 1
    spec = np.empty((n_fft // 2 + 1, n), dtype=np.float32)
    for j in range(n):
        chunk = x[j * hop:j * hop + win]
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        fft = np.fft.rfft(chunk * window, n=n_fft)
        spec[:, j] = (fft.real ** 2 + fft.imag ** 2).astype(np.float32)
    return spec


def _zscore2d(x):
    return ((x - x.mean()) / (x.std() + 1e-6)).astype(np.float32)


def response_invariant_normalize(x: np.ndarray, cfg: dict) -> np.ndarray:
    rcfg = cfg.get('risn', {})
    if not rcfg.get('enabled', True):
        return x.astype(np.float32)
    med = np.median(x, axis=-1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=-1, keepdims=True) * 1.4826
    mad = np.maximum(mad, float(rcfg.get('mad_floor', .15)))
    z = (x - med) / mad
    z = np.clip(z, -float(rcfg.get('clip', 5.0)), float(rcfg.get('clip', 5.0)))
    return _zscore2d(z)


def pcen_from_mel(mel: np.ndarray, cfg: dict) -> np.ndarray:
    p = cfg.get('pcen', {})
    if not p.get('enabled', True):
        return _zscore2d(np.log(mel + 1e-6))
    alpha = float(p.get('alpha', .96))
    delta = float(p.get('delta', 2.0))
    root = float(p.get('root', 2.0))
    smooth = float(p.get('smooth', .04))
    eps = float(p.get('eps', 1e-6))
    M = np.empty_like(mel, dtype=np.float32)
    M[:, 0] = mel[:, 0]
    for t in range(1, mel.shape[1]):
        M[:, t] = (1.0 - smooth) * M[:, t - 1] + smooth * mel[:, t]
    pcen = (mel / (eps + M) ** alpha + delta) ** (1.0 / root) - delta ** (1.0 / root)
    return _zscore2d(pcen)


def spectral_views(segment: np.ndarray, cfg: dict, fb=None) -> np.ndarray:
    """Return [log-Mel, PCEN, RISN] channels."""
    sr = int(cfg['sample_rate'])
    n_fft = int(cfg['n_fft'])
    win = int(cfg['win_length'])
    hop = int(cfg['hop_length'])
    x = segment.astype(np.float32)
    if cfg.get('normalize_waveform') == 'robust':
        x = robust_normalize_waveform(x)
    elif cfg.get('normalize_waveform') == 'zscore':
        x = (x - x.mean()) / (x.std() + 1e-6)
    spec = _stft_power(x, sr, n_fft, win, hop)
    if fb is None:
        fb = mel_filterbank(sr, n_fft, int(cfg['n_mels']), float(cfg['fmin']), float(cfg['fmax']))
    # NumPy linked against Intel MKL and PyTorch may each load an OpenMP runtime on
    # Windows.  In the production feature process PyTorch is intentionally absent, so
    # the fast BLAS path is safe.  If this helper is ever called from a mixed
    # PyTorch+scientific-stack process (for example an IDE or a monolithic pytest
    # session), avoid BLAS completely to prevent Windows libiomp5md.dll Error #15.
    if os.name == 'nt' and 'torch' in sys.modules:
        fb32 = np.asarray(fb, dtype=np.float32)
        sp32 = np.asarray(spec, dtype=np.float32)
        mel = np.empty((fb32.shape[0], sp32.shape[1]), dtype=np.float32)
        for mi in range(fb32.shape[0]):
            mel[mi] = np.sum(sp32 * fb32[mi, :, None], axis=0, dtype=np.float32)
        mel = np.maximum(mel, 1e-10).astype(np.float32)
    else:
        mel = np.maximum(fb @ spec, 1e-10).astype(np.float32)
    logm = np.log(mel + 1e-6).astype(np.float32)
    if cfg.get('normalize_logmel') == 'zscore':
        logm = _zscore2d(logm)
    pcen = pcen_from_mel(mel, cfg)
    risn = response_invariant_normalize(logm, cfg)
    return np.stack([logm, pcen, risn], axis=0).astype(np.float32)


def segment_quality_features(segment: np.ndarray, cfg: dict) -> np.ndarray:
    """Eight transparent signal-quality descriptors, all segment-local and label-free."""
    sr = int(cfg['sample_rate'])
    x = np.asarray(segment, np.float64)
    if len(x) == 0:
        return np.zeros(8, np.float32)
    peak = float(np.max(np.abs(x)) + 1e-12)
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    thr = max(rms * 0.20, peak * 0.03, 1e-8)
    active = float(np.mean(np.abs(x) > thr))
    clip = float(np.mean(np.abs(x) >= float(cfg.get('quality', {}).get('clipping_threshold', .98)) * peak))
    n_fft = int(cfg['n_fft']); win = int(cfg['win_length']); hop = int(cfg['hop_length'])
    p = _stft_power(robust_normalize_waveform(x.astype(np.float32)), sr, n_fft, win, hop).astype(np.float64)
    ps = p.mean(axis=1) + 1e-12
    f = np.linspace(0, sr / 2, len(ps))
    total = float(ps.sum() + 1e-12)
    flat = float(np.exp(np.mean(np.log(ps))) / (np.mean(ps) + 1e-12))
    q = ps / total
    ent = float(-(q * np.log(q + 1e-12)).sum() / np.log(len(q) + 1e-12))
    low = float(ps[(f >= 60) & (f < 250)].sum() / total)
    high = float(ps[(f >= 900) & (f <= 1800)].sum() / total)
    # Conservative SNR proxy: upper/lower STFT frame-energy quantiles.
    fe = p.mean(axis=0) + 1e-12
    snr_proxy = float(10 * np.log10(np.quantile(fe, .80) / np.quantile(fe, .20))) if len(fe) > 2 else 0.0
    log_rms = float(np.log10(rms + 1e-8))
    vec = np.asarray([active, log_rms, clip, flat, ent, low, high, snr_proxy / 30.0], np.float32)
    return np.nan_to_num(np.clip(vec, -5, 5), nan=0.0, posinf=5.0, neginf=-5.0)


def physiological_features(segment: np.ndarray, cfg: dict) -> np.ndarray:
    """Twenty-two interpretable respiratory-acoustic features."""
    sr = int(cfg['sample_rate']); n_fft = int(cfg['n_fft']); win = int(cfg['win_length']); hop = int(cfg['hop_length'])
    x = segment.astype(np.float64); x = (x - x.mean()) / (x.std() + 1e-8)
    p = _stft_power(x.astype(np.float32), sr, n_fft, win, hop).astype(np.float64)
    ps = p.mean(axis=1) + 1e-12; f = np.linspace(0, sr / 2, len(ps)); total = ps.sum() + 1e-12
    bands = [(60, 200), (200, 400), (400, 800), (800, 1200), (1200, 1800)]
    bf = []
    for lo, hi in bands:
        m = (f >= lo) & (f < hi); bf.append(float(ps[m].sum() / total))
    centroid = float((f * ps).sum() / total)
    bandwidth = float(np.sqrt((((f - centroid) ** 2) * ps).sum() / total))
    cdf = np.cumsum(ps) / total
    roll85 = float(f[min(np.searchsorted(cdf, .85), len(f) - 1)])
    roll95 = float(f[min(np.searchsorted(cdf, .95), len(f) - 1)])
    gm = float(np.exp(np.mean(np.log(ps + 1e-12))))
    flatness = gm / (float(ps.mean()) + 1e-12)
    q = ps / total
    entropy = float(-(q * np.log(q + 1e-12)).sum() / np.log(len(q) + 1e-12))
    dom = float(f[int(np.argmax(ps))])
    m = (f >= 60) & (f <= 1800)
    slope = float(np.polyfit(f[m] / 1000.0, np.log(ps[m] + 1e-12), 1)[0]) if m.sum() > 2 else 0.0
    rms = float(np.sqrt(np.mean(x * x)))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x))).astype(np.float64)))
    crest = float(np.max(np.abs(x)) / (rms + 1e-8))
    sk = float(skew(x, bias=False)) if len(x) > 8 else 0.0
    ku = float(kurtosis(x, fisher=True, bias=False)) if len(x) > 8 else 0.0
    env = np.abs(hilbert(x)); env_mean = float(env.mean()); env_cv = float(env.std() / (env_mean + 1e-8)); env_sk = float(skew(env, bias=False)) if len(env) > 8 else 0.0
    ef = np.fft.rfftfreq(len(env), 1 / sr); ep = np.abs(np.fft.rfft(env - env.mean())) ** 2
    mm = (ef >= 0.1) & (ef <= 1.5); mod_peak = float(ef[mm][np.argmax(ep[mm])]) if mm.any() else 0.0
    norm = p / (np.sqrt((p * p).sum(axis=0, keepdims=True)) + 1e-12)
    flux = float(np.sqrt(np.diff(norm, axis=1) ** 2).sum(axis=0).mean()) if norm.shape[1] > 1 else 0.0
    vec = np.asarray(bf + [centroid / 2000.0, bandwidth / 2000.0, roll85 / 2000.0, roll95 / 2000.0,
                          flatness, entropy, dom / 2000.0, slope, rms, zcr, crest / 10.0, sk / 5.0,
                          ku / 10.0, env_cv, env_sk / 5.0, mod_peak / 1.5, flux], dtype=np.float32)
    return np.nan_to_num(np.clip(vec, -10, 10), nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)

"""
Heart-rate extraction strategies, benchmarked against each other.

THE PROBLEM
  On subject47 the model produced a good pulse waveform (MACC 0.97) whose spectrum
  peaks at the true rate, yet the reported HR was 8.8 BPM off. A plain argmax over
  a short periodogram picked a weaker neighbouring peak. The model was right; the
  extraction step was wrong.

Every function here takes a raw 1-D signal and the sampling rate, and returns BPM.
They are deliberately independent so a benchmark can pick a winner on evidence.
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, periodogram, welch
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve

LOW_HZ, HIGH_HZ = 0.75, 2.5          # 45-150 BPM, the band used by the toolbox
LOW_BPM, HIGH_BPM = LOW_HZ * 60, HIGH_HZ * 60


# ---------------------------------------------------------------------------
# shared preprocessing
# ---------------------------------------------------------------------------
def detrend(x, lam=100):
    """Smoothness-priors detrending (Tarvainen et al.), as used by the toolbox."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n < 3:
        return x
    d = np.vstack([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)])
    D = diags(d, [0, 1, 2], shape=(n - 2, n), format="csc")
    A = eye(n, format="csc") + (lam ** 2) * (D.T @ D)
    return x - spsolve(A, x)


def bandpass(x, fs, order=1):
    b, a = butter(order, [LOW_HZ / fs * 2, HIGH_HZ / fs * 2], btype="bandpass")
    return filtfilt(b, a, np.double(x))


def clean(x, fs, lam=100, order=1):
    return bandpass(detrend(x, lam), fs, order)


def _next_pow2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _band(f, p):
    m = (f >= LOW_HZ) & (f <= HIGH_HZ)
    return f[m], p[m]


# ---------------------------------------------------------------------------
# 1. baseline - exactly what the repo does
# ---------------------------------------------------------------------------
def hr_baseline(x, fs):
    """detrend -> bandpass -> periodogram(nfft=next_pow2) -> argmax. The status quo."""
    s = clean(x, fs)
    f, p = periodogram(s, fs=fs, nfft=_next_pow2(len(s)), detrend=False)
    f, p = _band(f, p)
    return float(f[np.argmax(p)] * 60)


# ---------------------------------------------------------------------------
# 2. zero-padded periodogram - finer peak localisation
# ---------------------------------------------------------------------------
def hr_zeropad(x, fs, nfft=1 << 16):
    """Same as baseline but heavily zero-padded, so the peak is located precisely
    instead of snapping to a coarse bin."""
    s = clean(x, fs)
    f, p = periodogram(s, fs=fs, nfft=nfft, detrend=False)
    f, p = _band(f, p)
    return float(f[np.argmax(p)] * 60)


# ---------------------------------------------------------------------------
# 3. Welch - averaged spectrum, lower variance
# ---------------------------------------------------------------------------
def hr_welch(x, fs, seg_s=12.0, overlap=0.5):
    """Split into overlapping segments and average their spectra. Averaging
    suppresses noise peaks that happen to be tall in one part of the recording."""
    s = clean(x, fs)
    nper = int(min(len(s), seg_s * fs))
    f, p = welch(s, fs=fs, nperseg=nper, noverlap=int(nper * overlap),
                 nfft=max(1 << 14, _next_pow2(nper)), detrend=False)
    f, p = _band(f, p)
    return float(f[np.argmax(p)] * 60)


# ---------------------------------------------------------------------------
# 4. harmonic-support scoring
# ---------------------------------------------------------------------------
def hr_harmonic(x, fs, nfft=1 << 16, n_harm=2, weight=0.5):
    """Score each candidate peak by its own power PLUS the power at its harmonics.

    A real pulse has energy at f0 AND at 2*f0 (the dicrotic notch / waveform shape).
    A spurious neighbouring peak usually does not. This is the most direct fix for
    the observed failure: it lets a slightly lower peak win if its harmonics agree.
    """
    s = clean(x, fs)
    f, p = periodogram(s, fs=fs, nfft=nfft, detrend=False)
    fb, pb = _band(f, p)
    if pb.max() <= 0:
        return float("nan")
    pb = pb / pb.max()

    # candidate peaks in the band
    idx, _ = find_peaks(pb)
    if len(idx) == 0:
        return float(fb[np.argmax(pb)] * 60)

    def power_at(freq):
        j = int(np.argmin(np.abs(f - freq)))
        lo, hi = max(0, j - 2), min(len(p), j + 3)
        return p[lo:hi].max()

    pmax = p.max() if p.max() > 0 else 1.0
    best, best_score = None, -np.inf
    for i in idx:
        f0 = fb[i]
        score = pb[i]
        for h in range(2, n_harm + 2):
            if f0 * h <= fs / 2:
                score += weight ** (h - 1) * (power_at(f0 * h) / pmax)
        if score > best_score:
            best_score, best = score, f0
    return float(best * 60)


# ---------------------------------------------------------------------------
# 5. autocorrelation
# ---------------------------------------------------------------------------
def hr_autocorr(x, fs):
    """Dominant repeat interval in the time domain. Insensitive to the spectral
    leakage that misleads an FFT argmax."""
    s = clean(x, fs)
    s = s - s.mean()
    ac = np.correlate(s, s, mode="full")[len(s) - 1:]
    if ac[0] > 0:
        ac = ac / ac[0]
    lo = int(fs * 60.0 / HIGH_BPM)       # shortest plausible beat interval
    hi = int(fs * 60.0 / LOW_BPM)
    hi = min(hi, len(ac) - 1)
    if hi <= lo:
        return float("nan")
    seg = ac[lo:hi + 1]
    lag = lo + int(np.argmax(seg))
    return float(60.0 * fs / lag)


# ---------------------------------------------------------------------------
# 6. time-domain peak counting
# ---------------------------------------------------------------------------
def hr_peakcount(x, fs):
    """Median inter-beat interval from detected waveform peaks."""
    s = clean(x, fs)
    dist = int(fs * 60.0 / HIGH_BPM)
    peaks, _ = find_peaks(s, distance=max(dist, 1), prominence=0.3 * np.std(s))
    if len(peaks) < 3:
        return float("nan")
    return float(60.0 / (np.median(np.diff(peaks)) / fs))


# ---------------------------------------------------------------------------
# 7. sliding-window FFT, aggregated
# ---------------------------------------------------------------------------
def _window_hrs(x, fs, win_s, step_s, fn):
    s = np.asarray(x, dtype=np.float64)
    w, st = int(win_s * fs), int(step_s * fs)
    out = []
    for a in range(0, max(len(s) - w, 0) + 1, st):
        v = fn(s[a:a + w], fs)
        if np.isfinite(v):
            out.append(v)
    if not out:
        v = fn(s, fs)
        return [v] if np.isfinite(v) else []
    return out


def hr_window_median(x, fs, win_s=20.0, step_s=5.0):
    """HR on overlapping windows, then the median. One corrupted stretch of the
    recording can no longer drag the whole estimate away."""
    hrs = _window_hrs(x, fs, win_s, step_s, hr_zeropad)
    return float(np.median(hrs)) if hrs else float("nan")


def hr_window_harmonic_median(x, fs, win_s=20.0, step_s=5.0):
    """Windowed median, with harmonic-aware peak picking inside each window."""
    hrs = _window_hrs(x, fs, win_s, step_s, hr_harmonic)
    return float(np.median(hrs)) if hrs else float("nan")


# ---------------------------------------------------------------------------
# 8. consensus of independent estimators
# ---------------------------------------------------------------------------
def hr_consensus(x, fs):
    """Median of several independent methods. No single failure mode dominates."""
    votes = [hr_harmonic(x, fs), hr_welch(x, fs), hr_autocorr(x, fs),
             hr_window_median(x, fs)]
    votes = [v for v in votes if np.isfinite(v)]
    return float(np.median(votes)) if votes else float("nan")


METHODS = {
    "baseline (repo)":        hr_baseline,
    "zero-padded FFT":        hr_zeropad,
    "Welch":                  hr_welch,
    "harmonic support":       hr_harmonic,
    "autocorrelation":        hr_autocorr,
    "peak counting":          hr_peakcount,
    "window median":          hr_window_median,
    "window+harmonic median": hr_window_harmonic_median,
    "consensus":              hr_consensus,
}

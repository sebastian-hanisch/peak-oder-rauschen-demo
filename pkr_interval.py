"""Kern: Konfidenzintervalle des Spektrums (χ² und Jackknife) mit eigenen Quantilfunktionen (regularisierte Gamma- und Betafunktion), Welch-Schätzer.

Reine Funktionen, kein Zufall, kein Streamlit; nur numpy und math. scipy dient nur als Orakel in den Tests.
"""

import math

import numpy as np

_EPS = 1e-14
_TINY = 1e-300


def gammainc(a: float, x: float) -> float:
    """Regularisierte untere unvollständige Gammafunktion P(a, x): Reihe für x < a + 1, sonst Kettenbruch (Lentz)."""
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:
        term = total = 1.0 / a
        n = a
        for _ in range(10000):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) < abs(total) * _EPS:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b = x + 1.0 - a
    c = 1.0 / _TINY
    d = 1.0 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = _TINY if abs(d) < _TINY else d
        c = b + an / c
        c = _TINY if abs(c) < _TINY else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_cdf(x: float, nu: float) -> float:
    return gammainc(nu / 2.0, x / 2.0)


def chi2_ppf(p: float, nu: float) -> float:
    """Quantil der χ²-Verteilung mit nu Freiheitsgraden (Bisektion über die Verteilungsfunktion)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p muss in (0, 1) liegen")
    lo, hi = 0.0, max(nu, 1.0)
    while chi2_cdf(hi, nu) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if chi2_cdf(mid, nu) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def betainc(a: float, b: float, x: float) -> float:
    """Regularisierte unvollständige Betafunktion I_x(a, b) (Kettenbruch nach Lentz)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = _TINY if abs(d) < _TINY else d
    d = 1.0 / d
    h = d
    for m in range(1, 10000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = _TINY if abs(d) < _TINY else d
        c = 1.0 + aa / c
        c = _TINY if abs(c) < _TINY else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = _TINY if abs(d) < _TINY else d
        c = 1.0 + aa / c
        c = _TINY if abs(c) < _TINY else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def t_cdf(t: float, nu: float) -> float:
    """Verteilungsfunktion der Student-t-Verteilung mit nu Freiheitsgraden."""
    tail = 0.5 * betainc(nu / 2.0, 0.5, nu / (nu + t * t))
    return 1.0 - tail if t > 0 else tail


def t_ppf(p: float, nu: float) -> float:
    """Quantil der Student-t-Verteilung (Bisektion)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p muss in (0, 1) liegen")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -t_ppf(1.0 - p, nu)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, nu) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, nu) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def hann_window(n: int) -> np.ndarray:
    """Periodisches Hann-Fenster der Länge n."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def welch(x, seg: int, hop: int):
    """Welch mit Hann-Fenster (Zeilen von x): (Schätzer, Zahl K der Segmente, Segment-Spektren runs x K x Bins). Je Segment Mittelwert abgezogen, Spektrum |FFT(w y)|^2 / sum w^2."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    if seg > n or hop < 1:
        raise ValueError("Segment länger als die Reihe oder Schrittweite zu klein")
    w = hann_window(seg)
    starts = list(range(0, n - seg + 1, hop))
    segs = np.stack([x[..., s:s + seg] for s in starts], axis=-2)
    segs = segs - segs.mean(axis=-1, keepdims=True)
    spec = np.abs(np.fft.rfft(segs * w, axis=-1)) ** 2 / float(w @ w)
    return spec.mean(axis=-2), len(starts), spec


def welch_dof(k: int) -> float:
    """Freiheitsgrade des Welch-Schätzers bei Hann-Fenster und 50 % Überlappung mit K Segmenten: 36 K^2 / (19 K - 1)."""
    return 36.0 * k * k / (19.0 * k - 1.0)


def chi2_interval(est, nu: float, level: float = 0.95):
    """Konfidenzintervall nu * est / chi2_(1 - a/2) bis nu * est / chi2_(a/2) für das Spektrum (est > 0), a = 1 - level."""
    a = 1.0 - level
    est = np.asarray(est, dtype=float)
    return nu * est / chi2_ppf(1.0 - a / 2.0, nu), nu * est / chi2_ppf(a / 2.0, nu)


def jackknife_interval(values, level: float = 0.95):
    """Jackknife-Intervall (Pseudowerte des Logarithmus des Mittels über die K Segmente, Student-t mit K - 1 Freiheitsgraden). values: Zeilen runs x K Segmentwerte (> 0)."""
    values = np.asarray(values, dtype=float)
    k = values.shape[-1]
    total = values.sum(axis=-1, keepdims=True)
    log_all = np.log(total / k)
    log_loo = np.log((total - values) / (k - 1))
    pseudo = k * log_all - (k - 1) * log_loo
    est = pseudo.mean(axis=-1)
    se = pseudo.std(axis=-1, ddof=1) / math.sqrt(k)
    q = t_ppf(1.0 - (1.0 - level) / 2.0, k - 1)
    return np.exp(est - q * se), np.exp(est + q * se)


def width_db(lo, hi) -> np.ndarray:
    """Breite eines Intervalls in dB: 10 log10 (hi / lo)."""
    return 10.0 * np.log10(np.asarray(hi, dtype=float) / np.asarray(lo, dtype=float))

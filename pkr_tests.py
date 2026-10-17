"""Kern: Periodogramm an den inneren Fourier-Frequenzen, Schwellen für den höchsten Gipfel (Einzelfrequenz, Bonferroni, Šidák, Fishers g) und Hintergrundmodelle (AR(1), lokaler Mittelwert).

Reine Funktionen, kein Zufall, kein Streamlit; nur numpy und math. Statistik: I_k = |X_k|^2 / n der mittelwertfreien Reihe an k = 1 .. m, m = (n - 1) // 2 (Gleichanteil und Nyquist
ausgeschlossen). Unter weißem Gaußrauschen der Varianz sigma^2 sind die I_k / sigma^2 unabhängig Exp(1).
"""

import math

import numpy as np


def periodogram_inner(x) -> np.ndarray:
    """Periodogramm |X_k|^2 / n der mittelwertfreien Zeilen von x an den inneren Fourier-Frequenzen k = 1 .. (n - 1) // 2."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    if n < 4:
        raise ValueError("mindestens vier Werte nötig")
    x = x - x.mean(axis=-1, keepdims=True)
    return (np.abs(np.fft.rfft(x, axis=-1)) ** 2 / n)[..., 1:(n - 1) // 2 + 1]


def inner_frequencies(n: int) -> np.ndarray:
    """Frequenzen k / n der inneren Fourier-Frequenzen (Zyklen je Abtastwert)."""
    return np.arange(1, (n - 1) // 2 + 1) / n


def fisher_g(e) -> np.ndarray:
    """Fishers g = größter Wert geteilt durch die Summe (entlang der letzten Achse)."""
    e = np.asarray(e, dtype=float)
    return e.max(axis=-1) / e.sum(axis=-1)


def fisher_p(g: float, m: int) -> float:
    """Exakte Überschreitungswahrscheinlichkeit P(g' > g) von Fishers g bei m unabhängigen Exp-Werten (Fisher 1929): sum_j (-1)^(j-1) C(m, j) (1 - j g)^(m - 1), j = 1 .. 1/g.
    Die Terme laufen über Logarithmen der Gammafunktion und werden mit fsum summiert (Fehler unter 1e-12 bis m = 4095)."""
    if m < 1:
        raise ValueError("mindestens eine Frequenz")
    if g >= 1.0:
        return 0.0
    if g <= 1.0 / m:
        return 1.0
    terms = []
    for j in range(1, min(m, int(math.floor(1.0 / g))) + 1):
        base = 1.0 - j * g
        if base <= 0.0:
            continue
        log_term = math.lgamma(m + 1) - math.lgamma(j + 1) - math.lgamma(m - j + 1) + (m - 1) * math.log(base)
        terms.append((-1) ** (j - 1) * math.exp(log_term))
    return min(1.0, max(0.0, math.fsum(terms)))


def fisher_crit(alpha: float, m: int) -> float:
    """Kritischer Wert g_alpha mit P(g' > g_alpha) = alpha (Bisektion über fisher_p); m = 1: 1 (g ist dann immer 1)."""
    if m == 1:
        return 1.0
    lo, hi = 1.0 / m, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if fisher_p(mid, m) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def single_threshold(alpha: float) -> float:
    """Schwelle z für I / sigma^2 an einer festen Frequenz: P(Exp(1) > z) = alpha."""
    return -math.log(alpha)


def bonferroni_threshold(alpha: float, m: int) -> float:
    """Schwelle für das Maximum über m Frequenzen nach Bonferroni: Niveau alpha / m je Frequenz."""
    return -math.log(alpha / m)


def sidak_per_frequency(alpha: float, m: int) -> float:
    """Niveau je Frequenz nach Šidák: 1 - (1 - alpha)^(1 / m)."""
    return -math.expm1(math.log1p(-alpha) / m)


def sidak_threshold(alpha: float, m: int) -> float:
    """Schwelle für das Maximum über m unabhängige Exp(1)-Werte (Šidák): -ln(1 - (1 - alpha)^(1 / m))."""
    return -math.log(sidak_per_frequency(alpha, m))


def family_p(p_single: float, m: int) -> float:
    """p-Wert des Maximums über m unabhängige Frequenzen bei Einzel-p-Wert p_single (Šidák): 1 - (1 - p)^m."""
    if p_single >= 1.0:
        return 1.0
    return -math.expm1(m * math.log1p(-p_single))


def ar1_fit(x) -> tuple[np.ndarray, np.ndarray]:
    """AR(1)-Anpassung (Yule-Walker) je Zeile: phi = r1 / r0 (auf ±0,999 begrenzt), Innovationsvarianz r0 (1 - phi^2); r_k mit Teilung durch n."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean(axis=-1, keepdims=True)
    r0 = (x * x).mean(axis=-1)
    r1 = (x[..., 1:] * x[..., :-1]).sum(axis=-1) / x.shape[-1]
    phi = np.clip(r1 / r0, -0.999, 0.999)
    return phi, r0 * (1.0 - phi ** 2)


def ar1_density(phi, sigma2, n: int) -> np.ndarray:
    """Spektrum des AR(1) an den inneren Fourier-Frequenzen: sigma2 / |1 - phi exp(-i w)|^2, w = 2 pi k / n (zeilenweise, wenn phi und sigma2 Felder sind)."""
    phi = np.asarray(phi, dtype=float)[..., None]
    sigma2 = np.asarray(sigma2, dtype=float)[..., None]
    w = 2.0 * np.pi * inner_frequencies(n)
    return sigma2 / np.abs(1.0 - phi * np.exp(-1j * w)) ** 2


def local_mean_background(e, width: int = 11) -> np.ndarray:
    """Mittel der width - 1 Nachbarn jedes Bins (ohne den Bin selbst; Ränder an den Enden gespiegelt), entlang der letzten Achse."""
    e = np.asarray(e, dtype=float)
    pad = width // 2
    if e.shape[-1] <= pad:
        raise ValueError("zu wenige Frequenzen für die Breite")
    ep = np.pad(e, [(0, 0)] * (e.ndim - 1) + [(pad, pad)], mode="reflect")
    win = np.lib.stride_tricks.sliding_window_view(ep, width, axis=-1)
    return (win.sum(axis=-1) - e) / (width - 1)


def local_p(ratio, width: int = 11):
    """Einzel-p-Wert des Quotienten I / (Mittel von N = width - 1 unabhängigen Nachbarn) bei gleichem Spektrum: (1 + r / N)^(-N)."""
    n_nb = width - 1
    return (1.0 + np.asarray(ratio, dtype=float) / n_nb) ** (-n_nb)


def local_threshold(alpha: float, m: int, width: int = 11) -> float:
    """Schwelle für das Maximum des Quotienten I / lokaler Mittelwert über m Frequenzen (Šidák-Niveau je Frequenz, Verteilung (1 + r / N)^(-N))."""
    n_nb = width - 1
    return n_nb * (sidak_per_frequency(alpha, m) ** (-1.0 / n_nb) - 1.0)


def median_background(e, width: int = 11) -> np.ndarray:
    """Naiver Hintergrund: gleitender Median (Breite width, Ränder gespiegelt), auf den Mittelwert skaliert (der Median von Exp(1) ist ln 2)."""
    e = np.asarray(e, dtype=float)
    pad = width // 2
    ep = np.pad(e, [(0, 0)] * (e.ndim - 1) + [(pad, pad)], mode="reflect")
    win = np.lib.stride_tricks.sliding_window_view(ep, width, axis=-1)
    return np.median(win, axis=-1) / math.log(2.0)

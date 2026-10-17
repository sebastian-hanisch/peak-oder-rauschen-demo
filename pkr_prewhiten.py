"""Kern: Vorfilterung (Burg-Anpassung eines AR-Modells und Residuen). Reine Funktionen, nur numpy."""

import numpy as np


def burg(x, p: int):
    """Burg: Reflexionskoeffizient je Stufe aus der Summe der Vorwärts- und Rückwärtsfehlerquadrate. Rückgabe (a, Fehlervarianzen 0..p, Reflexionskoeffizienten)."""
    x = np.asarray(x, dtype=float)
    ef, eb = x.copy(), x.copy()
    a = np.zeros(0)
    err = [float((x ** 2).mean())]
    refl = []
    for m in range(1, p + 1):
        efp, ebp = ef[1:], eb[:-1]
        denom = (efp ** 2).sum() + (ebp ** 2).sum()
        k = 2.0 * (efp * ebp).sum() / denom if denom > 0 else 0.0
        a = np.r_[a - k * a[::-1], k] if m > 1 else np.array([k])
        ef, eb = efp - k * ebp, ebp - k * efp
        err.append(err[-1] * (1.0 - k * k))
        refl.append(float(k))
    return a, np.array(err), np.array(refl)


def residuals(x, a) -> np.ndarray:
    """Vorhersagefehler e[t] = x[t] - sum_k a_k x[t-k] für t = p .. n-1 (die ersten p Werte entfallen); x darf mehrere Zeilen haben."""
    x = np.asarray(x, dtype=float)
    a = np.asarray(a, dtype=float)
    p = len(a)
    n = x.shape[-1]
    out = x[..., p:].copy()
    for k in range(1, p + 1):
        out = out - a[k - 1] * x[..., p - k:n - k]
    return out


def prewhiten(x, order: int = 2) -> np.ndarray:
    """Mittelwert abziehen, AR-Modell der Ordnung `order` per Burg anpassen und die Residuen liefern (je Zeile)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean(axis=-1, keepdims=True)
    if x.ndim == 1:
        return residuals(x, burg(x, order)[0])
    return np.stack([residuals(row, burg(row, order)[0]) for row in x])

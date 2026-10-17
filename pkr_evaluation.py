"""Analyse, Kennzahlen, Experimente (feste Ströme) und Verdikte der Demo. Reine Funktionen; Zufall nur über SplitMix64."""

import math
from dataclasses import dataclass

import numpy as np

import pkr_constants as C
import pkr_interval as I
import pkr_prewhiten as P
import pkr_tests as T
import signal_generator as G


def de(x: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


def amplitude(snr_db: float) -> float:
    """Amplitude einer Linie mit A^2 / 2 gleich 10^(SNR/10) mal der Varianz 1 des Hintergrunds."""
    return math.sqrt(2.0 * 10.0 ** (snr_db / 10.0))


def background_noise(background: str, runs: int, n: int, rng) -> np.ndarray:
    """runs x n Rauschen mit Varianz 1: weiß oder AR-Hintergrund aus den Polen von BACKGROUND_POLES (Innovationen so skaliert, dass die Varianz 1 ist)."""
    poles = C.BACKGROUND_POLES[background]
    if not poles:
        return G.make_noise_batch(runs, n, (), 0.0, 1.0, rng)
    return G.make_noise_batch(runs, n, poles, G.ar_unit_sigma(G.ar_from_poles(poles)), 0.0, rng)


def background_density(background: str, f) -> np.ndarray:
    """Wahre Dichte des Hintergrunds (Varianz 1, Leistung je Zyklus je Abtastwert) an den Frequenzen f."""
    poles = C.BACKGROUND_POLES[background]
    if not poles:
        return np.ones(len(np.atleast_1d(f)))
    return G.true_psd(f, 1.0, poles, G.ar_unit_sigma(G.ar_from_poles(poles)), 0.0)


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    background: str = C.DEFAULT_BACKGROUND
    line_mode: str = C.DEFAULT_LINE_MODE
    line: float = C.DEFAULT_LINE
    snr: int = C.DEFAULT_SNR
    test: str = C.DEFAULT_TEST
    alpha: float = C.DEFAULT_ALPHA
    band: str = C.DEFAULT_BAND
    seed: int = C.DEFAULT_SEED

    @property
    def has_line(self) -> bool:
        return self.line_mode == "an"


def build_signal(s: Settings) -> np.ndarray:
    """Hintergrund (Varianz 1), danach bei „Linie an“ eine Kosinuslinie mit zufälliger Phase. Verbrauch: Rauschen, dann uniform(1)."""
    rng = G.SplitMix64(s.seed)
    x = background_noise(s.background, 1, s.n, rng)[0]
    if s.has_line:
        ph = 2.0 * math.pi * float(rng.uniform(1)[0])
        x = x + amplitude(s.snr) * np.cos(2.0 * math.pi * s.line * np.arange(s.n) + ph)
    return x


@dataclass(frozen=True)
class TestResult:
    name: str
    freqs: np.ndarray               # Frequenzen der Statistik (Zyklen je Abtastwert)
    ratio: np.ndarray               # Quotient Periodogramm / Hintergrund je Frequenz
    threshold: float                # Schwelle für das Maximum des Quotienten
    stat: float                     # Maximum des Quotienten
    p: float                        # p-Wert des Maximums über alle Frequenzen
    alarm: bool
    peak_hz: float
    right_place: bool | None        # Gipfel nahe der Linie (None ohne Linie)


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    x: np.ndarray
    f: np.ndarray
    e: np.ndarray
    true_density: np.ndarray
    fitted_density: np.ndarray      # angepasstes AR(1)
    local_background: np.ndarray
    band: tuple | None              # (unten, oben) des χ²-Intervalls mit zwei Freiheitsgraden, falls gewünscht
    results: dict
    m: int


def _result(name, freqs, ratio, threshold, p, s: Settings) -> TestResult:
    i = int(np.argmax(ratio))
    stat = float(ratio[i])
    peak = float(freqs[i])
    right = (abs(peak - s.line) < C.FOUND_BINS / s.n) if s.has_line else None
    return TestResult(name, freqs, ratio, threshold, stat, p, stat > threshold, peak, right)


def run_tests(x, alpha: float, s: Settings) -> dict:
    """Alle sieben Tests auf der Reihe x (mit den Einstellungen s für die Zuordnung zur Linie)."""
    n = len(x)
    f = T.inner_frequencies(n)
    e = T.periodogram_inner(x)
    m = len(e)
    white = e / e.mean()
    top = float(white.max())
    out = {}
    out["Einzelfrequenz (weiß)"] = _result("Einzelfrequenz (weiß)", f, white, T.single_threshold(alpha), math.exp(-top), s)
    out["Bonferroni (weiß)"] = _result("Bonferroni (weiß)", f, white, T.bonferroni_threshold(alpha, m), min(1.0, m * math.exp(-top)), s)
    out["Šidák (weiß)"] = _result("Šidák (weiß)", f, white, T.sidak_threshold(alpha, m), T.family_p(math.exp(-top), m), s)
    g = T.fisher_g(e)
    out["Fisher g (weiß)"] = _result("Fisher g (weiß)", f, white, T.fisher_crit(alpha, m) * m, T.fisher_p(float(g), m), s)
    phi, s2 = T.ar1_fit(x)
    fitted = T.ar1_density(phi, s2, n)
    r_fit = e / fitted
    out["AR(1) angepasst + Šidák"] = _result("AR(1) angepasst + Šidák", f, r_fit, T.sidak_threshold(alpha, m), T.family_p(math.exp(-float(r_fit.max())), m), s)
    local = T.local_mean_background(e, C.LOCAL_WIDTH)
    r_loc = e / local
    out["Lokaler Mittelwert + F-Schwelle"] = _result("Lokaler Mittelwert + F-Schwelle", f, r_loc, T.local_threshold(alpha, m, C.LOCAL_WIDTH),
                                                    T.family_p(float(T.local_p(r_loc.max(), C.LOCAL_WIDTH)), m), s)
    res = P.prewhiten(x, C.PREWHITEN_ORDER)
    er = T.periodogram_inner(res)
    mr = len(er)
    fr = np.arange(1, mr + 1) / len(res)
    gr = T.fisher_g(er)
    out["Vorfilterung AR(2) + Fisher g"] = _result("Vorfilterung AR(2) + Fisher g", fr, er / er.mean(), T.fisher_crit(alpha, mr) * mr, T.fisher_p(float(gr), mr), s)
    return out


def analyse(s: Settings) -> Analysis:
    x = build_signal(s)
    n = s.n
    f = T.inner_frequencies(n)
    e = T.periodogram_inner(x)
    phi, s2 = T.ar1_fit(x)
    band = None
    if s.band == "an":
        band = tuple(np.asarray(v) for v in I.chi2_interval(e, 2.0, C.BAND_LEVEL))
    return Analysis(s, x, f, e, background_density(s.background, f), T.ar1_density(phi, s2, n), T.local_mean_background(e, C.LOCAL_WIDTH), band, run_tests(x, s.alpha, s), len(e))


def outcome(r: TestResult, has_line: bool) -> str:
    """Einordnung der Entscheidung eines Tests gegen die Wahrheit."""
    if has_line:
        if r.alarm:
            return "richtig erkannt" if r.right_place else "Alarm an der falschen Stelle"
        return "übersehen"
    return "Falschalarm" if r.alarm else "richtig: kein Befund"


# --- Experimente ---------------------------------------------------------------------------------------------------------------------------


def _line_batch(runs: int, n: int, snr: float, frange, rng):
    """Zufällige Lagen und Phasen (runs) und die Linien runs x n mit der Amplitude zum Rauschabstand snr."""
    lo, hi = frange
    f0 = lo + (hi - lo) * rng.uniform(runs)
    ph = 2.0 * math.pi * rng.uniform(runs)
    t = np.arange(n)
    return f0, amplitude(snr) * np.cos(2.0 * math.pi * f0[:, None] * t[None, :] + ph[:, None])


def sweep_alarm(ns=C.EXP_ALARM_NS, alpha: float = C.EXP_ALARM_ALPHA) -> list[dict]:
    """Falschalarm des höchsten Gipfels unter weißem Rauschen (Quotient I / mittleres I): Einzelfrequenz-Schwelle, Bonferroni, Šidák und Fishers g, je Länge n."""
    rows = []
    for n in ns:
        runs = C.EXP_ALARM_RUNS[n]
        rng = G.SplitMix64(C.EXP_ALARM_SEED + n)
        e = T.periodogram_inner(rng.normal(runs * n).reshape(runs, n))
        m = e.shape[1]
        top = (e / e.mean(axis=1, keepdims=True)).max(axis=1)
        rows.append({"n": n, "m": m, "runs": runs, "single": float(np.mean(top > T.single_threshold(alpha))), "bonferroni": float(np.mean(top > T.bonferroni_threshold(alpha, m))),
                     "sidak": float(np.mean(top > T.sidak_threshold(alpha, m))), "fisher": float(np.mean(T.fisher_g(e) > T.fisher_crit(alpha, m))), "theory": 1.0 - (1.0 - alpha) ** m})
    return rows


def sweep_power(ns=C.EXP_POWER_NS, snrs=C.EXP_POWER_SNR, runs: int = C.EXP_POWER_RUNS, alpha: float = C.EXP_ALARM_ALPHA) -> list[dict]:
    """Erkennungsleistung von Fishers g für eine Linie (zufällige Lage) in weißem Rauschen: Anteil erkannt, und erkannt an der richtigen Stelle (Gipfel innerhalb von 1,5 / n)."""
    rows = []
    for n in ns:
        gc = T.fisher_crit(alpha, (n - 1) // 2)
        for snr in snrs:
            rng = G.SplitMix64(C.EXP_POWER_SEED + n + snr)
            f0, line = _line_batch(runs, n, snr, C.EXP_POWER_F, rng)
            e = T.periodogram_inner(line + rng.normal(runs * n).reshape(runs, n))
            hit = T.fisher_g(e) > gc
            place = np.abs((np.argmax(e, axis=1) + 1) / n - f0) < C.FOUND_BINS / n
            rows.append({"n": n, "snr": snr, "detected": float(hit.mean()), "right_place": float((hit & place).mean()), "runs": runs})
    return rows


def _background_alarms(x, n: int, phi: float, alpha: float) -> dict:
    """Anteile Alarm je Hintergrundmodell auf den Zeilen x (nur Rauschen): weiß (Fisher), bekannter Hintergrund, AR(1) angepasst, lokaler Mittelwert mit F-Schwelle, naiver Median."""
    e = T.periodogram_inner(x)
    m = e.shape[1]
    z = T.sidak_threshold(alpha, m)
    s_true = background_density("weiß" if phi == 0 else f"AR(1) φ = {str(phi).replace('.', ',')}", T.inner_frequencies(n))
    ph, s2 = T.ar1_fit(x)
    fitted = T.ar1_density(ph, s2, n)
    return {"white": float(np.mean(T.fisher_g(e) > T.fisher_crit(alpha, m))), "known": float(np.mean((e / s_true).max(axis=1) > z)), "fitted": float(np.mean((e / fitted).max(axis=1) > z)),
            "local": float(np.mean((e / T.local_mean_background(e, C.LOCAL_WIDTH)).max(axis=1) > T.local_threshold(alpha, m, C.LOCAL_WIDTH))),
            "median": float(np.mean((e / T.median_background(e, C.LOCAL_WIDTH)).max(axis=1) > z))}


def sweep_background(ns=C.EXP_BG_NS, phis=C.EXP_BG_PHIS, runs: int = C.EXP_BG_RUNS, alpha: float = C.EXP_ALARM_ALPHA, power_n: int = C.EXP_BGP_N, power_phi: float = C.EXP_BGP_PHI,
                     power_snrs=C.EXP_BGP_SNR, power_runs: int = C.EXP_BGP_RUNS) -> dict:
    """Gefärbter Hintergrund. `alarm`: Falschalarm (nur Rauschen) je n und φ für fünf Hintergrundmodelle; `power`: Erkennung an der richtigen Stelle einer Linie auf AR(1)-Hintergrund
    für den weißen Test, den angepassten AR(1) und den lokalen Mittelwert."""
    alarm = []
    for n in ns:
        for phi in phis:
            name = "weiß" if phi == 0 else f"AR(1) φ = {str(phi).replace('.', ',')}"
            x = background_noise(name, runs, n, G.SplitMix64(C.EXP_BG_SEED + n + int(100 * phi)))
            alarm.append({"n": n, "phi": phi, "runs": runs, **_background_alarms(x, n, phi, alpha)})
    power = []
    name = f"AR(1) φ = {str(power_phi).replace('.', ',')}"
    m = (power_n - 1) // 2
    z = T.sidak_threshold(alpha, m)
    gc = T.fisher_crit(alpha, m)
    for snr in power_snrs:
        rng = G.SplitMix64(C.EXP_BGP_SEED + snr)
        f0, line = _line_batch(power_runs, power_n, snr, C.EXP_BGP_F, rng)
        x = line + background_noise(name, power_runs, power_n, rng)
        e = T.periodogram_inner(x)
        ph, s2 = T.ar1_fit(x)
        ratios = {"white": e / e.mean(axis=1, keepdims=True), "fitted": e / T.ar1_density(ph, s2, power_n), "local": e / T.local_mean_background(e, C.LOCAL_WIDTH)}
        thr = {"white": gc * m, "fitted": z, "local": T.local_threshold(alpha, m, C.LOCAL_WIDTH)}
        row = {"snr": snr, "runs": power_runs}
        for key, r in ratios.items():
            place = np.abs((np.argmax(r, axis=1) + 1) / power_n - f0) < C.FOUND_BINS / power_n
            row[key] = float(np.mean((r.max(axis=1) > thr[key]) & place))
        power.append(row)
    return {"alarm": alarm, "power": power}


def sweep_interval(n: int = C.EXP_INT_N, seg: int = C.EXP_INT_SEG, rs=C.EXP_INT_RS, runs: int = C.EXP_INT_RUNS, positions=C.EXP_INT_POSITIONS, level: float = C.BAND_LEVEL) -> list[dict]:
    """Überdeckung und Breite der 95-%-Intervalle des Spektrums (AR(2)-Resonanz, Wahrheit = wahre Dichte) für das Periodogramm (χ², ν = 2), Welch (χ², ν nach der Formel) und Welch mit Jackknife."""
    rows = []
    for r in rs:
        poles = ((C.EXP_INT_RESONANCE, r),)
        a = G.ar_from_poles(poles)
        sig = G.ar_unit_sigma(a)
        x = G.make_noise_batch(runs, n, poles, sig, 0.0, G.SplitMix64(C.EXP_INT_SEED + int(100 * r)))
        est, k_seg, segs = I.welch(x, seg, seg // 2)
        nu = I.welch_dof(k_seg)
        e = T.periodogram_inner(x)
        fw, fp = np.arange(seg // 2 + 1) / seg, T.inner_frequencies(n)
        for label, f_val in positions:
            kw, kp = int(np.argmin(np.abs(fw - f_val))), int(np.argmin(np.abs(fp - f_val)))
            for method, lohi, truth_f in (("Periodogramm χ²", I.chi2_interval(e[:, kp], 2.0, level), fp[kp]), ("Welch χ²", I.chi2_interval(est[:, kw], nu, level), fw[kw]),
                                          ("Welch Jackknife", I.jackknife_interval(segs[:, :, kw], level), fw[kw])):
                truth = float(G.true_psd(np.array([truth_f]), 1.0, poles, sig, 0.0)[0])
                lo, hi = lohi
                rows.append({"r": r, "position": label, "method": method, "coverage": float(np.mean((lo <= truth) & (truth <= hi))), "width_db": float(np.mean(I.width_db(lo, hi))), "K": k_seg,
                             "nu": 2.0 if method.startswith("Periodogramm") else nu, "runs": runs})
    return rows


def sweep_prewhiten(n: int = C.EXP_PRE_N, runs: int = C.EXP_PRE_RUNS, alpha: float = C.EXP_ALARM_ALPHA) -> dict:
    """Steiler Hintergrund (nur Rauschen): Falschalarm des weißen Tests auf dem Rohsignal, des Tests gegen den bekannten Hintergrund (Rechteckfenster, Leckage) und des Fisher-Tests auf den
    AR(2)-Residuen."""
    x = background_noise("steil (AR(2))", runs, n, G.SplitMix64(C.EXP_PRE_SEED))
    e = T.periodogram_inner(x)
    m = e.shape[1]
    known = background_density("steil (AR(2))", T.inner_frequencies(n))
    er = T.periodogram_inner(P.prewhiten(x, C.PREWHITEN_ORDER))
    dyn = float(10.0 * np.log10(known.max() / known.min()))
    return {"n": n, "runs": runs, "white": float(np.mean(T.fisher_g(e) > T.fisher_crit(alpha, m))), "known": float(np.mean((e / known).max(axis=1) > T.sidak_threshold(alpha, m))),
            "prewhitened": float(np.mean(T.fisher_g(er) > T.fisher_crit(alpha, er.shape[1]))), "dynamic_db": dyn}


# --- Verdikte (reine Funktionen: (Art, Text)) ----------------------------------------------------------------------------------------------


def verdict_main(a: Analysis) -> tuple[str, str]:
    s = a.settings
    r = a.results[s.test]
    head = f"{s.test}: höchster Gipfel bei {de(r.peak_hz, 3)}, Quotient {de(r.stat, 1)} gegen die Schwelle {de(r.threshold, 1)}, p = {de(r.p, 3)}."
    what = outcome(r, s.has_line)
    if what == "richtig erkannt":
        return "success", head + " Alarm an der Stelle der Linie: richtig erkannt."
    if what == "richtig: kein Befund":
        return "success", head + " Kein Alarm, und es ist auch keine Linie da."
    if what == "Alarm an der falschen Stelle":
        return "warning", head + " Alarm, aber der Gipfel liegt nicht bei der Linie: der Hintergrund oder das Rauschen, nicht die Linie."
    if what == "Falschalarm":
        return "warning", head + " Alarm, obwohl keine Linie da ist: ein Falschalarm."
    return "warning", head + " Kein Alarm, aber eine Linie ist da: übersehen."


def verdict_alarm(rows: list[dict]) -> tuple[str, str]:
    by = {r["n"]: r for r in rows}
    small, mid, big = min(by), sorted(by)[len(by) // 2], max(by)
    return "warning", (f"Unter weißem Rauschen überschreitet der höchste Gipfel die Einzelfrequenz-Schwelle (5 %) bei n = {small} in {de(100 * by[small]['single'], 0)} %, bei n = {mid} in {de(100 * by[mid]['single'], 0)} % "
                       f"und bei n = {big} in {de(100 * by[big]['single'], 0)} % der Läufe. Fishers g trifft die 5 % ({de(100 * by[mid]['fisher'], 1)} % bei n = {mid}), Šidák ist mit geschätztem Mittelwert "
                       f"konservativ ({de(100 * by[mid]['sidak'], 1)} %). **Der höchste Gipfel ist das Maximum über alle Frequenzen.**")


def verdict_power(rows: list[dict]) -> tuple[str, str]:
    def at(n: int, snr: int) -> dict:
        return [r for r in rows if r["n"] == n and r["snr"] == snr][0]
    lo_n, hi_n = min(r["n"] for r in rows), max(r["n"] for r in rows)
    snr = min(r["snr"] for r in rows)
    return "info", (f"Eine Linie {de(abs(snr), 0)} dB unter der Rauschvarianz erkennt Fishers g bei n = {lo_n} in {de(100 * at(lo_n, snr)['detected'], 0)} % der Läufe, bei n = {hi_n} in {de(100 * at(hi_n, snr)['detected'], 0)} %. "
                    "**Mehr Daten heben die Linie aus dem Rauschen, die Schwelle bleibt bei 5 %.**")


def verdict_background(res: dict) -> tuple[str, str]:
    def at(n: int, phi: float) -> dict:
        return [r for r in res["alarm"] if r["n"] == n and r["phi"] == phi][0]
    n = max(r["n"] for r in res["alarm"])
    phi = 0.8
    a = at(n, phi)
    top = max(r["snr"] for r in res["power"])
    low = min(r["snr"] for r in res["power"])
    p_low = [r for r in res["power"] if r["snr"] == low][0]
    p_top = [r for r in res["power"] if r["snr"] == top][0]
    return "warning", (f"Bei AR(1) mit φ = {de(phi, 1)} und n = {n} meldet der weiße Test in {de(100 * a['white'], 0)} % der Läufe Alarm (reines Rauschen), der Test gegen den angepassten AR(1)-Hintergrund in {de(100 * a['fitted'], 1)} %, "
                       f"der lokale Mittelwert mit F-Schwelle in {de(100 * a['local'], 1)} %, der naive Median in {de(100 * a['median'], 0)} %. Eine Linie {de(abs(low), 0)} dB unter der Hintergrundvarianz erkennt der angepasste Test an der richtigen Stelle "
                       f"in {de(100 * p_low['fitted'], 0)} %, der lokale Mittelwert in {de(100 * p_low['local'], 0)} % (bei {de(top, 0)} dB: {de(100 * p_top['local'], 0)} %, die Linie verdeckt sich selbst). "
                       "**Die Schwelle gilt nur gegen den richtigen Hintergrund.**")


def verdict_interval(rows: list[dict]) -> tuple[str, str]:
    def at(r: float, pos: str, method: str) -> dict:
        return [x for x in rows if x["r"] == r and x["position"] == pos and x["method"] == method][0]
    r_hi = max(x["r"] for x in rows)
    return "info", (f"Das 95-%-Intervall des Periodogramms ist {de(at(r_hi, 'Gipfel', 'Periodogramm χ²')['width_db'], 1)} dB breit und deckt die wahre Dichte in {de(100 * at(r_hi, 'Gipfel', 'Periodogramm χ²')['coverage'], 0)} %; "
                    f"Welch (K = {at(r_hi, 'Gipfel', 'Welch χ²')['K']}) ist {de(at(r_hi, 'Gipfel', 'Welch χ²')['width_db'], 1)} dB breit und deckt am scharfen Gipfel (Radius {de(r_hi, 2)}) {de(100 * at(r_hi, 'Gipfel', 'Welch χ²')['coverage'], 0)} %, "
                    f"das Jackknife {de(100 * at(r_hi, 'Gipfel', 'Welch Jackknife')['coverage'], 0)} %. **Schmaler heißt nicht sicherer: Am scharfen Gipfel fehlt der Erwartungswert.**")


def verdict_prewhiten(res: dict) -> tuple[str, str]:
    return "warning", (f"Bei einem Hintergrund mit {de(res['dynamic_db'], 0)} dB Spannweite meldet der weiße Test in {de(100 * res['white'], 0)} % der Läufe Alarm, der Test gegen den bekannten Hintergrund (Rechteckfenster) in "
                       f"{de(100 * res['known'], 0)} %, der Fisher-Test auf den AR(2)-Residuen in {de(100 * res['prewhitened'], 1)} %. **Leckage macht auch den wahren Hintergrund unbrauchbar; Vorfilterung hilft.**")

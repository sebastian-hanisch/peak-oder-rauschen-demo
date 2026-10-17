"""Verdikte: reine Funktionen an handgesetzten Zeilen; Art, Schwellen und jede Zahl im Text."""

import numpy as np

import pkr_evaluation as ev


def make_result(alarm=True, right=True, stat=33.4, thr=8.5, p=0.0004, peak=0.301, name="AR(1) angepasst + Šidák"):
    z = np.zeros(1)
    return ev.TestResult(name, z, z, thr, stat, p, alarm, peak, right)


def make_analysis(result, has_line=True, test="AR(1) angepasst + Šidák"):
    s = ev.Settings(line_mode="an" if has_line else "aus", test=test)
    z = np.zeros(1)
    return ev.Analysis(s, z, z, z, z, z, z, None, {test: result}, 100)


def test_main_verdict_all_five_outcomes():
    kind, text = ev.verdict_main(make_analysis(make_result()))
    assert kind == "success"
    assert text == "AR(1) angepasst + Šidák: höchster Gipfel bei 0,301, Quotient 33,4 gegen die Schwelle 8,5, p = 0,000. Alarm an der Stelle der Linie: richtig erkannt."
    kind, text = ev.verdict_main(make_analysis(make_result(alarm=True, right=False)))
    assert kind == "warning" and text.endswith("Alarm, aber der Gipfel liegt nicht bei der Linie: der Hintergrund oder das Rauschen, nicht die Linie.")
    kind, text = ev.verdict_main(make_analysis(make_result(alarm=False, right=True, stat=5.3, thr=7.6, p=0.455)))
    assert kind == "warning" and text.endswith("Kein Alarm, aber eine Linie ist da: übersehen.") and "p = 0,455" in text
    kind, text = ev.verdict_main(make_analysis(make_result(alarm=True, right=None), has_line=False))
    assert kind == "warning" and text.endswith("Alarm, obwohl keine Linie da ist: ein Falschalarm.")
    kind, text = ev.verdict_main(make_analysis(make_result(alarm=False, right=None, stat=5.3, thr=7.6, p=0.455), has_line=False))
    assert kind == "success" and text.endswith("Kein Alarm, und es ist auch keine Linie da.")


def test_outcome_mapping():
    assert ev.outcome(make_result(True, True), True) == "richtig erkannt" and ev.outcome(make_result(True, False), True) == "Alarm an der falschen Stelle"
    assert ev.outcome(make_result(False, True), True) == "übersehen" and ev.outcome(make_result(True, None), False) == "Falschalarm" and ev.outcome(make_result(False, None), False) == "richtig: kein Befund"


def alarm_rows():
    return [{"n": n, "m": m, "runs": 100, "single": s, "bonferroni": b, "sidak": si, "fisher": f, "theory": 0.9} for n, m, s, b, si, f in ((32, 15, 0.581, 0.02, 0.021, 0.0495), (128, 63, 0.989, 0.037, 0.038, 0.0512),
                                                                                                               (512, 255, 1.0, 0.042, 0.044, 0.0488))]


def test_verdict_alarm_text():
    kind, text = ev.verdict_alarm(alarm_rows())
    assert kind == "warning"
    assert text == ("Unter weißem Rauschen überschreitet der höchste Gipfel die Einzelfrequenz-Schwelle (5 %) bei n = 32 in 58 %, bei n = 128 in 99 % und bei n = 512 in 100 % der Läufe. "
                    "Fishers g trifft die 5 % (5,1 % bei n = 128), Šidák ist mit geschätztem Mittelwert konservativ (3,8 %). **Der höchste Gipfel ist das Maximum über alle Frequenzen.**")


def test_verdict_power_text():
    rows = [{"n": n, "snr": s, "detected": d, "right_place": d, "runs": 10} for n, s, d in ((128, -9, 0.427), (128, 0, 1.0), (512, -9, 0.9935), (512, 0, 1.0))]
    kind, text = ev.verdict_power(rows)
    assert kind == "info"
    assert text == ("Eine Linie 9 dB unter der Rauschvarianz erkennt Fishers g bei n = 128 in 43 % der Läufe, bei n = 512 in 99 %. **Mehr Daten heben die Linie aus dem Rauschen, die Schwelle bleibt bei 5 %.**")


def background_result():
    alarm = [{"n": n, "phi": phi, "runs": 100, "white": 1.0, "known": 0.055, "fitted": 0.043, "local": 0.053, "median": 0.478} for n in (128, 512) for phi in (0.5, 0.8)]
    power = [{"snr": snr, "runs": 10, "white": w, "fitted": f, "local": lo} for snr, w, f, lo in ((-18, 0.0, 0.803, 0.422), (-6, 0.925, 1.0, 0.786))]
    return {"alarm": alarm, "power": power}


def test_verdict_background_text():
    kind, text = ev.verdict_background(background_result())
    assert kind == "warning"
    assert text == ("Bei AR(1) mit φ = 0,8 und n = 512 meldet der weiße Test in 100 % der Läufe Alarm (reines Rauschen), der Test gegen den angepassten AR(1)-Hintergrund in 4,3 %, "
                    "der lokale Mittelwert mit F-Schwelle in 5,3 %, der naive Median in 48 %. Eine Linie 18 dB unter der Hintergrundvarianz erkennt der angepasste Test an der richtigen Stelle in 80 %, "
                    "der lokale Mittelwert in 42 % (bei -6 dB: 79 %, die Linie verdeckt sich selbst). **Die Schwelle gilt nur gegen den richtigen Hintergrund.**")


def test_verdict_interval_text():
    def r(method, cov, width):
        return {"r": 0.98, "position": "Gipfel", "method": method, "coverage": cov, "width_db": width, "K": 7, "nu": 13.4, "runs": 10}
    rows = [r("Periodogramm χ²", 0.9535, 21.63), r("Welch χ²", 0.9085, 6.83), r("Welch Jackknife", 0.8842, 7.65)]
    kind, text = ev.verdict_interval(rows)
    assert kind == "info"
    assert text == ("Das 95-%-Intervall des Periodogramms ist 21,6 dB breit und deckt die wahre Dichte in 95 %; Welch (K = 7) ist 6,8 dB breit und deckt am scharfen Gipfel (Radius 0,98) 91 %, "
                    "das Jackknife 88 %. **Schmaler heißt nicht sicherer: Am scharfen Gipfel fehlt der Erwartungswert.**")


def test_verdict_prewhiten_text():
    kind, text = ev.verdict_prewhiten({"n": 256, "runs": 10, "white": 1.0, "known": 0.8885, "prewhitened": 0.05, "dynamic_db": 63.7})
    assert kind == "warning"
    assert text == ("Bei einem Hintergrund mit 64 dB Spannweite meldet der weiße Test in 100 % der Läufe Alarm, der Test gegen den bekannten Hintergrund (Rechteckfenster) in 89 %, "
                    "der Fisher-Test auf den AR(2)-Residuen in 5,0 %. **Leckage macht auch den wahren Hintergrund unbrauchbar; Vorfilterung hilft.**")


def test_decimal_comma_and_amplitude():
    assert ev.de(1.26, 1) == "1,3" and ev.de(3.0, 0) == "3" and ev.de(0.04, 3) == "0,040"
    assert ev.amplitude(0) == np.sqrt(2.0) and ev.amplitude(-20) == np.sqrt(0.02)

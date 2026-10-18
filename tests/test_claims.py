"""Jede Zahl und Aussage in den Texten der App (Presets, Einleitung, Experiment-Überschriften) und in der README ist hier an Rechnung oder Messung gebunden."""

from pathlib import Path

import numpy as np
import pytest

import pkr_constants as C
import pkr_evaluation as ev
from sweeps_cache import get

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")


def preset(name, **over):
    return ev.Settings(**{**C.PRESETS[name], **over})


def A(name, **over):
    return ev.analyse(preset(name, **over))


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}".replace(".", ",")


def row(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())][0]


# --- Presets ----------------------------------------------------------------------------------------------------------------------------------


def test_preset_pure_noise_the_single_frequency_threshold_alarms_and_fisher_does_not():
    a = A("Reines Rauschen: der höchste Gipfel täuscht")
    assert a.settings.test == "Einzelfrequenz (weiß)" and a.results[a.settings.test].alarm and ev.outcome(a.results[a.settings.test], False) == "Falschalarm"
    assert not any(a.results[t].alarm for t in C.TESTS[1:])                                  # alle Tests für das Maximum schweigen


def test_preset_same_series_with_the_fisher_threshold_gives_no_alarm():
    a, b = A("Dieselbe Reihe mit Fisher-Schwelle: kein Alarm"), A("Reines Rauschen: der höchste Gipfel täuscht")
    assert np.array_equal(a.x, b.x) and not a.results["Fisher g (weiß)"].alarm and ev.outcome(a.results["Fisher g (weiß)"], False) == "richtig: kein Befund"


def test_preset_red_noise_the_white_test_alarms_and_the_tests_against_the_background_do_not():
    a = A("Rotes Rauschen: der weiße Test meldet fast immer")
    assert a.settings.background == "AR(1) φ = 0,8" and not a.settings.has_line
    assert a.results["Fisher g (weiß)"].alarm and ev.outcome(a.results["Fisher g (weiß)"], False) == "Falschalarm"
    assert not a.results["AR(1) angepasst + Šidák"].alarm and not a.results["Lokaler Mittelwert + F-Schwelle"].alarm


def test_preset_weak_line_on_red_background_is_found_in_the_right_place_only_against_the_background():
    a = A("Schwache Linie auf rotem Hintergrund")
    assert a.settings.snr == -12 and a.settings.has_line
    assert ev.outcome(a.results["AR(1) angepasst + Šidák"], True) == "richtig erkannt"
    white = A("Schwache Linie auf rotem Hintergrund", test="Fisher g (weiß)")
    assert white.results["Fisher g (weiß)"].alarm and not white.results["Fisher g (weiß)"].right_place and ev.outcome(white.results["Fisher g (weiß)"], True) == "Alarm an der falschen Stelle"


def test_preset_steep_background_the_prewhitening_holds_and_the_white_test_does_not():
    a = A("Steiler Hintergrund: Vorfilterung")
    assert not a.results["Vorfilterung AR(2) + Fisher g"].alarm and a.results["Fisher g (weiß)"].alarm
    dyn = 10 * np.log10(a.true_density.max() / a.true_density.min())
    assert round(dyn) == 64 and "64 dB" in C.PRESET_HELP["Steiler Hintergrund: Vorfilterung"] and "64 dB" in README


# --- README: Falschalarm und Erkennung --------------------------------------------------------------------------------------------------------


def test_readme_alarm_numbers():
    rows = {r["n"]: r for r in get("alarm")}
    assert [pct(rows[n]["single"]) for n in (32, 128)] == ["58,1", "98,9"] and rows[512]["single"] == 1.0
    assert [pct(rows[n]["fisher"]) for n in (32, 128, 512)] == ["5,0", "5,1", "4,9"]
    assert [pct(rows[n]["bonferroni"]) for n in (32, 128, 512)] == ["2,0", "3,7", "4,2"] and [pct(rows[n]["sidak"]) for n in (32, 128, 512)] == ["2,1", "3,8", "4,4"]
    for s in ("58,1 %", "98,9 %", "100 % (n = 512)", "5,0 / 5,1 / 4,9 %", "2,0 / 3,7 / 4,2 %", "2,1 / 3,8 / 4,4 %"):
        assert s in README
    assert all(rows[n]["sidak"] < 0.05 for n in (32, 128, 512)) and all(rows[n]["bonferroni"] <= rows[n]["sidak"] for n in rows)


def test_readme_power_numbers():
    p = {(r["n"], r["snr"]): r["right_place"] for r in get("power")}
    assert [pct(p[(128, s)], 0) for s in (-9, -6, -3, 0)] == ["41", "84", "99", "100"] and pct(p[(512, -9)], 0) == "99"
    for s in ("41 % (−9 dB), 84 % (−6 dB), 99 % (−3 dB), 100 % (0 dB)", "99 % bei −9 dB"):
        assert s in README


# --- README: gefärbter Hintergrund ----------------------------------------------------------------------------------------------------------


def test_readme_background_numbers():
    rows = {(r["n"], r["phi"]): r for r in get("background")["alarm"]}
    def seq(n, key, digits=1):
        return " / ".join(pct(rows[(n, phi)][key], digits) for phi in (0.5, 0.8, 0.95))
    assert seq(128, "white") == "61,1 / 99,4 / 100,0" and seq(512, "white") == "87,7 / 100,0 / 100,0"
    assert seq(128, "fitted") == "2,7 / 3,5 / 3,2" and seq(512, "fitted") == "3,9 / 4,3 / 4,7"
    assert seq(128, "local") == "5,0 / 5,8 / 21,9" and seq(512, "local") == "5,5 / 5,3 / 6,2" and seq(128, "median") == "28,7 / 32,2 / 57,0"
    for s in ("61,1 / 99,4 / 100 %", "87,7 / 100 / 100 %", "2,7 / 3,5 / 3,2 %", "3,9 / 4,3 / 4,7 %", "5,0 / 5,8 / 21,9 %", "5,5 / 5,3 / 6,2 %", "28,7 / 32,2 / 57,0 %", "(21,9 %)"):
        assert s in README
    assert C.EXP_BG_RUNS == 4000


def test_readme_background_power_numbers():
    p = {r["snr"]: r for r in get("background")["power"]}
    assert (pct(p[-18]["fitted"], 0), pct(p[-15]["fitted"], 0), pct(p[-12]["fitted"], 0), pct(p[-6]["fitted"], 0)) == ("80", "98", "100", "100")
    assert (pct(p[-18]["local"], 0), pct(p[-6]["local"], 0), pct(p[-9]["white"], 0), pct(p[-18]["white"], 0)) == ("42", "79", "53", "0") and max(r["local"] for r in p.values()) == p[-6]["local"]
    for s in ("80 % (−18 dB), 98 % (−15 dB), 100 % ab −12 dB", "42 % (−18 dB), bis −6 dB höchstens 79 %", "0 % (−18 dB), 53 % (−9 dB)"):
        assert s in README
    assert C.EXP_BGP_N == 512 and C.EXP_BGP_PHI == 0.8


# --- README: Intervalle und Vorfilterung ------------------------------------------------------------------------------------------------------


def test_readme_interval_numbers():
    rows = get("interval")
    def cov(method, r):
        return [x["coverage"] for x in rows if x["method"] == method and x["r"] == r]
    assert (pct(min(cov("Periodogramm χ²", 0.98) + cov("Periodogramm χ²", 0.9))), pct(max(cov("Periodogramm χ²", 0.98) + cov("Periodogramm χ²", 0.9)))) == ("94,0", "95,3")
    assert (pct(min(cov("Welch χ²", 0.9))), pct(max(cov("Welch χ²", 0.9)))) == ("94,7", "95,3")
    assert pct(row(rows, r=0.98, position="Gipfel", method="Welch χ²")["coverage"]) == "90,8" and pct(row(rows, r=0.98, position="Gipfel", method="Welch Jackknife")["coverage"]) == "88,4"
    assert (pct(min(cov("Welch Jackknife", 0.9))), pct(max(cov("Welch Jackknife", 0.9)))) == ("92,7", "94,3")
    w = {m: row(rows, r=0.9, position="Gipfel", method=m) for m in ("Periodogramm χ²", "Welch χ²", "Welch Jackknife")}
    assert [round(w[m]["width_db"], 1) for m in w] == [21.6, 6.8, 8.0] and round(w["Welch χ²"]["nu"], 1) == 13.4 and w["Welch χ²"]["K"] == 7
    for s in ("21,6 dB", "94,0 bis 95,3 %", "6,8 dB", "94,7 bis 95,3 %", "90,8 %", "8,0 dB", "92,7 bis 94,3 %", "88,4 %", "ν = 13,4", "K = 7"):
        assert s in README


def test_readme_prewhitening_numbers():
    res = get("prewhiten")
    assert (pct(res["white"]), pct(res["known"]), pct(res["prewhitened"])) == ("100,0", "88,8", "5,0") and round(res["dynamic_db"]) == 64
    for s in ("in 100 %", "88,8 %", "in 5,0 %", "64 dB Spannweite"):
        assert s in README


def test_readme_method_remarks():
    assert "höchstens 1,2 Prozentpunkte" in README and np.sqrt(0.25 / 2000) * 100 <= 1.2 and "Seeds ab 810000" in README and C.EXP_ALARM_SEED == 810000
    assert C.EXP_POWER_RUNS == 2000 and max(C.EXP_ALARM_RUNS.values()) == 20000


def test_the_app_text_the_periodogram_interval_is_over_twenty_decibels_wide():
    assert get("interval")[0]["width_db"] > 20.0

"""Randfälle: kleinste und größte Länge, alle Hintergründe und Tests, Linien an den Reglergrenzen, Abbildungen und PDF."""

import dataclasses

import numpy as np
import pytest

import pkr_constants as C
import pkr_evaluation as ev
import pkr_visualization as V
from pkr_pdf_export import build_pdf


@pytest.mark.parametrize("background", list(C.BACKGROUNDS))
def test_every_background_runs_for_the_shortest_and_the_longest_record(background):
    for n in (min(C.N_CHOICES), max(C.N_CHOICES)):
        a = ev.analyse(ev.Settings(n=n, background=background, band="an"))
        assert np.isfinite(a.e).all() and (a.e >= 0).all() and all(np.isfinite(r.stat) and 0.0 <= r.p <= 1.0 for r in a.results.values())


@pytest.mark.parametrize("test", list(C.TESTS))
def test_every_test_runs_with_and_without_a_line(test):
    for line_mode in C.LINE_MODES:
        a = ev.analyse(ev.Settings(test=test, line_mode=line_mode))
        r = a.results[test]
        assert r.alarm == (r.stat > r.threshold) and ev.outcome(r, line_mode == "an")


@pytest.mark.parametrize("line", [C.LINE_MIN, C.LINE_MAX])
@pytest.mark.parametrize("snr", [C.SNR_MIN, C.SNR_MAX])
def test_lines_at_the_slider_limits_run(line, snr):
    for n in (64, 1024):
        a = ev.analyse(ev.Settings(n=n, line_mode="an", line=line, snr=snr))
        assert np.isfinite(a.e).all()
    strong = ev.analyse(ev.Settings(n=1024, line_mode="an", line=line, snr=C.SNR_MAX, background="weiß"))
    assert strong.results["Fisher g (weiß)"].alarm and strong.results["Fisher g (weiß)"].right_place


@pytest.mark.parametrize("alpha", [C.ALPHA_MIN, C.ALPHA_MAX])
def test_extreme_levels_and_seed_limits_run(alpha):
    for seed in (0, C.SEED_MAX):
        a = ev.analyse(ev.Settings(alpha=alpha, seed=seed))
        assert all(np.isfinite(r.threshold) for r in a.results.values())


def test_pdf_builds_for_both_line_modes():
    for line_mode in C.LINE_MODES:
        pdf = build_pdf(ev.analyse(ev.Settings(line_mode=line_mode)))
        assert pdf[:4] == b"%PDF" and len(pdf) > 1000


def test_every_figure_builds():
    a = ev.analyse(ev.Settings(band="an", line_mode="an"))
    plain = ev.analyse(ev.Settings())
    assert len(V.build_spectrum(a).data) == 6 and len(V.build_spectrum(plain).data) == 4
    assert len(V.build_spectrum(a).layout.shapes) == 1 and len(V.build_spectrum(plain).layout.shapes) == 0
    for test in C.TESTS:
        fig = V.build_test(dataclasses.replace(a, settings=dataclasses.replace(a.settings, test=test)))
        assert len(fig.data) == 2 and len(fig.layout.shapes) == 2
    assert len(V.build_test(plain).layout.shapes) == 1
    alarm = [{"n": n, "m": 1, "runs": 1, "single": 1.0, "bonferroni": 0.02, "sidak": 0.02, "fisher": 0.05, "theory": 1.0} for n in (32, 128)]
    assert len(V.build_alarm(alarm).data) == 4
    power = [{"n": n, "snr": s, "detected": 1.0, "right_place": 1.0, "runs": 1} for n in (128, 512) for s in (-9, 0)]
    assert len(V.build_power(power).data) == 2
    bg = {"alarm": [{"n": 128, "phi": phi, "runs": 1, "white": 1.0, "known": 0.05, "fitted": 0.03, "local": 0.05, "median": 0.3} for phi in (0.0, 0.5)],
          "power": [{"snr": s, "runs": 1, "white": 0.5, "fitted": 1.0, "local": 0.8} for s in (-18, -6)]}
    assert len(V.build_background_alarm(bg["alarm"]).data) == 5 and len(V.build_background_power(bg["power"]).data) == 3
    iv = [{"r": r, "position": p, "method": m, "coverage": 0.95, "width_db": 7.0, "K": 7, "nu": 13.4, "runs": 1} for r in (0.9, 0.98) for p in ("Gipfel", "Tal") for m in ("Periodogramm χ²", "Welch χ²")]
    assert len(V.build_interval(iv).data) == 2
    assert len(V.build_prewhiten({"white": 1.0, "known": 0.88, "prewhitened": 0.05}).data) == 1

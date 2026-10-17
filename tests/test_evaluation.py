"""Einstellungen, Signalaufbau und Analyse: Datenstrom, Hintergrund, Schwellen der sieben Tests, Einordnung, Experimente mit kleinen Parametern."""

import math

import numpy as np
import pytest

import pkr_constants as C
import pkr_evaluation as ev
import pkr_tests as T
import signal_generator as G


def test_background_noise_has_unit_variance_and_the_documented_stream():
    for bg in C.BACKGROUNDS:
        x = ev.background_noise(bg, 400, 256, G.SplitMix64(1))
        assert x.shape == (400, 256) and x.var() == pytest.approx(1.0, rel=0.12), bg
    assert np.array_equal(ev.background_noise("weiß", 1, 50, G.SplitMix64(3)), G.make_noise_batch(1, 50, (), 0.0, 1.0, G.SplitMix64(3)))


def test_background_density_integrates_to_the_variance_and_is_flat_for_white_noise():
    assert ev.background_density("weiß", np.array([0.1, 0.2, 0.3])).tolist() == [1.0, 1.0, 1.0]
    f = np.arange(1, 2001) / 4000
    for bg in C.BACKGROUNDS[1:]:
        assert 2.0 * ev.background_density(bg, f).mean() * 0.5 == pytest.approx(1.0, rel=0.05), bg      # Fläche über (0, 1/2) ist 1/2, zweiseitig 1 insgesamt
    d = ev.background_density("AR(1) φ = 0,8", np.array([1e-6, 0.5]))
    assert d[0] == pytest.approx((1 - 0.64) / (1 - 0.8) ** 2, rel=1e-4) and d[1] == pytest.approx((1 - 0.64) / (1 + 0.8) ** 2, rel=1e-4)


def test_build_signal_stream_noise_first_then_the_phase():
    s = ev.Settings(n=64, background="weiß", line_mode="an", line=0.2, snr=0, seed=5)
    rng = G.SplitMix64(5)
    noise = ev.background_noise("weiß", 1, 64, rng)[0]
    ph = 2 * math.pi * float(rng.uniform(1)[0])
    assert ev.build_signal(s) == pytest.approx(noise + math.sqrt(2.0) * np.cos(2 * math.pi * 0.2 * np.arange(64) + ph), abs=1e-12)
    assert ev.build_signal(ev.Settings(n=64, line_mode="aus", seed=5)) == pytest.approx(noise, abs=1e-12)
    assert ev.Settings(line_mode="an").has_line and not ev.Settings().has_line


def test_the_seven_tests_have_the_documented_thresholds_and_p_values():
    s = ev.Settings(n=128, seed=3)
    x = ev.build_signal(s)
    r = ev.run_tests(x, 0.05, s)
    m = 63
    assert list(r) == list(C.TESTS) and r["Einzelfrequenz (weiß)"].threshold == pytest.approx(T.single_threshold(0.05)) and r["Bonferroni (weiß)"].threshold == pytest.approx(T.bonferroni_threshold(0.05, m))
    assert r["Šidák (weiß)"].threshold == pytest.approx(T.sidak_threshold(0.05, m)) and r["Fisher g (weiß)"].threshold == pytest.approx(T.fisher_crit(0.05, m) * m)
    assert r["Lokaler Mittelwert + F-Schwelle"].threshold == pytest.approx(T.local_threshold(0.05, m, 11)) and r["AR(1) angepasst + Šidák"].threshold == pytest.approx(T.sidak_threshold(0.05, m))
    e = T.periodogram_inner(x)
    top = (e / e.mean()).max()
    assert r["Einzelfrequenz (weiß)"].stat == pytest.approx(top) and r["Einzelfrequenz (weiß)"].p == pytest.approx(math.exp(-top))
    assert r["Bonferroni (weiß)"].p == pytest.approx(min(1.0, m * math.exp(-top))) and r["Šidák (weiß)"].p == pytest.approx(1 - (1 - math.exp(-top)) ** m)
    assert r["Fisher g (weiß)"].p == pytest.approx(T.fisher_p(float(T.fisher_g(e)), m)) and r["Fisher g (weiß)"].stat == pytest.approx(top)
    for t in r.values():
        assert t.alarm == (t.stat > t.threshold) and 0.0 <= t.p <= 1.0 and t.freqs[int(np.argmax(t.ratio))] == pytest.approx(t.peak_hz)
    assert len(r["Vorfilterung AR(2) + Fisher g"].freqs) == (126 - 1) // 2 and r["Fisher g (weiß)"].freqs.shape == (63,)


def test_the_selected_test_ordering_for_a_strong_line():
    s = ev.Settings(n=256, line_mode="an", line=0.2, snr=0, seed=2)
    r = ev.run_tests(ev.build_signal(s), 0.05, s)
    assert all(t.alarm and t.right_place for t in r.values())
    quiet = ev.Settings(n=256, line_mode="aus", seed=2)
    assert all(t.right_place is None for t in ev.run_tests(ev.build_signal(quiet), 0.05, quiet).values())


def test_right_place_uses_one_and_a_half_bins():
    s = ev.Settings(n=100, line_mode="an", line=0.2049)
    freqs = np.arange(1, 50) / 100

    def place(peak_index):
        ratio = np.zeros(49)
        ratio[peak_index - 1] = 9.0
        return ev._result("t", freqs, ratio, 5.0, 0.0, s)

    assert place(21).right_place is True and place(19).right_place is True            # 0,51 und 1,49 Bins neben der Linie
    assert place(22).right_place is False and place(23).right_place is False          # 1,51 und 2,51 Bins
    assert place(21).alarm and not ev._result("t", freqs, np.full(49, 1.0), 5.0, 0.0, s).alarm


def test_analysis_shapes_band_and_densities():
    a = ev.analyse(ev.Settings(band="an", background="AR(1) φ = 0,8"))
    assert len(a.f) == len(a.e) == a.m == 127 and a.band is not None and np.all(a.band[0] < a.e) and np.all(a.e < a.band[1])
    assert a.band[1] / a.band[0] == pytest.approx(7.377758908 / 0.050635616, rel=1e-6)
    assert ev.analyse(ev.Settings()).band is None and a.true_density == pytest.approx(ev.background_density("AR(1) φ = 0,8", a.f))
    assert a.fitted_density.shape == a.e.shape and a.local_background.shape == a.e.shape and set(a.results) == set(C.TESTS)


def test_the_analysis_is_deterministic_and_the_seed_changes_the_data():
    a, b = ev.analyse(ev.Settings()), ev.analyse(ev.Settings())
    assert np.array_equal(a.x, b.x) and a.results["Fisher g (weiß)"].p == b.results["Fisher g (weiß)"].p
    assert not np.array_equal(a.x, ev.analyse(ev.Settings(seed=40)).x)


def test_alpha_moves_the_thresholds_the_right_way():
    lo, hi = ev.analyse(ev.Settings(alpha=0.01)), ev.analyse(ev.Settings(alpha=0.10))
    for t in C.TESTS:
        assert lo.results[t].threshold > hi.results[t].threshold and lo.results[t].p == pytest.approx(hi.results[t].p)


def test_sweeps_accept_small_parameters():
    a = ev.sweep_alarm(ns=(32,))
    assert len(a) == 1 and a[0]["m"] == 15 and a[0]["theory"] == pytest.approx(1 - 0.95 ** 15)
    assert len(ev.sweep_power(ns=(64,), snrs=(0,), runs=20)) == 1
    bg = ev.sweep_background(ns=(64,), phis=(0.0, 0.8), runs=30, power_n=64, power_snrs=(-6,), power_runs=20)
    assert len(bg["alarm"]) == 2 and len(bg["power"]) == 1
    iv = ev.sweep_interval(n=256, seg=64, rs=(0.9,), runs=40)
    assert len(iv) == 9 and all(0.0 <= r["coverage"] <= 1.0 for r in iv)
    pre = ev.sweep_prewhiten(n=128, runs=30)
    assert set(pre) == {"n", "runs", "white", "known", "prewhitened", "dynamic_db"}

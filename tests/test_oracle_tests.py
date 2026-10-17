"""Kern der Tests: Mini-Instanz von Hand, Fishers g gegen exakte Bruchrechnung und Simulation, Schwellen, Hintergrundmodelle gegen Handrechnung und scipy."""

import math
from fractions import Fraction

import numpy as np
import pytest
from scipy import stats

import pkr_tests as T
import signal_generator as G


def test_the_mini_instance_by_hand(mini_x):
    e = T.periodogram_inner(mini_x)
    assert e == pytest.approx([0.5, 1.0 / 6.0], abs=1e-12) and T.inner_frequencies(6).tolist() == pytest.approx([1 / 6, 2 / 6])
    g = T.fisher_g(e)
    assert g == pytest.approx(0.75, abs=1e-12) and T.fisher_p(float(g), 2) == pytest.approx(0.5, abs=1e-12)
    assert T.fisher_crit(0.5, 2) == pytest.approx(0.75, abs=1e-9)


def test_periodogram_equals_the_double_sum_and_ignores_the_mean():
    x = np.random.default_rng(1).normal(size=11)
    e = T.periodogram_inner(x + 5.0)
    xm = x - x.mean()
    ref = [abs(sum(xm[t] * np.exp(-2j * np.pi * k * t / 11) for t in range(11))) ** 2 / 11 for k in range(1, 6)]
    assert len(e) == 5 and e == pytest.approx(ref, rel=1e-10)
    assert len(T.periodogram_inner(np.ones(12) * 2 + np.arange(12) % 3)) == 5                       # gerade n: Nyquist ausgeschlossen
    assert T.periodogram_inner(np.random.default_rng(2).normal(size=(3, 20))).shape == (3, 9)
    with pytest.raises(ValueError):
        T.periodogram_inner(np.ones(3))


def test_inner_frequencies_and_fisher_g_along_the_last_axis():
    assert T.inner_frequencies(8).tolist() == pytest.approx([0.125, 0.25, 0.375]) and len(T.inner_frequencies(9)) == 4
    e = np.array([[1.0, 3.0], [2.0, 2.0]])
    assert T.fisher_g(e).tolist() == pytest.approx([0.75, 0.5])


def _exact_p(g, m):
    x = Fraction(g)
    return float(sum((-1) ** (j - 1) * math.comb(m, j) * (1 - j * x) ** (m - 1) for j in range(1, int(math.floor(1 / g)) + 1) if 1 - j * x > 0))


@pytest.mark.parametrize("m,g", [(3, 0.4), (7, 0.5612), (15, 0.3346), (63, 0.1086), (255, 0.033), (1023, 0.0096), (1023, 0.009), (2047, 0.005)])
def test_fisher_p_equals_the_exact_rational_sum(m, g):
    assert T.fisher_p(g, m) == pytest.approx(_exact_p(g, m), abs=1e-11)


def test_fisher_p_closed_forms_and_limits():
    for g in (0.55, 0.7, 0.99):
        assert T.fisher_p(g, 2) == pytest.approx(2 * (1 - g))                                 # m = 2: gleichverteilt auf [1/2, 1]
    assert T.fisher_p(0.4, 3) == pytest.approx(3 * 0.6 ** 2 - 3 * 0.2 ** 2, abs=1e-12)        # m = 3, 1/3 <= g < 1/2: C(3,1)(1-g)^2 - C(3,2)(1-2g)^2
    assert T.fisher_p(1.0, 5) == 0.0 and T.fisher_p(1.5, 5) == 0.0 and T.fisher_p(0.2, 5) == 1.0 and T.fisher_p(0.1, 5) == 1.0
    assert T.fisher_p(1.0 / 7.0, 7) == 1.0
    assert all(0.0 <= T.fisher_p(g, 20) <= 1.0 for g in np.linspace(0.05, 0.9, 30))
    ps = [T.fisher_p(g, 20) for g in np.linspace(0.06, 0.9, 30)]
    assert all(a >= b - 1e-12 for a, b in zip(ps, ps[1:]))                                     # monoton fallend
    with pytest.raises(ValueError):
        T.fisher_p(0.5, 0)


def test_fisher_crit_inverts_the_p_value_and_matches_simulation():
    for m in (2, 7, 63, 255):
        for alpha in (0.1, 0.05, 0.01):
            assert T.fisher_p(T.fisher_crit(alpha, m), m) == pytest.approx(alpha, abs=1e-9)
    assert T.fisher_crit(0.05, 1) == 1.0 and T.fisher_crit(0.05, 7) == pytest.approx(0.56115, abs=2e-5)
    e = G.SplitMix64(3).normal(30000 * 33).reshape(30000, 33)
    g = T.fisher_g(T.periodogram_inner(e))
    assert np.mean(g > T.fisher_crit(0.05, 16)) == pytest.approx(0.05, abs=0.005)


def test_thresholds_by_hand():
    assert T.single_threshold(0.05) == pytest.approx(math.log(20.0)) and T.single_threshold(math.exp(-3)) == pytest.approx(3.0)
    assert T.bonferroni_threshold(0.05, 10) == pytest.approx(math.log(200.0))
    assert T.sidak_per_frequency(0.05, 1) == pytest.approx(0.05) and T.sidak_per_frequency(0.05, 2) == pytest.approx(1 - math.sqrt(0.95))
    assert T.sidak_threshold(0.05, 2) == pytest.approx(-math.log(1 - math.sqrt(0.95))) and T.sidak_threshold(0.05, 63) == pytest.approx(7.1, abs=0.02)
    assert T.sidak_threshold(1e-12, 3) == pytest.approx(math.log(3e12), rel=1e-9)                # kleines Niveau numerisch stabil
    assert T.single_threshold(0.05) < T.sidak_threshold(0.05, 10) < T.bonferroni_threshold(0.05, 10) < T.bonferroni_threshold(0.05, 20)
    assert T.family_p(0.05, 1) == pytest.approx(0.05) and T.family_p(0.5, 2) == pytest.approx(0.75) and T.family_p(1.0, 5) == pytest.approx(1.0)
    assert T.family_p(math.exp(-T.sidak_threshold(0.05, 40)), 40) == pytest.approx(0.05, abs=1e-12)


def test_ar1_fit_by_hand_and_the_density_equals_the_generator():
    x = np.array([1.0, 2.0, 1.0, 0.0, -1.0, -2.0])
    xc = x - x.mean()
    r0, r1 = np.mean(xc ** 2), np.sum(xc[1:] * xc[:-1]) / 6
    phi, s2 = T.ar1_fit(x)
    assert phi == pytest.approx(r1 / r0, abs=1e-12) and s2 == pytest.approx(r0 * (1 - (r1 / r0) ** 2), abs=1e-12)
    assert T.ar1_fit(np.array([1.0, -1.0] * 5))[0] == pytest.approx(-0.9, abs=1e-12)          # r1 = -9/10 r0: mit Teilung durch n
    assert T.ar1_fit(np.tile([1.0, -1.0, 1.0, 1.0], 3).astype(float))[0] <= 0.999
    n = 64
    poles = ((0.0, 0.8),)
    a = G.ar_from_poles(poles)
    sig = G.ar_unit_sigma(a)
    d = T.ar1_density(0.8, sig ** 2, n)
    assert d == pytest.approx(G.true_psd(T.inner_frequencies(n), 1.0, poles, sig, 0.0), rel=1e-10)
    assert T.ar1_density(np.array([0.5, 0.0]), np.array([1.0, 2.0]), 8).shape == (2, 3) and T.ar1_density(0.0, 2.0, 8) == pytest.approx([2.0] * 3)
    xs = G.make_noise_batch(400, 512, poles, sig, 0.0, G.SplitMix64(4))
    phis, s2s = T.ar1_fit(xs)
    assert phis.mean() == pytest.approx(0.8, abs=0.02) and (s2s.mean()) == pytest.approx(sig ** 2, rel=0.05)


def test_the_ar1_fit_is_clipped_to_the_stationary_range():
    assert T.ar1_fit(np.arange(10.0))[0] <= 0.999 and T.ar1_fit(np.array([5.0, 5.0, 5.0, 6.0]))[0] >= -0.999


def test_local_mean_background_by_hand():
    e = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    lm = T.local_mean_background(e, 3)                                                           # Nachbarn links und rechts, Ränder gespiegelt
    assert lm == pytest.approx([(2 + 2) / 2, (1 + 3) / 2, (2 + 4) / 2, (3 + 5) / 2, (4 + 6) / 2, (5 + 7) / 2, (6 + 6) / 2])
    lm5 = T.local_mean_background(e, 5)
    assert lm5[3] == pytest.approx((2 + 3 + 5 + 6) / 4)
    assert T.local_mean_background(np.ones((2, 20)), 11).shape == (2, 20)
    with pytest.raises(ValueError):
        T.local_mean_background(np.ones(5), 11)


def test_local_ratio_distribution_equals_the_f_distribution():
    for r in (0.5, 3.0, 12.0):
        assert T.local_p(r, 11) == pytest.approx(stats.f.sf(r, 2, 20), rel=1e-9)
    for m in (15, 255):
        assert T.local_threshold(0.05, m, 11) == pytest.approx(stats.f.ppf(1 - T.sidak_per_frequency(0.05, m), 2, 20), rel=1e-9)
    assert T.local_threshold(0.05, 63, 5) == pytest.approx(stats.f.ppf(1 - T.sidak_per_frequency(0.05, 63), 2, 8), rel=1e-9)
    assert T.local_p(0.0, 11) == 1.0 and T.local_p(np.array([1.0, 2.0]), 11).shape == (2,)


def test_the_local_ratio_has_the_right_tail_under_flat_noise():
    e = G.SplitMix64(5).normal(4000 * 120).reshape(4000, 120)
    p = T.periodogram_inner(e)
    r = p / T.local_mean_background(p, 11)
    assert np.mean(r[:, 5:-5] > T.local_threshold(0.05, 1, 11)) == pytest.approx(0.05, abs=0.006)


def test_median_background_by_hand_and_scale():
    e = np.array([1.0, 5.0, 2.0, 9.0, 3.0, 4.0, 8.0])
    mb = T.median_background(e, 3)
    assert mb[2] == pytest.approx(np.median([5.0, 2.0, 9.0]) / math.log(2.0)) and mb[0] == pytest.approx(np.median([5.0, 1.0, 5.0]) / math.log(2.0))
    flat = G.SplitMix64(6).normal(2000 * 100).reshape(2000, 100)
    p = T.periodogram_inner(flat)
    assert T.median_background(p, 11).mean() > 1.0                                              # Mittel des Medians über ln 2 ist etwa 1,1


def test_single_frequency_edge_cases_of_fisher_and_the_shortest_series():
    assert T.fisher_p(1.0, 1) == 0.0 and T.fisher_p(0.9, 1) == 1.0 and T.fisher_crit(0.05, 1) == 1.0
    e = T.periodogram_inner(np.array([1.0, 2.0, 4.0, 3.0]))
    assert len(e) == 1 and e[0] == pytest.approx(abs(sum((v - 2.5) * np.exp(-2j * np.pi * t / 4) for t, v in enumerate([1.0, 2.0, 4.0, 3.0]))) ** 2 / 4, rel=1e-10)

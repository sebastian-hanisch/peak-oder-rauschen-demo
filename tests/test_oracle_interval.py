"""Intervalle: Gamma-, Beta-, χ²- und t-Funktionen gegen scipy, Welch gegen scipy.signal.welch, Intervalle gegen Handrechnung und Simulation."""

import math

import numpy as np
import pytest
from scipy import special, stats
from scipy.signal import welch as sp_welch

import pkr_interval as I
import signal_generator as G


@pytest.mark.parametrize("a,x", [(0.5, 0.2), (1.0, 3.0), (2.5, 1.0), (2.5, 6.0), (10.0, 4.0), (10.0, 25.0), (50.0, 49.0)])
def test_gammainc_equals_scipy(a, x):
    assert I.gammainc(a, x) == pytest.approx(special.gammainc(a, x), abs=1e-12)


def test_gammainc_edges():
    assert I.gammainc(2.0, 0.0) == 0.0 and I.gammainc(2.0, -1.0) == 0.0
    assert I.chi2_cdf(2.0, 2) == pytest.approx(1 - math.exp(-1.0))                              # χ²_2: 1 - exp(-x/2)


@pytest.mark.parametrize("nu", [2, 7, 13.4, 50, 200])
def test_chi2_ppf_equals_scipy(nu):
    for p in (0.025, 0.5, 0.975, 0.999):
        assert I.chi2_ppf(p, nu) == pytest.approx(stats.chi2.ppf(p, nu), rel=1e-9)
    with pytest.raises(ValueError):
        I.chi2_ppf(1.0, nu)
    with pytest.raises(ValueError):
        I.chi2_ppf(0.0, nu)


def test_chi2_ppf_for_two_degrees_of_freedom_by_hand():
    assert I.chi2_ppf(0.975, 2) == pytest.approx(-2 * math.log(0.025), rel=1e-10) and I.chi2_ppf(0.025, 2) == pytest.approx(-2 * math.log(0.975), rel=1e-10)


@pytest.mark.parametrize("a,b,x", [(1.0, 1.0, 0.3), (2.5, 0.5, 0.3), (0.5, 3.0, 0.8), (4.0, 4.0, 0.5), (3.0, 7.0, 0.95), (3.0, 7.0, 0.05)])
def test_betainc_equals_scipy(a, b, x):
    assert I.betainc(a, b, x) == pytest.approx(special.betainc(a, b, x), abs=1e-12)
    assert I.betainc(a, b, 0.0) == 0.0 and I.betainc(a, b, 1.0) == 1.0 and I.betainc(a, b, -0.1) == 0.0 and I.betainc(a, b, 1.2) == 1.0


@pytest.mark.parametrize("nu", [1, 2, 6, 30])
def test_t_cdf_and_ppf_equal_scipy(nu):
    for t in (-2.5, -0.3, 0.0, 0.7, 3.0):
        assert I.t_cdf(t, nu) == pytest.approx(stats.t.cdf(t, nu), abs=1e-12)
    for p in (0.01, 0.025, 0.2, 0.5, 0.8, 0.975, 0.995):
        assert I.t_ppf(p, nu) == pytest.approx(stats.t.ppf(p, nu), rel=1e-8, abs=1e-10)
    with pytest.raises(ValueError):
        I.t_ppf(1.0, nu)


def test_hann_window_is_the_periodic_one():
    w = I.hann_window(8)
    assert w == pytest.approx([0.0, 0.1464466, 0.5, 0.8535534, 1.0, 0.8535534, 0.5, 0.1464466], abs=1e-6)


def test_welch_equals_scipy_up_to_the_one_sided_factor():
    x = np.random.default_rng(3).normal(size=400) + np.sin(0.9 * np.arange(400))
    est, k, segs = I.welch(x, 100, 50)
    f, ref = sp_welch(x, fs=1.0, window="hann", nperseg=100, noverlap=50, detrend="constant", scaling="density")
    assert k == 7 and segs.shape == (7, 51) and est.shape == (51,)
    assert est[1:50] == pytest.approx(ref[1:50] / 2.0, rel=1e-9)
    est2, k2, segs2 = I.welch(np.stack([x, x * 2]), 100, 50)
    assert est2[1][1:50] == pytest.approx(4.0 * est2[0][1:50], rel=1e-9) and segs2.shape == (2, 7, 51)
    with pytest.raises(ValueError):
        I.welch(x, 500, 50)
    with pytest.raises(ValueError):
        I.welch(x, 100, 0)


def test_welch_dof_by_hand():
    assert I.welch_dof(1) == pytest.approx(36.0 / 18.0) and I.welch_dof(7) == pytest.approx(36.0 * 49 / 132.0)
    assert I.welch_dof(7) == pytest.approx(13.3636, abs=1e-4) and I.welch_dof(100) > 1.8 * 100 * 0.95


def test_chi2_interval_by_hand():
    lo, hi = I.chi2_interval(np.array([1.0, 2.0]), 2.0, 0.95)
    assert lo == pytest.approx([2 / 7.377758908, 4 / 7.377758908], rel=1e-8) and hi == pytest.approx([2 / 0.050635616, 4 / 0.050635616], rel=1e-8)
    lo2, hi2 = I.chi2_interval(np.array([1.0]), 14.0, 0.9)
    assert lo2[0] == pytest.approx(14 / stats.chi2.ppf(0.95, 14), rel=1e-8) and hi2[0] == pytest.approx(14 / stats.chi2.ppf(0.05, 14), rel=1e-8)


def test_jackknife_interval_equals_an_independent_loop():
    rng = np.random.default_rng(7)
    vals = rng.exponential(size=(3, 6)) + 0.5
    lo, hi = I.jackknife_interval(vals, 0.95)
    for i in range(3):
        v = vals[i]
        k = len(v)
        full = math.log(v.mean())
        pseudo = [k * full - (k - 1) * math.log((v.sum() - v[j]) / (k - 1)) for j in range(k)]
        est, se = np.mean(pseudo), np.std(pseudo, ddof=1) / math.sqrt(k)
        q = stats.t.ppf(0.975, k - 1)
        assert lo[i] == pytest.approx(math.exp(est - q * se), rel=1e-8) and hi[i] == pytest.approx(math.exp(est + q * se), rel=1e-8)
    assert np.all(lo < hi)


def test_width_in_db_and_coverage_of_the_chi2_interval_by_simulation():
    assert I.width_db(1.0, 10.0) == pytest.approx(10.0) and I.width_db(np.array([1.0, 2.0]), np.array([100.0, 2.0])).tolist() == pytest.approx([20.0, 0.0])
    e = np.random.default_rng(8).exponential(size=20000)                                        # I ~ S Exp(1) mit S = 1: 2 I / S ~ χ²_2
    lo, hi = I.chi2_interval(e, 2.0, 0.95)
    assert np.mean((lo <= 1.0) & (1.0 <= hi)) == pytest.approx(0.95, abs=0.005)
    assert I.width_db(lo, hi).mean() == pytest.approx(10 * math.log10(7.3778 / 0.050636), abs=1e-3)


def test_welch_chi2_interval_covers_white_noise_density():
    x = G.SplitMix64(9).normal(3000 * 512).reshape(3000, 512)
    est, k, _ = I.welch(x, 128, 64)
    lo, hi = I.chi2_interval(est[:, 20], I.welch_dof(k), 0.95)
    assert np.mean((lo <= 1.0) & (1.0 <= hi)) == pytest.approx(0.95, abs=0.012)


def test_t_ppf_rejects_both_ends():
    for p in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError):
            I.t_ppf(p, 5)
        with pytest.raises(ValueError):
            I.chi2_ppf(p, 5)


def test_welch_accepts_a_single_segment_and_a_hop_of_one_and_rejects_bad_arguments_with_a_message():
    x = np.arange(20.0) + np.sin(np.arange(20.0))
    est, k, segs = I.welch(x, 20, 5)                                      # Segment gleich Reihe: ein Segment
    assert k == 1 and segs.shape == (1, 11)
    est1, k1, _ = I.welch(x, 8, 1)                                        # Schrittweite 1: 13 Segmente
    assert k1 == 13
    est2, k2, segs2 = I.welch(x, 8, 13)                                   # n - seg + 1 = 13 ist ein Vielfaches der Schrittweite: das Segment ab 13 gäbe es nicht
    assert k2 == 1 and segs2.shape == (1, 5)
    with pytest.raises(ValueError, match="Segment länger"):
        I.welch(x, 25, 5)
    with pytest.raises(ValueError, match="Segment länger"):
        I.welch(x, 8, 0)
    with pytest.raises(ValueError, match="Segment länger"):
        I.welch(x, 25, 0)

"""Vorfilterung: Burg von Hand, Residuen, Weißmachen eines AR-Prozesses."""

import numpy as np
import pytest

import pkr_prewhiten as P
import signal_generator as G


def test_burg_of_the_mini_series_by_hand():
    a, err, refl = P.burg(np.array([1.0, 2.0, 3.0, 4.0]), 1)                                    # a = 2 (2 + 6 + 12) / ((4 + 9 + 16) + (1 + 4 + 9)) = 40/43
    assert a == pytest.approx([40.0 / 43.0], abs=1e-12) and refl == pytest.approx([40.0 / 43.0], abs=1e-12)
    assert err[0] == pytest.approx(7.5) and err[1] == pytest.approx(7.5 * (1 - (40.0 / 43.0) ** 2), abs=1e-12)


def test_burg_keeps_the_reflection_coefficients_inside_the_unit_interval_and_handles_zeros():
    a, err, refl = P.burg(np.random.default_rng(1).normal(size=64), 12)
    assert np.all(np.abs(refl) < 1.0) and np.all(np.diff(err) < 0) and len(a) == 12
    a0, err0, refl0 = P.burg(np.zeros(20), 3)
    assert np.all(a0 == 0.0) and np.all(np.isfinite(err0)) and np.all(refl0 == 0.0)


def test_residuals_by_hand():
    x = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
    assert P.residuals(x, [0.5]).tolist() == pytest.approx([2 - 0.5, 4 - 1.0, 7 - 2.0, 11 - 3.5])
    assert P.residuals(x, [1.0, -1.0]).tolist() == pytest.approx([4 - 2 + 1, 7 - 4 + 2, 11 - 7 + 4])
    assert P.residuals(x, []).tolist() == x.tolist()
    assert P.residuals(np.stack([x, 2 * x]), [0.5]).shape == (2, 4) and P.residuals(np.stack([x, 2 * x]), [0.5])[1].tolist() == pytest.approx([2 * v for v in P.residuals(x, [0.5])])


def test_prewhitening_an_ar2_series_leaves_white_residuals():
    poles = ((0.05, 0.97),)
    sig = G.ar_unit_sigma(G.ar_from_poles(poles))
    x = G.make_noise_batch(1, 4000, poles, sig, 0.0, G.SplitMix64(3))[0]
    res = P.prewhiten(x, 2)
    assert len(res) == 3998 and abs(res.mean()) < 0.05
    r = res - res.mean()
    assert abs(np.sum(r[1:] * r[:-1]) / np.sum(r * r)) < 0.04 and abs(np.sum(r[2:] * r[:-2]) / np.sum(r * r)) < 0.04
    assert res.var() < 0.1 * x.var()                                                            # der steile Anteil ist herausgefiltert


def test_prewhiten_handles_rows_and_removes_the_mean():
    x = np.stack([np.random.default_rng(i).normal(size=50) + 10.0 for i in range(3)])
    res = P.prewhiten(x, 2)
    assert res.shape == (3, 48) and abs(res.mean()) < 0.5
    assert res[1] == pytest.approx(P.prewhiten(x[1], 2), abs=1e-12)

"""Festwerte der Experimente (feste Zufallsströme): Das Ergebnis ist bei jedem Lauf dasselbe; Abweichungen bedeuten eine geänderte Rechnung."""

import pytest

from sweeps_cache import get

ALARM = {32: (15, 0.58145, 0.02, 0.0205, 0.04955, 0.536709), 128: (63, 0.98885, 0.037, 0.0382, 0.05125, 0.960501), 512: (255, 1.0, 0.04225, 0.04385, 0.0488, 0.999998), 2048: (1023, 1.0, 0.0526, 0.0548, 0.0568, 1.0)}

POWER = {(128, -9): (0.427, 0.4075), (128, -6): (0.8415, 0.837), (128, -3): (0.9935, 0.9935), (128, 0): (1.0, 1.0), (512, -9): (0.9935, 0.993), (512, -6): (1.0, 1.0), (512, -3): (1.0, 1.0), (512, 0): (1.0, 1.0)}

BG_ALARM = {(128, 0.0): (0.0545, 0.05525, 0.0325, 0.0515, 0.296), (128, 0.5): (0.611, 0.05075, 0.02725, 0.04975, 0.28725), (128, 0.8): (0.994, 0.0645, 0.0345, 0.05825, 0.3225), (128, 0.95): (1.0, 0.11375, 0.0325, 0.21875, 0.57), (512, 0.0): (0.04675, 0.0435, 0.03725, 0.05475, 0.48575), (512, 0.5): (0.877, 0.05125, 0.0385, 0.0545, 0.4985), (512, 0.8): (1.0, 0.05475, 0.043, 0.053, 0.47775), (512, 0.95): (1.0, 0.07625, 0.047, 0.06175, 0.50625)}

BG_POWER = {-18: (0.0, 0.803333, 0.422333), -15: (0.001333, 0.976, 0.626), -12: (0.067, 0.999, 0.718333), -9: (0.528667, 1.0, 0.768667), -6: (0.925333, 1.0, 0.785667)}

INTERVAL = {(0.9, 'Gipfel', 'Periodogramm χ²'): (0.9515, 21.634684, 2.0), (0.9, 'Gipfel', 'Welch χ²'): (0.947, 6.833821, 13.363636), (0.9, 'Gipfel', 'Welch Jackknife'): (0.93, 7.983133, 13.363636), (0.9, 'Flanke', 'Periodogramm χ²'): (0.94325, 21.634684, 2.0), (0.9, 'Flanke', 'Welch χ²'): (0.94775, 6.833821, 13.363636), (0.9, 'Flanke', 'Welch Jackknife'): (0.92675, 8.035045, 13.363636), (0.9, 'Tal', 'Periodogramm χ²'): (0.94725, 21.634684, 2.0), (0.9, 'Tal', 'Welch χ²'): (0.9535, 6.833821, 13.363636), (0.9, 'Tal', 'Welch Jackknife'): (0.943, 8.062778, 13.363636), (0.98, 'Gipfel', 'Periodogramm χ²'): (0.9535, 21.634684, 2.0), (0.98, 'Gipfel', 'Welch χ²'): (0.9085, 6.833821, 13.363636), (0.98, 'Gipfel', 'Welch Jackknife'): (0.88425, 7.650282, 13.363636), (0.98, 'Flanke', 'Periodogramm χ²'): (0.94075, 21.634684, 2.0), (0.98, 'Flanke', 'Welch χ²'): (0.952, 6.833821, 13.363636), (0.98, 'Flanke', 'Welch Jackknife'): (0.9335, 7.977737, 13.363636), (0.98, 'Tal', 'Periodogramm χ²'): (0.94025, 21.634684, 2.0), (0.98, 'Tal', 'Welch χ²'): (0.95075, 6.833821, 13.363636), (0.98, 'Tal', 'Welch Jackknife'): (0.93725, 7.975999, 13.363636)}

PREWHITEN = {'white': 1.0, 'known': 0.8885, 'prewhitened': 0.05, 'dynamic_db': 63.717128}



def test_alarm_golden():
    rows = get("alarm")
    assert [r["n"] for r in rows] == list(ALARM)
    for r in rows:
        m, single, bonf, sidak, fisher, theory = ALARM[r["n"]]
        assert r["m"] == m and (r["single"], r["bonferroni"], r["sidak"], r["fisher"], r["theory"]) == pytest.approx((single, bonf, sidak, fisher, theory), abs=1e-6)


def test_power_golden():
    rows = get("power")
    assert len(rows) == len(POWER) == 8
    for r in rows:
        assert (r["detected"], r["right_place"]) == pytest.approx(POWER[(r["n"], r["snr"])], abs=1e-6) and r["runs"] == 2000


def test_background_golden():
    res = get("background")
    assert len(res["alarm"]) == len(BG_ALARM) == 8 and len(res["power"]) == len(BG_POWER) == 5
    for r in res["alarm"]:
        assert (r["white"], r["known"], r["fitted"], r["local"], r["median"]) == pytest.approx(BG_ALARM[(r["n"], r["phi"])], abs=1e-6) and r["runs"] == 4000
    for r in res["power"]:
        assert (r["white"], r["fitted"], r["local"]) == pytest.approx(BG_POWER[r["snr"]], abs=1e-6) and r["runs"] == 3000


def test_interval_golden():
    rows = get("interval")
    assert len(rows) == len(INTERVAL) == 18
    for r in rows:
        cov, width, nu = INTERVAL[(r["r"], r["position"], r["method"])]
        assert (r["coverage"], r["width_db"], r["nu"]) == pytest.approx((cov, width, nu), abs=1e-5) and r["K"] == 7 and r["runs"] == 4000


def test_prewhiten_golden():
    res = get("prewhiten")
    for k, v in PREWHITEN.items():
        assert res[k] == pytest.approx(v, abs=1e-5), k
    assert res["n"] == 256 and res["runs"] == 2000

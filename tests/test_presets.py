"""Presets: Vollständigkeit, gültige Werte, Standardwerte."""

import pkr_constants as C
import pkr_evaluation as ev
import pkr_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_choices_and_bounds():
    for name, p in C.PRESETS.items():
        assert p["n"] in C.N_CHOICES and p["background"] in C.BACKGROUNDS and p["line_mode"] in C.LINE_MODES and p["test"] in C.TESTS and p["band"] in C.BAND_MODES, name
        assert C.LINE_MIN <= p["line"] <= C.LINE_MAX and C.SNR_MIN <= p["snr"] <= C.SNR_MAX and C.ALPHA_MIN <= p["alpha"] <= C.ALPHA_MAX and 0 <= p["seed"] <= C.SEED_MAX


def test_default_settings_are_valid_choices():
    s = ev.Settings()
    assert s.n in C.N_CHOICES and s.background in C.BACKGROUNDS and s.line_mode in C.LINE_MODES and s.test in C.TESTS and s.band in C.BAND_MODES
    assert C.ALPHA_MIN <= s.alpha <= C.ALPHA_MAX and C.LINE_MIN <= s.line <= C.LINE_MAX and C.SNR_MIN <= s.snr <= C.SNR_MAX


def test_setting_specs_cover_every_preset_key_and_the_defaults_match():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    s = ev.Settings()
    defaults = {"n": s.n, "background": s.background, "line_mode": s.line_mode, "line": s.line, "snr": s.snr, "test": s.test, "alpha": s.alpha, "band": s.band, "seed": s.seed}
    for key, state_key in P.PRESET_KEYS.items():
        assert P.SETTING_SPECS[state_key].default == defaults[key], key


def test_preset_names_are_short_enough_for_the_buttons_and_url_params_are_unique():
    assert max(len(n) for n in C.PRESETS) <= 48
    params = [spec.url_param for spec in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_every_background_has_poles_and_the_tests_are_distinct():
    assert set(C.BACKGROUND_POLES) == set(C.BACKGROUNDS) and C.BACKGROUND_POLES["weiß"] == () and len(set(C.TESTS)) == 7
    assert C.DEFAULT_TEST in C.TESTS and C.LINE_MIN < C.DEFAULT_LINE < C.LINE_MAX

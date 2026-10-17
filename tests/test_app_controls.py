"""AppTest: Auswahlfelder, Permalink, bedingte Regler, PDF (die Rauchtests in test_app.py decken nur Knöpfe und Slider ab)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import pkr_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 300
LINE = "Frequenz der Linie (Zyklen je Abtastwert)"
SNR = "Linie gegen die Hintergrundvarianz (dB)"


def _run(params=None):
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    if params:
        at.query_params.update(params)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _labels(at):
    return [x.label for x in at.slider]


@pytest.mark.parametrize("background", list(C.BACKGROUNDS))
def test_every_background_runs(background):
    at = _run()
    at.selectbox(key="background_select").set_value(background).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


@pytest.mark.parametrize("test", list(C.TESTS))
def test_every_test_runs(test):
    at = _run()
    at.selectbox(key="test_select").set_value(test).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_the_line_sliders_exist_only_with_a_line_and_keep_their_values():
    at = _run()
    assert LINE not in _labels(at) and SNR not in _labels(at)
    at.radio(key="line_mode_select").set_value("an").run(timeout=TIMEOUT)
    assert not at.exception and LINE in _labels(at) and SNR in _labels(at)
    at2 = _run({"lm": "an", "f": "0.37", "snr": "-14"})
    at2.radio(key="line_mode_select").set_value("aus").run(timeout=TIMEOUT)
    at2.radio(key="line_mode_select").set_value("an").run(timeout=TIMEOUT)
    by = {x.label: x for x in at2.slider}
    assert by[LINE].value == 0.37 and by[SNR].value == -14


def test_permalink_is_applied_and_invalid_values_are_ignored():
    at = _run({"n": "512", "bg": "AR(1) φ = 0,8", "lm": "an", "f": "0.9", "snr": "-99", "t": "Šidák (weiß)", "a": "0.5", "b": "an", "seed": "42"})
    assert at.select_slider(key="n_select").value == 512 and at.selectbox(key="background_select").value == "AR(1) φ = 0,8" and at.radio(key="line_mode_select").value == "an"
    assert at.selectbox(key="test_select").value == "Šidák (weiß)" and at.selectbox(key="band_select").value == "an" and at.number_input(key="seed_input").value == 42
    by = {x.label: x for x in at.slider}
    assert by[LINE].value == C.LINE_MAX and by[SNR].value == C.SNR_MIN and by["Niveau α (Falschalarmrate)"].value == C.ALPHA_MAX
    at2 = _run({"n": "3000", "bg": "unsinn", "lm": "vielleicht", "t": "unsinn", "b": "x"})
    assert at2.select_slider(key="n_select").value == C.DEFAULT_N and at2.selectbox(key="background_select").value == C.DEFAULT_BACKGROUND
    assert at2.radio(key="line_mode_select").value == C.DEFAULT_LINE_MODE and at2.selectbox(key="test_select").value == C.DEFAULT_TEST and at2.selectbox(key="band_select").value == C.DEFAULT_BAND


def test_preset_button_sets_the_widgets():
    at = _run()
    at.button(key="preset_Schwache Linie auf rotem Hintergrund").click().run(timeout=TIMEOUT)
    assert not at.exception
    assert at.radio(key="line_mode_select").value == "an" and at.select_slider(key="n_select").value == 512 and {x.label: x for x in at.slider}[SNR].value == -12
    assert at.selectbox(key="test_select").value == "AR(1) angepasst + Šidák"
    at.button(key="preset_Steiler Hintergrund: Vorfilterung").click().run(timeout=TIMEOUT)
    assert at.selectbox(key="background_select").value == "steil (AR(2))" and at.radio(key="line_mode_select").value == "aus"


def test_pdf_button_builds_a_download():
    at = _run()
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any(d.proto.label.startswith("⬇️") for d in at.get("download_button"))


def test_the_dice_button_changes_the_seed():
    at = _run()
    before = at.number_input(key="seed_input").value
    [b for b in at.button if "würfeln" in b.label][0].click().run(timeout=TIMEOUT)
    assert not at.exception and at.number_input(key="seed_input").value != before

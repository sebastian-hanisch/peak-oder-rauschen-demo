"""Peak oder Rauschen? Ein höchster Gipfel ist noch kein Befund - die Schwelle hängt an der Zahl der Frequenzen und am Hintergrund - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 8 der Reihe „Spektral- und Wavelet-Analyse“. Ein simuliertes Signal (weißer oder gefärbter Hintergrund, optional eine Linie) wird mit sieben Tests auf „Peak oder Rauschen“ geprüft;
gezeigt werden Falschalarm und Erkennungsleistung, der Einfluss des Hintergrunds, Konfidenzintervalle und die Vorfilterung. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import pkr_constants as C
import pkr_evaluation as ev
from pkr_evaluation import Settings, analyse, de, outcome
from pkr_pdf_export import build_pdf
from pkr_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from pkr_visualization import (
    build_alarm,
    build_background_alarm,
    build_background_power,
    build_interval,
    build_power,
    build_prewhiten,
    build_spectrum,
    build_test,
)

st.set_page_config(page_title="Peak oder Rauschen – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Rechne die Experimente …")
def _experiment(name):
    return getattr(ev, "sweep_" + name)()


st.title("🔍 Peak oder Rauschen? Wann ist ein Gipfel ein Befund")
st.markdown(
    """
Jede Aufzeichnung hat einen **höchsten Gipfel**, auch reines Rauschen. Ob er etwas bedeutet, entscheidet eine **Schwelle**, und die hängt an zwei Dingen: an der **Zahl der Frequenzen**, unter denen der höchste
Gipfel ausgesucht wurde (das Maximum über einige Dutzend Werte überschreitet die Schwelle für einen einzelnen fast immer), und am **Hintergrund**, gegen den getestet wird (rotes Rauschen hat bei tiefen Frequenzen von selbst
hohe Gipfel). Dazu kommt die Frage, wie unsicher ein Spektrum überhaupt ist: das **Konfidenzintervall**.

Dies ist die achte Demo der Reihe „Spektral- und Wavelet-Analyse“, auf die Demos zu **Periodogramm**, **Welch**, **Multitaper**, **AR-Spektrum** und **Lomb-Scargle** aufbauend (dort: die Verteilung der Leistung
unter Rauschen ist nur näherungsweise exponentiell). Mehrfachtests für A/B-Varianten behandelt die Demo „Mehrfachtests“; hier geht es um die Frequenzen **eines** Spektrums. Alles ist **simuliert**: Hintergrund und
Linie sind bekannt. Weiter unten: **🔬 Experimente**, **🚧 Wo die Annahmen enden** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So prüft man einen Gipfel", expanded=True):
    st.markdown(
        """
1. **Teststatistik:** Unter weißem Gaußrauschen sind die Werte des Periodogramms an den inneren Fourier-Frequenzen unabhängig exponentialverteilt. Der höchste Wert ist das Maximum über *m* solcher Werte.
2. **Schwelle für das Maximum:** Die Schwelle für eine Frequenz (−ln α) ist viel zu niedrig. **Bonferroni** und **Šidák** rechnen das Niveau auf *m* Frequenzen um, **Fishers g** (höchster Wert geteilt durch die Summe) ist die exakte Fassung,
   wenn die Varianz aus den Daten geschätzt wird.
3. **Hintergrund:** Bei gefärbtem Rauschen muss gegen den Hintergrund getestet werden: gegen ein angepasstes AR(1)-Modell, gegen den lokalen Mittelwert der Nachbarn (mit F-Schwelle) oder nach einer **Vorfilterung**,
   die das Signal mit einem AR-Modell weiß macht.
4. **Intervall:** Das Periodogramm hat zwei Freiheitsgrade und ist sehr unsicher (95-%-Intervall über 20 dB breit); Welch mittelt und verengt es.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Daten**")
    n = st.select_slider("Länge n der Aufzeichnung", options=list(C.N_CHOICES), key="n_select", help="Zahl der Messwerte; es gibt m = (n − 1)/2 innere Fourier-Frequenzen.")
    background = st.selectbox("Hintergrund", list(C.BACKGROUNDS), key="background_select",
                              help="Varianz immer 1. Weiß: Exp(1)-Periodogramm. AR(1): rotes Rauschen mit Pol bei 0. Steil: AR(2) mit Resonanz bei 0,02 und Radius 0,99, 64 dB Spannweite.")
    line_mode = st.radio("Linie", list(C.LINE_MODES), key="line_mode_select", horizontal=True, help="Eine Kosinuslinie mit zufälliger Phase zum Hintergrund addiert.")
    if line_mode == "an":
        line = st.slider("Frequenz der Linie (Zyklen je Abtastwert)", C.LINE_MIN, C.LINE_MAX, float(st.session_state["line_kept"]), step=C.LINE_STEP, format="%.3f", help="Lage der Linie.")
        st.session_state["line_kept"] = float(line)
        snr = st.slider("Linie gegen die Hintergrundvarianz (dB)", C.SNR_MIN, C.SNR_MAX, int(st.session_state["snr_kept"]),
                        help="Linienleistung A²/2 gegen die Varianz 1 des Hintergrunds. 0 dB: gleich viel Leistung, verteilt auf eine Frequenz statt auf alle.")
        st.session_state["snr_kept"] = int(snr)
    else:
        line, snr = float(st.session_state["line_kept"]), int(st.session_state["snr_kept"])
    st.markdown("**Test**")
    test = st.selectbox("Test", list(C.TESTS), key="test_select",
                        help="Einzelfrequenz: Schwelle für eine feste Frequenz. Bonferroni, Šidák, Fishers g: Schwelle für das Maximum über alle Frequenzen, gegen weißes Rauschen. AR(1) angepasst: Quotient durch "
                             "ein angepasstes AR(1)-Spektrum. Lokaler Mittelwert: Quotient durch den Mittelwert der 10 Nachbarn, F-Schwelle. Vorfilterung: AR(2) per Burg, Fisher-Test auf den Residuen.")
    alpha = st.slider("Niveau α (Falschalarmrate)", C.ALPHA_MIN, C.ALPHA_MAX, key="alpha_slider", step=C.ALPHA_STEP, format="%.2f", help="Erlaubte Wahrscheinlichkeit eines Alarms bei reinem Hintergrund.")
    band = st.selectbox("95-%-Konfidenzband", list(C.BAND_MODES), key="band_select", help="Zeigt das χ²-Intervall mit zwei Freiheitsgraden um jeden Wert des Periodogramms.")
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neu würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_select": int(n), "background_select": background, "line_mode_select": line_mode, "line_kept": float(line), "snr_kept": int(snr), "test_select": test,
                   "alpha_slider": float(alpha), "band_select": band, "seed_input": int(seed)})

settings = Settings(int(n), background, line_mode, float(line), int(snr), test, float(alpha), band, int(seed))
a = _analysis(settings)
res = a.results[test]

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Höchster Gipfel bei", de(res.peak_hz, 3), delta=("Linie bei " + de(settings.line, 3)) if settings.has_line else "keine Linie", delta_color="off", help="Zyklen je Abtastwert.")
m2.metric("Quotient gegen Schwelle", f"{de(res.stat, 1)} / {de(res.threshold, 1)}", delta=f"{a.m} Frequenzen", delta_color="off", help="Quotient Periodogramm / Hintergrund des gewählten Tests, Schwelle für das Maximum.")
m3.metric("p-Wert des höchsten Gipfels", de(res.p, 3), delta=f"Niveau {de(settings.alpha, 2)}", delta_color="off", help="Wahrscheinlichkeit, bei reinem Hintergrund einen mindestens so hohen höchsten Gipfel zu sehen.")
m4.metric("Entscheidung", "Alarm" if res.alarm else "kein Alarm", delta=outcome(res, settings.has_line), delta_color="off")
st.plotly_chart(build_spectrum(a), width="stretch", key="spectrum")
st.caption("Periodogramm der Aufzeichnung (blau) mit der wahren Dichte des Hintergrunds (rot gestrichelt), dem angepassten AR(1) (grün gepunktet) und, falls gewählt, dem 95-%-Intervall. Der lokale Mittelwert lässt sich in der Legende einblenden. "
           "Rot strichpunktiert: die Linie.")
st.plotly_chart(build_test(a), width="stretch", key="test")
st.caption("Teststatistik des gewählten Tests: Quotient aus Periodogramm und Hintergrund je Frequenz; gestrichelt die Schwelle für das **Maximum** über alle Frequenzen. Bei der Vorfilterung gehört die Statistik zu den Residuen "
           "des AR(2)-Modells.")
st.dataframe(pd.DataFrame([{"Test": r.name, "Quotient": de(r.stat, 1), "Schwelle": de(r.threshold, 1), "p-Wert": de(r.p, 3), "Gipfel bei": de(r.peak_hz, 3), "Entscheidung": "Alarm" if r.alarm else "kein Alarm",
                            "Einordnung": outcome(r, settings.has_line)} for r in a.results.values()]), hide_index=True, width="stretch")
st.caption("Alle sieben Tests auf derselben Aufzeichnung. „Einordnung“ vergleicht die Entscheidung mit der Wahrheit (Linie ja oder nein; Gipfel innerhalb von 1,5/n der Linie gilt als richtige Stelle).")

if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = build_pdf(a)
    st.session_state["pdf_for"] = settings
if st.session_state.get("pdf_for") == settings and "pdf_bytes" in st.session_state:
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="peak_oder_rauschen.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Ist der höchste Gipfel ein Befund?")
kind, text = ev.verdict_main(a)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experimente")
st.caption(f"Feste Zufallsströme (Seeds ab {C.EXP_ALARM_SEED}), unabhängig von den Reglern. Niveau 5 %, Rauschen mit Varianz 1, Rechteckfenster-Periodogramm an den inneren Fourier-Frequenzen. "
           "„Alarm“ heißt: der höchste Gipfel überschreitet die Schwelle des Tests.")

st.markdown("#### Falschalarm des höchsten Gipfels")
alarm = _experiment("alarm")
st.caption("Nur weißes Rauschen: Anteil der Läufe mit Alarm bei Niveau 5 %. Die Statistik ist der Quotient aus Periodogramm und seinem Mittelwert; „theoretisch“ ist 1 − 0,95^m für unabhängige Frequenzen mit bekannter Varianz.")
st.dataframe(pd.DataFrame([{"n": r["n"], "Frequenzen m": r["m"], "Läufe": f"{r['runs']:,}".replace(",", "."), "Einzelfrequenz": f"{de(100 * r['single'], 1)} %", "Bonferroni": f"{de(100 * r['bonferroni'], 1)} %",
                            "Šidák": f"{de(100 * r['sidak'], 1)} %", "Fishers g": f"{de(100 * r['fisher'], 1)} %", "theoretisch (Einzelfrequenz)": f"{de(100 * r['theory'], 1)} %"} for r in alarm]),
             hide_index=True, width="stretch")
st.plotly_chart(build_alarm(alarm), width="stretch", key="exp_alarm")
kind, text = ev.verdict_alarm(alarm)
getattr(st, kind)(text)

st.markdown("#### Erkennungsleistung")
power = _experiment("power")
st.caption(f"Eine Linie in zufälliger Lage (0,1 bis 0,4) in weißem Rauschen, Fishers g bei 5 %, {C.EXP_POWER_RUNS} Läufe je Zelle; Anteil „erkannt an der richtigen Stelle“ (Alarm und Gipfel innerhalb von 1,5/n der Linie).")
st.dataframe(pd.DataFrame([{"n": n_, **{f"{s_} dB": f"{de(100 * [r for r in power if r['n'] == n_ and r['snr'] == s_][0]['right_place'], 0)} %" for s_ in C.EXP_POWER_SNR}} for n_ in C.EXP_POWER_NS]),
             hide_index=True, width="stretch")
st.plotly_chart(build_power(power), width="stretch", key="exp_power")
kind, text = ev.verdict_power(power)
getattr(st, kind)(text)

st.markdown("#### Gefärbter Hintergrund")
bg = _experiment("background")
st.caption(f"AR(1)-Hintergrund mit Varianz 1, **nur Rauschen**, {C.EXP_BG_RUNS} Läufe je Zelle, Niveau 5 %. Fünf Tests: weißer Test (Fishers g), Quotient durch den **bekannten** Hintergrund (Obergrenze, Šidák), durch das angepasste AR(1) (Šidák), "
           "durch den lokalen Mittelwert mit F-Schwelle und durch einen gleitenden Median mit Šidák-Schwelle (naiv).")
st.dataframe(pd.DataFrame([{"n": r["n"], "φ": de(r["phi"], 2), "weiß": f"{de(100 * r['white'], 1)} %", "bekannt": f"{de(100 * r['known'], 1)} %", "AR(1) angepasst": f"{de(100 * r['fitted'], 1)} %",
                            "lokaler Mittelwert": f"{de(100 * r['local'], 1)} %", "gleitender Median": f"{de(100 * r['median'], 1)} %"} for r in bg["alarm"]]), hide_index=True, width="stretch")
st.plotly_chart(build_background_alarm(bg["alarm"]), width="stretch", key="exp_bg_alarm")
st.caption(f"Erkennung einer Linie (zufällige Lage 0,15 bis 0,4) auf AR(1) mit φ = {C.EXP_BGP_PHI:g}, n = {C.EXP_BGP_N}, {C.EXP_BGP_RUNS} Läufe: Anteil erkannt an der richtigen Stelle. "
           "Der weiße Test meldet fast immer, aber an der falschen Stelle (am Hintergrund).")
st.dataframe(pd.DataFrame([{"Linie gegen Hintergrund": f"{r['snr']} dB", "weiß": f"{de(100 * r['white'], 0)} %", "AR(1) angepasst": f"{de(100 * r['fitted'], 0)} %", "lokaler Mittelwert": f"{de(100 * r['local'], 0)} %"}
                           for r in bg["power"]]), hide_index=True, width="stretch")
st.plotly_chart(build_background_power(bg["power"]), width="stretch", key="exp_bg_power")
kind, text = ev.verdict_background(bg)
getattr(st, kind)(text)

st.markdown("#### Konfidenzintervalle des Spektrums")
interval = _experiment("interval")
st.caption(f"AR(2)-Resonanz bei {C.EXP_INT_RESONANCE:g} (Radius {C.EXP_INT_RS[0]:g} bzw. {C.EXP_INT_RS[1]:g}), n = {C.EXP_INT_N}, {C.EXP_INT_RUNS} Läufe; 95-%-Intervalle, geprüft wird, wie oft sie die **wahre Dichte** an der Stelle "
           f"überdecken. Welch: Hann-Fenster, Länge {C.EXP_INT_SEG}, 50 % Überlappung; Freiheitsgrade ν = 36K²/(19K − 1). Jackknife: Pseudowerte des Logarithmus über die K Segmente, t-Quantil mit K − 1 Freiheitsgraden.")
st.dataframe(pd.DataFrame([{"Radius": de(r["r"], 2), "Stelle": r["position"], "Verfahren": r["method"], "Freiheitsgrade": de(r["nu"], 1), "Überdeckung": f"{de(100 * r['coverage'], 1)} %", "Breite": f"{de(r['width_db'], 1)} dB"}
                           for r in interval]), hide_index=True, width="stretch")
st.plotly_chart(build_interval(interval), width="stretch", key="exp_interval")
kind, text = ev.verdict_interval(interval)
getattr(st, kind)(text)

st.markdown("#### Vorfilterung bei steilem Hintergrund")
pre = _experiment("prewhiten")
st.caption(f"AR(2)-Hintergrund mit Resonanz bei 0,02 und Radius 0,99 (Spannweite {de(pre['dynamic_db'], 0)} dB), n = {pre['n']}, {pre['runs']} Läufe, **nur Rauschen**, Niveau 5 %. Das Rechteckfenster lässt die Leistung des steilen Gipfels "
           "in die hohen Frequenzen lecken.")
st.plotly_chart(build_prewhiten(pre), width="stretch", key="exp_prewhiten")
kind, text = ev.verdict_prewhiten(pre)
getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Gaußsches, stationäres Rauschen** | Die Verteilung der Statistik ändert sich; die Schwellen sind dann nicht mehr exakt. | – |
| **Der Hintergrund ist weiß oder bekannt** | Der weiße Test meldet bei starkem rotem Rauschen (φ ≥ 0,8) fast immer; der Hintergrund muss gegen angepasst, lokal geschätzt oder vorgefiltert werden. | Zeit-Frequenz-Signifikanz |
| **Eine Linie, ein Test** | Mehrere Linien oder mehrere Tests brauchen eine weitere Korrektur; Fishers g prüft nur den höchsten Gipfel. | Linienzahl (MUSIC, MDL) |
| **Das Rechteckfenster ist erlaubt** | Bei steilen Spektren lässt die Leckage selbst den wahren Hintergrund als Maßstab scheitern. | Fenster, Multitaper |
| **Die Frequenzen sind unabhängig** | Zero-Padding oder unregelmäßige Zeiten erzeugen mehr Gitterpunkte, aber nicht mehr unabhängige; die Zahl m ist dann unklar. | Lomb-Scargle |
| **Das Intervall deckt die Wahrheit** | Am scharfen Gipfel streut Welch stärker, als ν = 13,4 annimmt (überlappende Segmente, Gipfel schmaler als das Fenster); das Intervall ist zu schmal. Das Jackknife unterdeckt bei wenigen, überlappenden Segmenten; mehr Segmente allein beheben das nicht. | Bootstrap |
""",
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Statistik.** $I_k = |X_k|^2/n$ der mittelwertfreien Reihe an $f_k = k/n$, $k = 1,\dots,m$, $m = \lfloor (n-1)/2\rfloor$. Unter weißem Gaußrauschen mit Varianz $\sigma^2$: $I_k/\sigma^2 \sim \mathrm{Exp}(1)$, unabhängig.

**Schwellen für das Maximum** (Niveau $\alpha$, bekannte Varianz): eine Frequenz $z = -\ln\alpha$; Bonferroni $z = -\ln(\alpha/m)$; Šidák $z = -\ln\bigl(1-(1-\alpha)^{1/m}\bigr)$.

**Fishers g.** $g = \max_k I_k / \sum_k I_k$; $P(g' > g) = \sum_{j=1}^{\lfloor 1/g\rfloor} (-1)^{j-1}\binom{m}{j}(1-jg)^{m-1}$ (Fisher 1929). Der Test braucht die Varianz nicht.

**Hintergrund.** AR(1): $S(f) = \sigma_e^2/|1-\varphi e^{-2\pi i f}|^2$, $\hat\varphi = r_1/r_0$, $\hat\sigma_e^2 = r_0(1-\hat\varphi^2)$; Test auf $\max_k I_k/\hat S(f_k)$ mit der Šidák-Schwelle.
Lokaler Mittelwert: $\hat S_k$ = Mittel der $N = 10$ Nachbarn; $I_k/\hat S_k$ hat die Verteilung $P(R > r) = (1 + r/N)^{-N}$ (F-Verteilung $F_{2,2N}$), die Schwelle folgt mit dem Šidák-Niveau je Frequenz.

**Konfidenzintervall.** Für $\hat S$ mit $\nu$ Freiheitsgraden: $\bigl[\nu\hat S/\chi^2_{\nu;\,1-a/2},\ \nu\hat S/\chi^2_{\nu;\,a/2}\bigr]$; Periodogramm $\nu = 2$, Welch (Hann, 50 %) $\nu = 36K^2/(19K-1)$.
Jackknife: Pseudowerte $K\ln\bar S - (K-1)\ln\bar S_{(-i)}$ über die $K$ Segmente, Intervall $\bar p \pm t_{K-1}\,\mathrm{se}$ im Logarithmus.

**Vorfilterung.** AR($p$) per Burg anpassen, Residuen $e_t = x_t - \sum_k a_k x_{t-k}$; Fisher-Test auf dem Periodogramm der Residuen.

**Whittle.** Die Whittle-Näherung der Likelihood, $\sum_k [\ln S(f_k;\theta) + I_k/S(f_k;\theta)]$, schätzt die Parameter eines parametrischen Hintergrunds. Für AR(1) hat sie bei sehr kurzen Reihen in einer Nachmessung (n = 16 bis 64) einen etwa halb so großen Bias wie
Yule-Walker; der Gewinn im mittleren Fehler ist klein (0 bis 10 %); hier nicht gebaut.

**Quellen.** Lehrbuchwissen und Aufsätze (Fisher 1929; Torrence und Compo 1998; Mann und Lees 1996; Whittle 1953). Formeln hier per Simulation, Handrechnung und gegen scipy geprüft; die Quellen selbst wurden nicht gelesen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Spektren und Wavelets: von Fourier zum Vergleich](https://sebastianhanisch.net/konzepte-spektral-wavelet-analyse.html)."
)

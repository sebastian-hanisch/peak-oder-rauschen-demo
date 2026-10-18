# Peak oder Rauschen? Ein höchster Gipfel ist noch kein Befund – die Schwelle hängt an der Zahl der Frequenzen und am Hintergrund – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-peak-oder-rauschen-demo.streamlit.app/)** -->

Achte Demo der Reihe **Spektral- und Wavelet-Analyse** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, aufbauend auf den Demos zum **Periodogramm**,
zu **Welch**, zum **Multitaper**, zum **AR-Spektrum** und zu **Lomb-Scargle**. Ein simuliertes Signal (weißer oder gefärbter Hintergrund, optional eine Linie) wird mit sieben Tests auf „Peak oder Rauschen“ geprüft;
Falschalarm und Erkennungsleistung, der Einfluss des Hintergrunds, Konfidenzintervalle und die Vorfilterung werden gegen die Wahrheit gemessen.

**Abgrenzung:** Mehrfachtests für A/B-Varianten (Bonferroni, Holm, Benjamini-Hochberg) behandelt die Demo `multiple-testing-demo`. Hier sind die m Tests die Frequenzen **eines** Spektrums; ihre Statistik hat eine bekannte
Verteilung (Fishers g ist exakt), und der Hintergrund kann gefärbt sein.

## Warum dieses Problem

Jede Aufzeichnung hat einen höchsten Gipfel, auch reines Rauschen. Die Schwelle für eine einzelne Frequenz reicht für das Maximum über viele Frequenzen nicht (ab einigen Dutzend Frequenzen überschreitet es sie fast immer), und bei rotem Rauschen ist der Hintergrund selbst bei
tiefen Frequenzen hoch. Die Demo zeigt, wie hoch der Falschalarm ohne die richtige Schwelle ist, was die richtige Schwelle kostet, und wie unsicher ein Spektrum bleibt (Konfidenzintervalle).

## Modell

Hintergrund mit Varianz 1: weiß, AR(1) mit φ = 0,5 / 0,8 / 0,95 (rotes Rauschen) oder ein steiler AR(2) (Resonanz bei 0,02, Radius 0,99, 64 dB Spannweite); optional eine Kosinuslinie mit zufälliger Phase.
Statistik: Rechteckfenster-Periodogramm an den inneren Fourier-Frequenzen (m = (n − 1)/2). Sieben Tests: Einzelfrequenz-Schwelle, Bonferroni, Šidák, Fishers g (alle gegen weißes Rauschen), Quotient durch ein angepasstes
AR(1) (Šidák), Quotient durch den lokalen Mittelwert der 10 Nachbarn (F-Schwelle) und die Vorfilterung mit einem AR(2)-Modell (Burg) mit Fisher-Test auf den Residuen. Konfidenzintervalle: χ² (Periodogramm, Welch)
und Jackknife über die Segmente. Alles selbst in numpy, auch die Quantile von χ², t und die Fisher-Verteilung; scipy dient nur als Orakel in den Tests.

## Methodik

n = 64 bis 1024 in der App, 32 bis 2048 in den Experimenten. Alarm heißt: der höchste Gipfel überschreitet die Schwelle für das Maximum bei Niveau 5 %. Experimente über feste Zufallsströme (Seeds ab 810000), jede Zahl ein
Mittel über 2 000 bis 20 000 Läufe (Standardfehler eines Anteils höchstens 1,2 Prozentpunkte).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` und `tests/test_experiments_golden.py`.

| Frage | Befund |
|---|---|
| Wie oft täuscht der höchste Gipfel bei reinem weißem Rauschen (Niveau 5 %)? | Mit der Einzelfrequenz-Schwelle in 58,1 % (n = 32), 98,9 % (n = 128) und 100 % (n = 512) der Läufe. Fishers g: 5,0 / 5,1 / 4,9 %; Bonferroni 2,0 / 3,7 / 4,2 % und Šidák 2,1 / 3,8 / 4,4 % sind mit geschätztem Mittelwert konservativ. |
| Wie gut ist die Erkennung (Linie in weißem Rauschen, Fishers g)? | Erkannt an der richtigen Stelle bei n = 128: 41 % (−9 dB), 84 % (−6 dB), 99 % (−3 dB), 100 % (0 dB); bei n = 512 schon 99 % bei −9 dB. |
| Was macht rotes Rauschen mit dem weißen Test (nur Rauschen)? | Falschalarm bei AR(1) mit φ = 0,5 / 0,8 / 0,95: 61,1 / 99,4 / 100 % (n = 128) und 87,7 / 100 / 100 % (n = 512). Der Test gegen das angepasste AR(1): 2,7 / 3,5 / 3,2 % und 3,9 / 4,3 / 4,7 %; der lokale Mittelwert mit F-Schwelle 5,0 / 5,8 / 21,9 % (n = 128) und 5,5 / 5,3 / 6,2 % (n = 512); ein naiver gleitender Median mit Šidák-Schwelle 28,7 / 32,2 / 57,0 % (n = 128). |
| Findet der Test gegen den Hintergrund die Linie (AR(1) φ = 0,8, n = 512, Linie gegen die Hintergrundvarianz)? | Angepasstes AR(1): 80 % (−18 dB), 98 % (−15 dB), 100 % ab −12 dB; lokaler Mittelwert 42 % (−18 dB), bis −6 dB höchstens 79 % (bei stärkerer Linie in Nachmessungen etwa 82 %), vermutlich weil die Linie ihre Nachbarbins anhebt; der weiße Test meldet zwar, aber an der falschen Stelle: richtig 0 % (−18 dB), 53 % (−9 dB). |
| Wie unsicher ist das Spektrum (95-%-Intervall, n = 512, AR(2)-Resonanz)? | Das Periodogramm (ν = 2) ist 21,6 dB breit, Überdeckung 94,0 bis 95,3 %. Welch (Hann, Länge 128, K = 7, ν = 13,4) ist 6,8 dB breit, Überdeckung 94,7 bis 95,3 % bei Radius 0,9 und **90,8 % am scharfen Gipfel** bei Radius 0,98. Jackknife über die 7 Segmente: 8,0 dB breit, 92,7 bis 94,3 % und 88,4 % am scharfen Gipfel. |
| Was hilft bei steilem Hintergrund (64 dB Spannweite, n = 256, nur Rauschen)? | Der weiße Test meldet in 100 %, der Quotient durch den **bekannten** Hintergrund (Rechteckfenster, Leckage) in 88,8 %, der Fisher-Test auf den AR(2)-Residuen in 5,0 %. |

## Ehrliche Grenzen

- **Heimvorteil:** Gaußsches stationäres Rauschen und eine bekannte Modellklasse (AR); bei anderem Rauschen wären die Zahlen schlechter.
- **Rechteckfenster:** Die Leckage-Aussage gilt für das Rechteckfenster; ein Hann-Fenster verschiebt sie, ändert aber die Verteilung der Statistik (Fishers g ist dann nicht mehr exakt).
- **Der lokale Mittelwert** verdeckt starke Linien (die Linie hebt ihre Nachbarn an); bei sehr steilen Spektren (φ = 0,95, n = 128) hält er das Niveau nicht (21,9 %).
- **Jackknife und Welch** unterdecken bei K = 7 Segmenten (überlappend, nicht unabhängig); in Nachmessungen verringerten mehr Segmente die Unterdeckung nicht. Am scharfen Gipfel streut Welch stärker, als ν = 13,4 annimmt, das Intervall ist dort zu schmal.
- **Vorfilterung:** gezeigt wird nur der Falschalarm; die Erkennungsleistung wurde nicht gemessen.
- **Whittle-Likelihood:** nicht gebaut (Gewinn im mittleren Fehler gegenüber Yule-Walker bei AR(1) klein, nur als Absatz in der Mathematik).
- **Quellen** (Fisher 1929, Torrence und Compo 1998, Mann und Lees 1996, Whittle 1953): nicht gelesen; Formeln per Simulation, Handrechnung und gegen scipy geprüft.

## Verifikation

- **Kern gegen Handrechnung:** n = 6, x = (1, 1, 0, 0, 0, 0): I = (1/2, 1/6), g = 3/4, p-Wert nach Fisher 1/2 (m = 2: g ist auf [1/2, 1] gleichverteilt).
- **Orakel (`tests/test_oracle_tests.py`, `tests/test_oracle_interval.py`, `tests/test_prewhiten.py`):** Fishers p-Wert gegen eine exakte Bruchrechnung (m bis 1023) und die Simulation, Quantile von χ² und t sowie die
  unvollständigen Gamma- und Betafunktionen gegen `scipy`, Welch gegen `scipy.signal.welch`, die F-Schwelle des lokalen Mittelwerts gegen `scipy.stats.f`, AR(1)-Dichte gegen den Generator, Burg gegen Handrechnung.
- **Generator** (`tests/test_generator.py`): wortgleich aus der ersten Demo.
- **Zahlen:** jede Zahl dieser README in `tests/test_claims.py`; Festwerte der Experimente in `tests/test_experiments_golden.py`; Verdikte in `tests/test_verdicts.py`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an den Grenzen), Auswahlfelder, Permalink, bedingte Regler, Footer, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `signal_generator.py` | Gemeinsamer Generator der Reihe (wortgleich) |
| `pkr_tests.py` | Periodogramm, Fishers g, Schwellen, AR(1)-Anpassung, lokaler Mittelwert, Median |
| `pkr_interval.py` | Gamma-, Beta-, χ²- und t-Quantile, Welch, χ²- und Jackknife-Intervall |
| `pkr_prewhiten.py` | Burg und Vorfilterung |
| `pkr_constants.py` | Regler-Grenzen, feste Annahmen, Experimente, Presets |
| `pkr_evaluation.py` | Analyse, Experimente, Verdikte |
| `pkr_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `pkr_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `pkr_pdf_export.py` | PDF der aktuellen Einstellung (erst nach Klick) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur Gegenproben) |
| `tests/` | Kern, Orakel, Zahlen, Verdikte, Presets, Randfälle, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Multitaper-F-Test, Bootstrap-Intervalle, Mehrfach-Linien:** nicht enthalten (Linienzahl: Folgedemos zu Unterraumverfahren; Signifikanz im Zeit-Frequenz-Bild: Spektrogramm und Wavelets).
- **Whittle-Schätzer und Bayessche Spektraldichte:** nur als Absatz.
- **Gemessene Daten:** alles ist simuliert, damit die Wahrheit bekannt ist.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Python 3.12 oder neuer. Die Experimente brauchen beim ersten Aufruf rund 15 Sekunden (danach zwischengespeichert).

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).

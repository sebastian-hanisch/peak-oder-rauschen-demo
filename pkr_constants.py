"""Konstanten: Regler-Grenzen, feste Annahmen, Hintergründe, Tests, Experimente, Farben und Presets (Kern: pkr_tests.py, pkr_interval.py, pkr_prewhiten.py)."""

N_CHOICES = (64, 128, 256, 512, 1024)
DEFAULT_N = 256
BACKGROUNDS = ("weiß", "AR(1) φ = 0,5", "AR(1) φ = 0,8", "AR(1) φ = 0,95", "steil (AR(2))")
BACKGROUND_POLES = {"weiß": (), "AR(1) φ = 0,5": ((0.0, 0.5),), "AR(1) φ = 0,8": ((0.0, 0.8),), "AR(1) φ = 0,95": ((0.0, 0.95),), "steil (AR(2))": ((0.02, 0.99),)}
DEFAULT_BACKGROUND = "weiß"
LINE_MODES = ("aus", "an")
DEFAULT_LINE_MODE = "aus"
LINE_MIN, LINE_MAX, LINE_STEP, DEFAULT_LINE = 0.05, 0.45, 0.005, 0.3          # Frequenz der Linie in Zyklen je Abtastwert
SNR_MIN, SNR_MAX, DEFAULT_SNR = -30, 10, -6                                    # Linienleistung A²/2 gegen die Varianz 1 des Hintergrunds, in dB
TESTS = ("Einzelfrequenz (weiß)", "Bonferroni (weiß)", "Šidák (weiß)", "Fisher g (weiß)", "AR(1) angepasst + Šidák", "Lokaler Mittelwert + F-Schwelle", "Vorfilterung AR(2) + Fisher g")
DEFAULT_TEST = "Fisher g (weiß)"
ALPHA_MIN, ALPHA_MAX, ALPHA_STEP, DEFAULT_ALPHA = 0.01, 0.10, 0.01, 0.05          # Niveau des Tests
BAND_MODES = ("aus", "an")
DEFAULT_BAND = "aus"
SEED_MAX, DEFAULT_SEED = 999999, 39
LOCAL_WIDTH = 11                                                               # Breite des lokalen Mittelwerts (10 Nachbarn)
PREWHITEN_ORDER = 2
FOUND_BINS = 1.5                                                               # Gipfel „an der richtigen Stelle“: Abstand zur Linie unter diesem Vielfachen von 1/n
BAND_LEVEL = 0.95

# Experimente (feste Ströme; Rauschen mit Varianz 1)
EXP_ALARM_NS = (32, 128, 512, 2048)
EXP_ALARM_RUNS = {32: 20000, 128: 20000, 512: 20000, 2048: 5000}
EXP_ALARM_ALPHA, EXP_ALARM_SEED = 0.05, 810000
EXP_POWER_NS, EXP_POWER_SNR, EXP_POWER_RUNS, EXP_POWER_F, EXP_POWER_SEED = (128, 512), (-9, -6, -3, 0), 2000, (0.1, 0.4), 820000
EXP_BG_NS, EXP_BG_PHIS, EXP_BG_RUNS, EXP_BG_SEED = (128, 512), (0.0, 0.5, 0.8, 0.95), 4000, 830000
EXP_BGP_N, EXP_BGP_PHI, EXP_BGP_SNR, EXP_BGP_RUNS, EXP_BGP_F, EXP_BGP_SEED = 512, 0.8, (-18, -15, -12, -9, -6), 3000, (0.15, 0.4), 840000
EXP_INT_N, EXP_INT_SEG, EXP_INT_RS, EXP_INT_RUNS, EXP_INT_SEED = 512, 128, (0.9, 0.98), 4000, 850000
EXP_INT_POSITIONS = (("Gipfel", 0.2), ("Flanke", 0.12), ("Tal", 0.4))
EXP_INT_RESONANCE = 0.2
EXP_PRE_N, EXP_PRE_RUNS, EXP_PRE_SEED = 256, 2000, 860000

COLOR_SPEC = "rgba(76,120,168,0.95)"
COLOR_TRUE = "#B03A2E"
COLOR_BG = "#4C9F70"
COLOR_THRESHOLD = "#E8833A"
COLOR_BAND = "rgba(76,120,168,0.20)"

PRESETS = {
    "Reines Rauschen: der höchste Gipfel täuscht": {"n": 256, "background": "weiß", "line_mode": "aus", "line": 0.3, "snr": -6, "test": "Einzelfrequenz (weiß)", "alpha": 0.05, "band": "aus", "seed": DEFAULT_SEED},
    "Dieselbe Reihe mit Fisher-Schwelle: kein Alarm": {"n": 256, "background": "weiß", "line_mode": "aus", "line": 0.3, "snr": -6, "test": "Fisher g (weiß)", "alpha": 0.05, "band": "aus", "seed": DEFAULT_SEED},
    "Rotes Rauschen: der weiße Test meldet immer": {"n": 256, "background": "AR(1) φ = 0,8", "line_mode": "aus", "line": 0.3, "snr": -6, "test": "Fisher g (weiß)", "alpha": 0.05, "band": "aus", "seed": DEFAULT_SEED},
    "Schwache Linie auf rotem Hintergrund": {"n": 512, "background": "AR(1) φ = 0,8", "line_mode": "an", "line": 0.3, "snr": -12, "test": "AR(1) angepasst + Šidák", "alpha": 0.05, "band": "aus", "seed": DEFAULT_SEED},
    "Steiler Hintergrund: Vorfilterung": {"n": 256, "background": "steil (AR(2))", "line_mode": "aus", "line": 0.3, "snr": -6, "test": "Vorfilterung AR(2) + Fisher g", "alpha": 0.05, "band": "aus", "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Reines Rauschen: der höchste Gipfel täuscht": "Nur weißes Rauschen, keine Linie: Der höchste Gipfel überschreitet die Schwelle für eine einzelne Frequenz, obwohl nichts da ist.",
    "Dieselbe Reihe mit Fisher-Schwelle: kein Alarm": "Dieselben Daten, aber die Schwelle für das Maximum über alle Frequenzen (Fishers g): kein Alarm.",
    "Rotes Rauschen: der weiße Test meldet immer": "AR(1)-Hintergrund ohne Linie: Der Test auf weißes Rauschen schlägt an, weil der Hintergrund bei tiefen Frequenzen von selbst hoch liegt.",
    "Schwache Linie auf rotem Hintergrund": "Eine Linie 12 dB unter der Hintergrundvarianz: Der Test gegen den angepassten AR(1)-Hintergrund findet sie an der richtigen Stelle, der weiße Test meldet nur den Hintergrund.",
    "Steiler Hintergrund: Vorfilterung": "Hintergrund mit 64 dB Spannweite, keine Linie: Der weiße Test meldet immer; erst die Vorfilterung mit einem AR(2)-Modell macht die Residuen weiß und der Test hält.",
}

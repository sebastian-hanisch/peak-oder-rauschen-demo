"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from pkr_evaluation import Analysis, de, outcome

_REPLACE = {"–": "-", "—": "-", "−": "-", "²": "2", "€": "EUR", "→": "->", "σ": "sigma", "Σ": "Summe", "±": "+-", "√": "Wurzel aus ", "×": "x", "é": "e", "„": '"', "“": '"', "φ": "phi", "α": "alpha",
            "Š": "S", "š": "s", "ν": "nu", "χ": "chi"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Peak oder Rauschen: Einstellung und Ergebnis"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    line = f"Linie bei {de(s.line, 3)} ({s.snr} dB gegen die Hintergrundvarianz)" if s.has_line else "keine Linie"
    pdf.multi_cell(0, 5, _s(f"n = {s.n} Werte, Hintergrund: {s.background}, {line}, Niveau {de(s.alpha, 2)}, gewählter Test: {s.test}, Seed {s.seed}. Alles simuliert. Frequenzen m = {a.m}."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    for name, r in a.results.items():
        pdf.multi_cell(0, 5, _s(f"{name}: Gipfel bei {de(r.peak_hz, 3)}, Quotient {de(r.stat, 1)}, Schwelle {de(r.threshold, 1)}, p = {de(r.p, 3)}, {'Alarm' if r.alarm else 'kein Alarm'} ({outcome(r, s.has_line)})."),
                       new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: peak-oder-rauschen-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())

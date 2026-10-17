"""Plotly-Figuren. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import pkr_constants as C
from pkr_evaluation import Analysis

EPS = 1e-30
METHOD_COLORS = {"single": "#B03A2E", "bonferroni": "#E8833A", "sidak": "#bcbd22", "fisher": "rgba(76,120,168,0.95)", "white": "#7f7f7f", "known": "#7B5EA7",
                 "fitted": "#4C9F70", "local": "#E8833A", "median": "#B03A2E"}
METHOD_NAMES = {"single": "Einzelfrequenz-Schwelle", "bonferroni": "Bonferroni", "sidak": "Šidák", "fisher": "Fishers g", "white": "weißer Test (Fisher)", "known": "bekannter Hintergrund",
                "fitted": "AR(1) angepasst", "local": "lokaler Mittelwert + F", "median": "gleitender Median (naiv)"}


def _db(v):
    return 10.0 * np.log10(np.maximum(np.asarray(v, dtype=float), EPS))


def _fixed(fig: go.Figure, height: int = 360, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_spectrum(a: Analysis) -> go.Figure:
    """Periodogramm in dB mit der wahren Dichte des Hintergrunds, dem angepassten AR(1), dem lokalen Mittelwert und, falls gewählt, dem χ²-Band (zwei Freiheitsgrade)."""
    s = a.settings
    fig = go.Figure()
    if a.band is not None:
        fig.add_trace(go.Scatter(x=a.f, y=_db(a.band[1]), mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=a.f, y=_db(a.band[0]), mode="lines", line=dict(width=0), fill="tonexty", fillcolor=C.COLOR_BAND, name="95-%-Intervall (χ², ν = 2)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.f, y=_db(a.e), mode="lines", line=dict(color=C.COLOR_SPEC, width=1.4), name="Periodogramm", hovertemplate="f = %{x:.4f}: %{y:.1f} dB<extra></extra>"))
    fig.add_trace(go.Scatter(x=a.f, y=_db(a.true_density), mode="lines", line=dict(color=C.COLOR_TRUE, width=2, dash="dash"), name="wahre Dichte des Hintergrunds", hovertemplate="f = %{x:.4f}: %{y:.1f} dB<extra></extra>"))
    fig.add_trace(go.Scatter(x=a.f, y=_db(a.fitted_density), mode="lines", line=dict(color=C.COLOR_BG, width=1.6, dash="dot"), name="angepasstes AR(1)", hovertemplate="f = %{x:.4f}: %{y:.1f} dB<extra></extra>"))
    fig.add_trace(go.Scatter(x=a.f, y=_db(a.local_background), mode="lines", line=dict(color=C.COLOR_THRESHOLD, width=1.2), name="lokaler Mittelwert (11 Bins)", visible="legendonly",
                             hovertemplate="f = %{x:.4f}: %{y:.1f} dB<extra></extra>"))
    if s.has_line:
        fig.add_vline(x=s.line, line=dict(color=C.COLOR_TRUE, width=1.2, dash="dashdot"))
    fig.update_xaxes(title_text="Frequenz (Zyklen je Abtastwert)", range=[0, 0.5])
    fig.update_yaxes(title_text="Leistungsdichte (dB)")
    return _fixed(fig, height=400)


def build_test(a: Analysis) -> go.Figure:
    """Teststatistik des gewählten Tests: Quotient Periodogramm / Hintergrund je Frequenz gegen die Schwelle für das Maximum."""
    s = a.settings
    r = a.results[s.test]
    fig = go.Figure(go.Scatter(x=r.freqs, y=r.ratio, mode="lines", line=dict(color=C.COLOR_SPEC, width=1.4), name="Quotient", hovertemplate="f = %{x:.4f}: %{y:.2f}<extra></extra>"))
    fig.add_hline(y=r.threshold, line=dict(color=C.COLOR_THRESHOLD, width=1.8, dash="dash"), annotation_text=f"Schwelle {r.threshold:.1f}".replace(".", ","), annotation_position="top left")
    fig.add_trace(go.Scatter(x=[r.peak_hz], y=[r.stat], mode="markers", marker=dict(size=11, color=C.COLOR_TRUE if r.alarm else "gray", symbol="diamond"), name="höchster Gipfel",
                             hovertemplate="f = %{x:.4f}: %{y:.2f}<extra></extra>"))
    if s.has_line:
        fig.add_vline(x=s.line, line=dict(color=C.COLOR_TRUE, width=1.2, dash="dashdot"))
    fig.update_xaxes(title_text="Frequenz (Zyklen je Abtastwert)", range=[0, 0.5])
    fig.update_yaxes(title_text="Quotient Periodogramm / Hintergrund", rangemode="tozero")
    fig.update_layout(showlegend=False)
    return _fixed(fig, height=320)


def build_alarm(rows) -> go.Figure:
    labels = [f"n = {r['n']}" for r in rows]
    fig = go.Figure()
    for key in ("single", "bonferroni", "sidak", "fisher"):
        fig.add_trace(go.Bar(x=labels, y=[100 * r[key] for r in rows], name=METHOD_NAMES[key], marker_color=METHOD_COLORS[key], hovertemplate="%{x}: %{y:.1f} %<extra></extra>"))
    fig.add_hline(y=5.0, line=dict(color="black", width=1, dash="dot"), annotation_text="Ziel 5 %", annotation_position="top right")
    fig.update_yaxes(title_text="Falschalarm (% der Läufe)", range=[0, 105])
    fig.update_layout(barmode="group")
    return _fixed(fig, height=380)


def build_power(rows) -> go.Figure:
    fig = go.Figure()
    for n in sorted({r["n"] for r in rows}):
        sel = [r for r in rows if r["n"] == n]
        fig.add_trace(go.Scatter(x=[r["snr"] for r in sel], y=[100 * r["right_place"] for r in sel], mode="lines+markers", name=f"n = {n}", hovertemplate=f"n = {n}, %{{x}} dB: %{{y:.0f}} %<extra></extra>"))
    fig.update_xaxes(title_text="Linie gegen die Rauschvarianz (dB)")
    fig.update_yaxes(title_text="erkannt an der richtigen Stelle (%)", range=[-3, 103])
    return _fixed(fig, height=340)


def build_background_alarm(rows) -> go.Figure:
    labels = [f"n = {r['n']}, φ = {r['phi']:g}".replace(".", ",") for r in rows]
    fig = go.Figure()
    for key in ("white", "known", "fitted", "local", "median"):
        fig.add_trace(go.Bar(x=labels, y=[100 * r[key] for r in rows], name=METHOD_NAMES[key], marker_color=METHOD_COLORS[key], hovertemplate="%{x}: %{y:.1f} %<extra></extra>"))
    fig.add_hline(y=5.0, line=dict(color="black", width=1, dash="dot"))
    fig.update_yaxes(title_text="Falschalarm (% der Läufe)", range=[0, 105])
    fig.update_layout(barmode="group")
    return _fixed(fig, height=420)


def build_background_power(rows) -> go.Figure:
    fig = go.Figure()
    for key in ("white", "fitted", "local"):
        fig.add_trace(go.Scatter(x=[r["snr"] for r in rows], y=[100 * r[key] for r in rows], mode="lines+markers", name=METHOD_NAMES[key], line=dict(color=METHOD_COLORS[key]),
                                 hovertemplate=METHOD_NAMES[key] + ", %{x} dB: %{y:.0f} %<extra></extra>"))
    fig.update_xaxes(title_text="Linie gegen die Varianz des Hintergrunds (dB)")
    fig.update_yaxes(title_text="erkannt an der richtigen Stelle (%)", range=[-3, 103])
    return _fixed(fig, height=340)


def build_interval(rows) -> go.Figure:
    labels = [f"{pos}, Radius {r:g}".replace(".", ",") for r in dict.fromkeys(x["r"] for x in rows) for pos in dict.fromkeys(x["position"] for x in rows)]
    fig = go.Figure()
    for method in dict.fromkeys(x["method"] for x in rows):
        sel = [x for x in rows if x["method"] == method]
        fig.add_trace(go.Bar(x=labels, y=[100 * x["coverage"] for x in sel], name=method, hovertemplate="%{x}: %{y:.1f} %<extra></extra>"))
    fig.add_hline(y=95.0, line=dict(color="black", width=1, dash="dot"), annotation_text="Soll 95 %", annotation_position="top right")
    fig.update_yaxes(title_text="Überdeckung der wahren Dichte (%)", range=[80, 100])
    fig.update_layout(barmode="group")
    return _fixed(fig, height=400)


def build_prewhiten(res) -> go.Figure:
    labels = ["weißer Test, Rohsignal", "bekannter Hintergrund (Rechteckfenster)", "Vorfilterung AR(2) + Fisher"]
    vals = [100 * res["white"], 100 * res["known"], 100 * res["prewhitened"]]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=[METHOD_COLORS["white"], METHOD_COLORS["known"], METHOD_COLORS["fitted"]], hovertemplate="%{x}: %{y:.1f} %<extra></extra>"))
    fig.add_hline(y=5.0, line=dict(color="black", width=1, dash="dot"), annotation_text="Ziel 5 %", annotation_position="top right")
    fig.update_yaxes(title_text="Falschalarm (% der Läufe)", range=[0, 105])
    fig.update_layout(showlegend=False)
    return _fixed(fig, height=340)

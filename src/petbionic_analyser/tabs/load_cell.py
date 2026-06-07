"""Tab – Célula de Carga."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout
from PyQt6.QtCore import Qt

from ..config import C, _SCATTER_PT
from ..widgets import ToggleBtn, _FitBtn


class LoadCellTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        tbar = QHBoxLayout()
        self._btn_kg  = ToggleBtn("Weight (kg)", C["kg"])
        self._btn_raw = ToggleBtn("Raw ADC",   C["raw"])
        self._btn_fit = _FitBtn()
        for b in [self._btn_kg, self._btn_raw, self._btn_fit]:
            tbar.addWidget(b)
        tbar.addStretch()
        lay.addLayout(tbar)

        self._fig = Figure(figsize=(11, 4))
        self._fig.subplots_adjust(left=0.07, right=0.93, top=0.94, bottom=0.11)
        self._ax_kg  = self._fig.add_subplot(111)
        self._ax_raw = self._ax_kg.twinx()
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._toolbar.setFixedHeight(30)
        lay.addWidget(self._toolbar)
        lay.addWidget(self._canvas)

        self._sc_kg = self._sc_raw = None
        self._ln_kg = self._ln_raw = None   # smooth lines
        self._span: SpanSelector | None = None

        self._btn_kg.toggled.connect(lambda v: self._set_vis_lc(self._sc_kg, self._ln_kg, v))
        self._btn_raw.toggled.connect(lambda v: self._set_vis_lc(self._sc_raw, self._ln_raw, v))
        self._btn_fit.toggled.connect(self._on_fit)

    def _set_vis_lc(self, sc, ln, v: bool):
        show_line = self._btn_fit.isChecked()
        if sc: sc.set_visible(v and not show_line)
        if ln: ln.set_visible(v and show_line)
        self._canvas.draw_idle()

    def load(self, df: pd.DataFrame):
        self._ax_kg.cla(); self._ax_raw.cla()
        x = (df["sample_us"] - df["sample_us"].iloc[0]) / 1e6
        w = max(5, len(df) // 60)

        if "load_cell_est_kg" in df.columns:
            y = df["load_cell_est_kg"].to_numpy(float)
            show_line = self._btn_fit.isChecked()
            self._sc_kg = self._ax_kg.scatter(
                x, y, s=_SCATTER_PT, c=C["kg"], label="Weight (kg)",
                alpha=0.85, linewidths=0)
            self._sc_kg.set_visible(self._btn_kg.isChecked() and not show_line)
            ys = pd.Series(y).rolling(w, center=True, min_periods=1).mean().to_numpy()
            self._ln_kg, = self._ax_kg.plot(
                x, ys, color=C["kg"], linewidth=1.4, alpha=0.9, label="Weight (kg)")
            self._ln_kg.set_visible(self._btn_kg.isChecked() and show_line)

        if "load_cell_raw" in df.columns:
            y = df["load_cell_raw"].to_numpy(float)
            show_line = self._btn_fit.isChecked()
            self._sc_raw = self._ax_raw.scatter(
                x, y, s=_SCATTER_PT, c=C["raw"], label="Raw ADC",
                alpha=0.65, linewidths=0)
            self._sc_raw.set_visible(self._btn_raw.isChecked() and not show_line)
            ys = pd.Series(y).rolling(w, center=True, min_periods=1).mean().to_numpy()
            self._ln_raw, = self._ax_raw.plot(
                x, ys, color=C["raw"], linewidth=1.4, alpha=0.9, label="Raw ADC")
            self._ln_raw.set_visible(self._btn_raw.isChecked() and show_line)

        self._ax_kg.set_xlabel("Time (s)", fontsize=10)
        self._ax_kg.set_ylabel("Estimated weight (kg)", color=C["kg"], fontsize=10)
        self._ax_raw.set_ylabel("Raw ADC", color=C["raw"], fontsize=10)
        self._ax_kg.tick_params(axis="y", labelcolor=C["kg"])
        self._ax_raw.tick_params(axis="y", labelcolor=C["raw"])
        self._ax_kg.grid(True, linestyle="--", alpha=0.3)
        self._ax_kg.set_title(
            "Load Cell  —  drag = zoom · double-click = reset", fontsize=10)

        self._span = SpanSelector(
            self._ax_kg, self._on_span, "horizontal",
            useblit=True, props=dict(alpha=0.18, facecolor="#1f77b4"))
        self._canvas.mpl_connect(
            "button_press_event", lambda e: self._reset_zoom() if e.dblclick else None)
        self._canvas.draw_idle()

    def _on_fit(self, show_line: bool):
        self._btn_fit.setText("Points" if show_line else "Line")
        for sc, ln, btn in [
            (self._sc_kg,  self._ln_kg,  self._btn_kg),
            (self._sc_raw, self._ln_raw, self._btn_raw),
        ]:
            v = btn.isChecked()
            if sc: sc.set_visible(v and not show_line)
            if ln: ln.set_visible(v and show_line)
        self._canvas.draw_idle()

    def _visible_y(self, xmin, xmax):
        """Collect Y values in [xmin,xmax] from whichever artists are currently visible."""
        show_line = self._btn_fit.isChecked()
        yall: list[float] = []
        pairs = [(self._sc_kg, self._ln_kg, self._ax_kg),
                 (self._sc_raw, self._ln_raw, self._ax_raw)]
        for sc, ln, ax in pairs:
            if show_line and ln and ln.get_visible():
                xd = np.asarray(ln.get_xdata()); yd = np.asarray(ln.get_ydata())
                m = (xd >= xmin) & (xd <= xmax)
                if m.any(): yall.append((ax, yd[m]))
            elif sc and sc.get_visible():
                pts = sc.get_offsets()
                m = (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax)
                if m.any(): yall.append((ax, pts[m, 1]))
        return yall

    def _on_span(self, xmin: float, xmax: float):
        if xmax - xmin < 0.05 or self._toolbar.mode:
            return
        self._ax_kg.set_xlim(xmin, xmax)
        for ax, yv in self._visible_y(xmin, xmax):
            span = float(yv.max() - yv.min())
            pad = span * 0.08 if span > 0 else max(abs(float(yv.mean())) * 0.05, 0.5)
            ax.set_ylim(float(yv.min()) - pad, float(yv.max()) + pad)
        self._canvas.draw_idle()

    def _reset_zoom(self):
        self._ax_kg.autoscale(); self._ax_raw.autoscale()
        self._canvas.draw_idle()



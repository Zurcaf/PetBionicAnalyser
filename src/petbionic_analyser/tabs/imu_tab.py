"""Tab – IMU genérico (acelerómetro | giroscópio | magnetómetro | orientação)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt

from ..config import C, _SCATTER_PT
from ..widgets import ToggleBtn, _FitBtn
from ..gait import _smooth


class ImuTab(QWidget):
    def __init__(self,
                 cols: tuple[str, str, str],
                 labels: tuple[str, str, str],
                 title: str,
                 ylabel: str,
                 show_magnitude: bool = True,
                 unwrap: bool = False,
                 parent=None):
        super().__init__(parent)
        self._cols           = cols
        self._labels         = labels
        self._title          = title
        self._ylabel         = ylabel
        self._show_magnitude = show_magnitude
        self._unwrap         = unwrap

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        tbar = QHBoxLayout()
        self._btns: list[ToggleBtn] = []
        for lbl, key in zip(labels, ("x", "y", "z")):
            btn = ToggleBtn(lbl, C[key]); self._btns.append(btn); tbar.addWidget(btn)

        self._btn_mag: ToggleBtn | None = None
        if show_magnitude:
            self._btn_mag = ToggleBtn("Magnitude", "#ff7f0e")
            tbar.addWidget(self._btn_mag)

        self._btn_fit = _FitBtn()
        tbar.addWidget(self._btn_fit)
        tbar.addStretch()
        lay.addLayout(tbar)

        self._fig = Figure(figsize=(11, 4))
        self._fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.11)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._toolbar.setFixedHeight(30)
        lay.addWidget(self._toolbar)
        lay.addWidget(self._canvas)

        self._scs:   list = [None, None, None]
        self._lines: list = [None, None, None]
        self._sc_mag = self._ln_mag = None
        self._span: SpanSelector | None = None

        for i, btn in enumerate(self._btns):
            btn.toggled.connect(lambda v, idx=i: self._set_vis(idx, v))
        if self._btn_mag:
            self._btn_mag.toggled.connect(self._on_mag_toggle)
        self._btn_fit.toggled.connect(self._on_fit)

    # ── visibility helpers ────────────────────────────────────────────────────

    def _set_vis(self, idx: int, v: bool):
        show_line = self._btn_fit.isChecked()
        if self._scs[idx]:   self._scs[idx].set_visible(v and not show_line)
        if self._lines[idx]: self._lines[idx].set_visible(v and show_line)
        self._canvas.draw_idle()

    def _on_mag_toggle(self, v: bool):
        show_line = self._btn_fit.isChecked()
        if self._sc_mag: self._sc_mag.set_visible(v and not show_line)
        if self._ln_mag: self._ln_mag.set_visible(v and show_line)
        self._canvas.draw_idle()

    def _on_fit(self, show_line: bool):
        self._btn_fit.setText("Points" if show_line else "Line")
        for i, btn in enumerate(self._btns):
            v = btn.isChecked()
            if self._scs[i]:   self._scs[i].set_visible(v and not show_line)
            if self._lines[i]: self._lines[i].set_visible(v and show_line)
        if self._btn_mag:
            v = self._btn_mag.isChecked()
            if self._sc_mag: self._sc_mag.set_visible(v and not show_line)
            if self._ln_mag: self._ln_mag.set_visible(v and show_line)
        self._canvas.draw_idle()

    # ── load ──────────────────────────────────────────────────────────────────

    def load(self, df: pd.DataFrame):
        self._ax.cla()
        x   = (df["sample_us"] - df["sample_us"].iloc[0]) / 1e6
        w   = max(5, len(df) // 60)
        clr = [C["x"], C["y"], C["z"]]
        show_line = self._btn_fit.isChecked()

        for i, (col, lbl, col_c) in enumerate(zip(self._cols, self._labels, clr)):
            if col in df.columns:
                y = df[col].to_numpy(float)
                if self._unwrap:
                    y = np.degrees(np.unwrap(np.radians(y)))
                sc = self._ax.scatter(x, y, s=_SCATTER_PT, c=col_c,
                                      label=lbl, alpha=0.75, linewidths=0)
                sc.set_visible(self._btns[i].isChecked() and not show_line)
                self._scs[i] = sc

                ln, = self._ax.plot(x, _smooth(y, w), color=col_c,
                                    linewidth=1.4, alpha=0.9)
                ln.set_visible(self._btns[i].isChecked() and show_line)
                self._lines[i] = ln

        # módulo (sqrt(x²+y²+z²))
        if self._show_magnitude and all(c in df.columns for c in self._cols):
            mag = np.sqrt(sum(df[c].to_numpy(float)**2 for c in self._cols))
            sc  = self._ax.scatter(x, mag, s=_SCATTER_PT, c="#ff7f0e",
                                   label="Magnitude", alpha=0.75, linewidths=0)
            sc.set_visible(self._btn_mag.isChecked() and not show_line)
            self._sc_mag = sc

            ln, = self._ax.plot(x, _smooth(mag, w), color="#ff7f0e",
                                linewidth=1.4, alpha=0.9)
            ln.set_visible(self._btn_mag.isChecked() and show_line)
            self._ln_mag = ln

        self._ax.set_xlabel("Time (s)", fontsize=10)
        self._ax.set_ylabel(self._ylabel, fontsize=10)
        self._ax.legend(loc="upper right", fontsize=9, markerscale=3)
        self._ax.grid(True, linestyle="--", alpha=0.3)
        self._ax.set_title(
            f"{self._title}  —  arrastar = zoom · duplo-clique = reset", fontsize=10)

        self._span = SpanSelector(
            self._ax, self._on_span, "horizontal",
            useblit=True, props=dict(alpha=0.18, facecolor="#1f77b4"))
        self._canvas.mpl_connect(
            "button_press_event", lambda e: self._reset_zoom() if e.dblclick else None)
        self._canvas.draw_idle()

    # ── zoom ──────────────────────────────────────────────────────────────────

    def _on_span(self, xmin: float, xmax: float):
        if xmax - xmin < 0.05 or self._toolbar.mode:
            return
        self._ax.set_xlim(xmin, xmax)
        show_line = self._btn_fit.isChecked()
        yall: list[float] = []

        def _collect(sc, ln):
            if show_line and ln and ln.get_visible():
                xd = np.asarray(ln.get_xdata()); yd = np.asarray(ln.get_ydata())
                m = (xd >= xmin) & (xd <= xmax)
                if m.any(): yall.extend(yd[m].tolist())
            elif sc and sc.get_visible():
                pts = sc.get_offsets()
                m   = (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax)
                if m.any(): yall.extend(pts[m, 1].tolist())

        for sc, ln in zip(self._scs, self._lines):
            _collect(sc, ln)
        _collect(self._sc_mag, self._ln_mag)

        if yall:
            mn, mx = min(yall), max(yall)
            pad = (mx - mn) * 0.08 if mx != mn else max(abs(mn) * 0.05, 1.0)
            self._ax.set_ylim(mn - pad, mx + pad)
        self._canvas.draw_idle()

    def _reset_zoom(self):
        self._ax.autoscale(); self._canvas.draw_idle()



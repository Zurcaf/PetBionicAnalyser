"""Tab – Features de Marcha."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

from ..config import C
from ..gait import compute_gait_features


class GaitFeaturesTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        self._summary = QLabel("Load a file to compute the gait features.")
        self._summary.setWordWrap(True)
        self._summary.setTextFormat(Qt.TextFormat.RichText)
        self._summary.setStyleSheet(
            "font-size:11px; padding:6px 8px; background:#fafafa; "
            "border:1px solid #e0e0e0; border-radius:4px;")
        lay.addWidget(self._summary)

        self._fig = Figure(figsize=(11, 8.0))
        self._fig.subplots_adjust(left=0.07, right=0.97, top=0.96,
                                  bottom=0.06, hspace=0.55, wspace=0.22)
        gs = self._fig.add_gridspec(3, 2)
        self._ax_main = self._fig.add_subplot(gs[0, :])
        self._ax_pvf = self._fig.add_subplot(gs[1, 0])
        self._ax_str = self._fig.add_subplot(gs[1, 1])
        self._ax_imu = self._fig.add_subplot(gs[2, :])
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._toolbar.setFixedHeight(30)
        lay.addWidget(self._toolbar)
        lay.addWidget(self._canvas)

    @staticmethod
    def _fmt(v, unit="", nd=2):
        return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}{unit}"

    def _build_summary(self, F: dict) -> str:
        f = self._fmt
        col = "#1f77b4"
        head = (f"<b>{F['n_strides']} strides</b> detected in "
                f"{f(F['dur'], ' s', 1)}")

        def grp(title, rows):
            body = "".join(
                f"<tr><td style='padding:1px 10px 1px 0'>{n}</td>"
                f"<td style='padding:1px 0;color:{col}'><b>{v}</b></td>"
                f"<td style='padding:1px 0 1px 12px;color:#888'>{d}</td></tr>"
                for n, v, d in rows)
            return (f"<p style='margin:4px 0 1px'><b>{title}</b></p>"
                    f"<table style='border-collapse:collapse'>{body}</table>")

        forca = grp("Force", [
            ("Mean PVF", f(F['PVF_mean'], " kg"), "how much weight on the prosthesis"),
            ("Mean impulse", f(F['impulse_mean'], " kg·s"), "total load per step"),
            ("Loading rate", f(F['lr_mean'], " kg/s"), "loading confidence"),
            ("Weight-bearing", f(100 * F['wbr'], " %", 0), "strides with full load"),
        ])
        temp = grp("Temporal", [
            ("Stride time", f(F['stride_mean'], " s"), "gait pace"),
            ("Stance / swing", f"{f(F['stance_mean'],' s')} / {f(F['swing_mean'],' s')}", "stance vs swing"),
            ("Duty factor", f(F['duty_mean']), "fraction of cycle under load"),
            ("Cadence", f(F['cadence'], " strides/s"), "locomotor rhythm"),
        ])
        var = grp("Variability", [
            ("CV stride time", f(100 * F['cv_stride'], " %", 1), "↓ = more consistent"),
            ("CV of PVF", f(100 * F['cv_pvf'], " %", 1), "↓ = consistent load"),
            ("Regularity", f(F['reg']), "↑ = more periodic [0–1]"),
        ])
        cells = [("33%", forca), ("34%", temp), ("33%", var)]
        if F.get("has_imu"):
            kin = grp("Kinematic (IMU)", [
                ("Pitch ROM", f(F.get('pitch_rom_mean'), " °", 1), "angular excursion/stride"),
                ("Roll ROM", f(F.get('roll_rom_mean'), " °", 1), "lateral sway"),
                ("Peak angular rate", f(F.get('gpeak_mean'), "", 0), "swing vigour (counts)"),
                ("Smoothness (SPARC)", f(F.get('sparc')), "↑ (less neg.) = smoother"),
            ])
            cells = [("25%", forca), ("25%", temp), ("25%", var), ("25%", kin)]
        tds = "".join(f"<td width='{wd}'>{c}</td>" for wd, c in cells)
        return (f"<div style='line-height:1.35'>{head}"
                f"<table width='100%'><tr style='vertical-align:top'>{tds}</tr></table></div>")

    def load(self, df: pd.DataFrame):
        for ax in (self._ax_main, self._ax_pvf, self._ax_str, self._ax_imu):
            ax.cla()
        self._df = df
        F = compute_gait_features(df)

        if F is None:
            self._summary.setText("No force column — gait features unavailable.")
            self._canvas.draw_idle()
            return

        t, fs = F["t"], F["fs"]
        self._ax_main.plot(t, fs, color=C["kg"], lw=1.1, alpha=0.9, label="Force (kg)")
        self._ax_main.axhline(F["contact_thr"], color="#999", ls="--", lw=1,
                              label="contact threshold")
        first = True
        for s in F["stances"]:
            self._ax_main.axvspan(s["t0"], s["t1"], color="#4daf4a", alpha=0.12,
                                  label="stance" if first else None)
            first = False
        if F["ok"]:
            self._ax_main.scatter(F["peaks_t"], F["pvf"], s=26, color="#e41a1c",
                                  zorder=5, label="PVF (peak)")
            self._ax_main.axhline(F["PVF_mean"], color="#e41a1c", ls=":", lw=1,
                                  label="mean PVF")
        self._ax_main.set_xlabel("Time (s)", fontsize=9)
        self._ax_main.set_ylabel("Force (kg)", fontsize=9)
        self._ax_main.set_title("Stride segmentation and force", fontsize=10)
        self._ax_main.grid(True, ls="--", alpha=0.3)
        self._ax_main.legend(fontsize=7, ncol=5, loc="upper right", framealpha=0.85)

        if not F["ok"]:
            self._ax_pvf.text(0.5, 0.5, "insufficient strides",
                              ha="center", va="center", fontsize=9, color="#888")
            self._ax_str.axis("off")
            self._summary.setText(
                f"Only {len(F['stances'])} stride(s) detected — "
                "signal too short or irregular for statistics.")
            self._canvas.draw_idle()
            return

        def _bars(ax, vals, color, title, cv):
            idx = np.arange(1, len(vals) + 1)
            ax.bar(idx, vals, color=color, alpha=0.75, width=0.8)
            m, sd = float(np.mean(vals)), float(np.std(vals))
            ax.axhspan(m - sd, m + sd, color=color, alpha=0.12)
            ax.axhline(m, color=color, ls="--", lw=1)
            ax.set_title(f"{title}  (CV = {100*cv:.1f} %)", fontsize=9)
            ax.set_xlabel("stride", fontsize=8)
            ax.grid(True, axis="y", ls="--", alpha=0.3)

        _bars(self._ax_pvf, F["pvf"], "#e41a1c", "PVF per stride", F["cv_pvf"])
        self._ax_pvf.set_ylabel("kg", fontsize=8)
        _bars(self._ax_str, F["stride_t"], "#1f77b4", "Stride time", F["cv_stride"])
        self._ax_str.set_ylabel("s", fontsize=8)

        self._draw_imu(F, df)
        self._summary.setText(self._build_summary(F))
        self._canvas.draw_idle()

    def _draw_imu(self, F: dict, df: pd.DataFrame):
        ax = self._ax_imu
        if not F.get("has_imu") or "roll_deg" not in df.columns:
            ax.text(0.5, 0.5, "No IMU data (roll/pitch) in this file",
                    ha="center", va="center", fontsize=9, color="#888")
            ax.axis("off")
            return
        t = F["t"]
        for col, c, lbl in [("pitch_deg", C["pitch"], "Pitch"),
                            ("roll_deg", C["roll"], "Roll")]:
            ang = np.degrees(np.unwrap(np.radians(df[col].to_numpy(float))))
            ax.plot(t, ang, color=c, lw=1.0, alpha=0.9, label=lbl)
        first = True
        for s in F["stances"]:
            ax.axvspan(s["t0"], s["t1"], color="#4daf4a", alpha=0.10,
                       label="stance" if first else None)
            first = False
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Angle (°)", fontsize=9)
        sp = self._fmt(F.get("sparc"))
        rp = self._fmt(F.get("pitch_rom_mean"), " °", 1)
        ax.set_title(f"IMU kinematics — orientation per stride  "
                     f"(pitch ROM ≈ {rp} · smoothness SPARC = {sp})", fontsize=10)
        ax.grid(True, ls="--", alpha=0.3)
        ax.legend(fontsize=7, ncol=3, loc="upper right", framealpha=0.85)



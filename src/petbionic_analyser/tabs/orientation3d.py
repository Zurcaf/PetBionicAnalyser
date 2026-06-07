"""Tab – Orientação 3D (Main)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QSplitter,
)
from PyQt6.QtCore import Qt, QTimer

from ..config import C
from ..widgets import ToggleBtn, _FitBtn
from ..gait import _smooth
from ..viz3d import draw_prosthetic
from ..imu import _turns_to_R, apply_imu_correction
from ..persistence import _load_model_turns


class Orientation3DTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        import time as _time_mod
        self._time_mod = _time_mod

        # orientação base do modelo 3D (carregada cedo p/ a UI a poder usar)
        self._model_turns = _load_model_turns()
        self._R_base = _turns_to_R(self._model_turns)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(3)

        # ── toggles: RPY + Kg + Linha/Pontos ──────────────────────────────────
        tbar = QHBoxLayout()
        self._btn_roll  = ToggleBtn("Roll",  C["roll"])
        self._btn_pitch = ToggleBtn("Pitch", C["pitch"])
        self._btn_yaw   = ToggleBtn("Yaw",   C["yaw"])
        self._btn_kg    = ToggleBtn("Kg",    C["kg"])
        self._btn_fit   = _FitBtn()
        hint = QLabel("   drag on the plots or slider  ·  unwrapped angles")
        hint.setStyleSheet("color:#888;font-size:10px;")
        for b in [self._btn_roll, self._btn_pitch, self._btn_yaw,
                  QLabel(" | "), self._btn_kg,
                  QLabel(" | "), self._btn_fit]:
            tbar.addWidget(b)
        tbar.addWidget(hint)
        tbar.addStretch()
        lay.addLayout(tbar)

        # (A orientação-base do modelo 3D e os respetivos botões passaram para o
        #  tab de Calibração; aqui o Main só mostra o modelo único corrigido.)

        # ── splitter vertical: RPY | Kg | 3D ─────────────────────────────────
        vsplit = QSplitter(Qt.Orientation.Vertical)
        lay.addWidget(vsplit, stretch=1)

        def _make_plot_widget(fig, canvas):
            w = QWidget(); wl = QVBoxLayout(w)
            wl.setContentsMargins(0, 0, 0, 0)
            wl.addWidget(canvas); return w

        # Gráfico RPY (sem toolbar, sem SpanSelector — drag = scrub)
        self._fig_rpy = Figure(figsize=(11, 2.5))
        self._fig_rpy.subplots_adjust(left=0.07, right=0.97, top=0.87, bottom=0.22)
        self._ax_rpy  = self._fig_rpy.add_subplot(111)
        self._cv_rpy  = FigureCanvasQTAgg(self._fig_rpy)
        vsplit.addWidget(_make_plot_widget(self._fig_rpy, self._cv_rpy))

        # Gráfico Kg (sincronizado — drag = scrub)
        self._fig_kg2 = Figure(figsize=(11, 2.0))
        self._fig_kg2.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.25)
        self._ax_kg2  = self._fig_kg2.add_subplot(111)
        self._cv_kg2  = FigureCanvasQTAgg(self._fig_kg2)
        vsplit.addWidget(_make_plot_widget(self._fig_kg2, self._cv_kg2))

        # Modelo 3D — único: orientação real (corrigida pelo R; ou firmware se sem R)
        self._fig_3d = Figure(figsize=(11, 4.2))
        self._ax_3d  = self._fig_3d.add_subplot(111, projection="3d")
        self._cv_3d  = FigureCanvasQTAgg(self._fig_3d)
        # mantém o cubo centrado e sem cortes ao redimensionar a janela
        self._cv_3d.mpl_connect("resize_event", lambda _e: self._center_3d())
        vsplit.addWidget(_make_plot_widget(self._fig_3d, self._cv_3d))

        vsplit.setSizes([200, 150, 380])

        # ── slider scrubber ───────────────────────────────────────────────────
        s_row = QHBoxLayout()
        self._lbl_s0  = QLabel("0 s");  self._lbl_s0.setStyleSheet("font-size:10px;color:#888;")
        self._lbl_s1  = QLabel("0 s");  self._lbl_s1.setStyleSheet("font-size:10px;color:#888;")
        self._slider  = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0); self._slider.setMaximum(0)
        self._slider.setTracking(True)
        s_row.addWidget(self._lbl_s0)
        s_row.addWidget(self._slider, stretch=1)
        s_row.addWidget(self._lbl_s1)
        lay.addLayout(s_row)

        # ── controlos de play ─────────────────────────────────────────────────
        ctrl = QHBoxLayout()
        self._btn_play = QPushButton("▶  Play")
        self._btn_play.setFixedHeight(28); self._btn_play.setCheckable(True)
        self._btn_play.clicked.connect(self._toggle_play)
        ctrl.addWidget(self._btn_play)

        ctrl.addWidget(QLabel("  Speed:"))
        self._speed_group: list[QPushButton] = []
        for lbl, spd in [("0.5×", 0.5), ("1×", 1.0), ("2×", 2.0), ("5×", 5.0)]:
            b = QPushButton(lbl)
            b.setFixedHeight(24); b.setFixedWidth(42)
            b.setCheckable(True); b.setChecked(spd == 1.0)
            b.clicked.connect(lambda _, s=spd: self._set_speed(s))
            ctrl.addWidget(b); self._speed_group.append(b)
        ctrl.addStretch()
        self._lbl_time = QLabel("t = 0.00 s")
        self._lbl_time.setStyleSheet("font-size:11px;color:#555;min-width:90px;")
        ctrl.addWidget(self._lbl_time)
        lay.addLayout(ctrl)

        # ── estado interno ────────────────────────────────────────────────────
        self._df:       pd.DataFrame | None = None
        self._x:        np.ndarray   | None = None
        self._R_calib:  np.ndarray   | None = None   # matriz de calibração IMU → prótese
        self._calib_signs = {"roll": 1, "pitch": 1, "yaw": 1}  # flips da calibração
        # scatter + smooth lines para RPY e Kg
        self._sc_roll   = self._sc_pitch = self._sc_yaw = self._sc_kg_line = None
        self._ln_roll   = self._ln_pitch = self._ln_yaw = self._ln_kg2    = None
        # ângulos suavizados p/ modelo 3D — twin: raw (firmware) + corrigido (R)
        self._roll_s:  np.ndarray | None = None   # raw firmware
        self._pitch_s: np.ndarray | None = None
        self._yaw_s:   np.ndarray | None = None
        self._roll_c:  np.ndarray | None = None   # corrigido pelo R
        self._pitch_c: np.ndarray | None = None
        self._yaw_c:   np.ndarray | None = None
        self._vl_rpy    = self._vl_kg = None
        self._dragging  = False
        self._speed     = 1.0
        self._timer     = QTimer()
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_tick)
        self._play_wall0 = self._play_data0 = 0.0

        # toggle connections (via _set_vis_rpy / _set_vis_kg que respeitam fit mode)
        self._btn_roll.toggled.connect(lambda v: self._set_vis_rpy("roll",  v))
        self._btn_pitch.toggled.connect(lambda v: self._set_vis_rpy("pitch", v))
        self._btn_yaw.toggled.connect(lambda v: self._set_vis_rpy("yaw",   v))
        self._btn_kg.toggled.connect(self._set_vis_kg)
        self._btn_fit.toggled.connect(self._on_fit)

        # slider
        self._slider.valueChanged.connect(self._on_slider)

        # mouse scrub: RPY canvas
        for sig, fn in [("button_press_event",   self._on_press),
                        ("motion_notify_event",   self._on_motion),
                        ("button_release_event",  self._on_release)]:
            self._cv_rpy.mpl_connect(sig, fn)
        # mouse scrub: Kg canvas
        for sig, fn in [("button_press_event",   self._on_press),
                        ("motion_notify_event",   self._on_motion),
                        ("button_release_event",  self._on_release)]:
            self._cv_kg2.mpl_connect(sig, fn)

    # ── helpers de visibilidade ───────────────────────────────────────────────

    def _set_vis_rpy(self, axis: str, v: bool):
        show_line = self._btn_fit.isChecked()
        sc = getattr(self, f"_sc_{axis}", None)
        ln = getattr(self, f"_ln_{axis}", None)
        if sc: sc.set_visible(v and not show_line)
        if ln: ln.set_visible(v and show_line)
        self._cv_rpy.draw_idle()

    def _set_vis_kg(self, v: bool):
        show_line = self._btn_fit.isChecked()
        if self._sc_kg_line: self._sc_kg_line.set_visible(v and not show_line)
        if self._ln_kg2:     self._ln_kg2.set_visible(v and show_line)
        self._cv_kg2.draw_idle()

    def _on_fit(self, show_line: bool):
        self._btn_fit.setText("Points" if show_line else "Line")
        for axis, btn in [("roll", self._btn_roll),
                          ("pitch", self._btn_pitch),
                          ("yaw",   self._btn_yaw)]:
            v = btn.isChecked()
            sc = getattr(self, f"_sc_{axis}", None)
            ln = getattr(self, f"_ln_{axis}", None)
            if sc: sc.set_visible(v and not show_line)
            if ln: ln.set_visible(v and show_line)
        v = self._btn_kg.isChecked()
        if self._sc_kg_line: self._sc_kg_line.set_visible(v and not show_line)
        if self._ln_kg2:     self._ln_kg2.set_visible(v and show_line)
        self._cv_rpy.draw_idle()
        self._cv_kg2.draw_idle()

    def set_model_base(self, payload):
        """Recebe a orientação-base do modelo 3D (turns) vinda da Calibração."""
        self._model_turns = dict(payload) if isinstance(payload, dict) \
            else _load_model_turns()
        self._R_base = _turns_to_R(self._model_turns)
        self._redraw_3d()

    def _center_3d(self):
        """Coloca o eixo 3D como um quadrado centrado na figura — evita que o
        cubo fique descentrado ou cortado num painel largo-e-baixo. Reserva
        espaço no topo para o título (2 linhas)."""
        w = self._fig_3d.get_figwidth()  * self._fig_3d.dpi
        h = self._fig_3d.get_figheight() * self._fig_3d.dpi
        if w <= 1 or h <= 1:
            return
        top_res, bot_res = 0.12, 0.04          # folga p/ título / rótulos
        avail = 1 - top_res - bot_res
        side = min(w * 0.92, h * avail)
        fw, fh = side / w, side / h
        self._ax_3d.set_position([(1 - fw) / 2, 1 - top_res - fh, fw, fh])
        self._cv_3d.draw_idle()

    def _redraw_3d(self):
        """Redesenha o taco na posição actual do slider (após mudar a pose)."""
        if self._roll_s is not None:
            self._update_3d(self._slider.value())

    def _update_3d(self, idx: int):
        """Modelo único: corrigido (R) se houver calibração; senão firmware (raw)."""
        if self._roll_s is None:
            return
        if self._roll_c is not None:
            draw_prosthetic(self._ax_3d,
                            float(self._roll_c[idx]),
                            float(self._pitch_c[idx]),
                            float(self._yaw_c[idx]),
                            label="PROSTHESIS (corrected)",
                            R_base=self._R_base)
        else:
            draw_prosthetic(self._ax_3d,
                            float(self._roll_s[idx]),
                            float(self._pitch_s[idx]),
                            float(self._yaw_s[idx]),
                            label="FIRMWARE (raw — no R)",
                            R_base=self._R_base)
        self._center_3d()

    def _set_idx(self, idx: int):
        if self._x is None:
            return
        idx = max(0, min(idx, len(self._x) - 1))
        t = float(self._x[idx])
        if self._vl_rpy: self._vl_rpy.set_xdata([t, t])
        if self._vl_kg:  self._vl_kg.set_xdata([t, t])
        self._cv_rpy.draw_idle()
        self._cv_kg2.draw_idle()
        self._update_3d(idx)
        self._lbl_time.setText(f"t = {t:.2f} s")

    # ── carga de dados ────────────────────────────────────────────────────────

    def load(self, df: pd.DataFrame):
        self._stop_play()
        self._df = df
        x  = (df["sample_us"] - df["sample_us"].iloc[0]) / 1e6
        self._x = x.to_numpy()
        w  = max(5, len(df) // 60)
        sl = self._btn_fit.isChecked()

        # ── ângulos: corrigidos (se R disponível) ou firmware ─────────────────
        def _uw(arr: np.ndarray) -> np.ndarray:
            return np.degrees(np.unwrap(np.radians(arr)))

        def _col(c: str) -> np.ndarray:
            return df[c].to_numpy(float) if c in df.columns else np.zeros(len(df))

        _RAW_COLS = ("imu_ax","imu_ay","imu_az","imu_gx","imu_gy","imu_gz",
                     "imu_mx","imu_my","imu_mz")
        has_raw = all(c in df.columns for c in _RAW_COLS)

        # ── ângulos RAW (firmware) — sempre; alimentam o taco da ESQUERDA ──────
        roll_raw  = _uw(_col("roll_deg"))
        pitch_raw = _uw(_col("pitch_deg"))
        yaw_raw   = _uw(_col("yaw_deg"))
        self._roll_s  = _smooth(roll_raw,  w)
        self._pitch_s = _smooth(pitch_raw, w)
        self._yaw_s   = _smooth(yaw_raw,   w)

        # ── ângulos CORRIGIDOS (R) — se houver calibração; taco da DIREITA ─────
        if self._R_calib is not None and has_raw:
            s = self._calib_signs
            dfc = apply_imu_correction(df, self._R_calib,
                                       sr=s["roll"], sp=s["pitch"], sy=s["yaw"])
            roll_uw  = _uw(dfc["pro_roll"].values)
            pitch_uw = _uw(dfc["pro_pitch"].values)
            yaw_uw   = _uw(dfc["pro_yaw"].values)
            self._roll_c  = _smooth(roll_uw,  w)
            self._pitch_c = _smooth(pitch_uw, w)
            self._yaw_c   = _smooth(yaw_uw,   w)
            title_rpy = "Roll / Pitch / Yaw  — corrected by calibration R"
        else:
            roll_uw, pitch_uw, yaw_uw = roll_raw, pitch_raw, yaw_raw
            self._roll_c = self._pitch_c = self._yaw_c = None
            title_rpy = "Roll / Pitch / Yaw  — firmware  (no R calibration)"

        # ── Gráfico RPY ────────────────────────────────────────────────────────
        self._ax_rpy.cla()
        for uw_arr, btn, color, lbl, sc_attr, ln_attr in [
            (roll_uw,  self._btn_roll,  C["roll"],  "Roll",  "_sc_roll",  "_ln_roll"),
            (pitch_uw, self._btn_pitch, C["pitch"], "Pitch", "_sc_pitch", "_ln_pitch"),
            (yaw_uw,   self._btn_yaw,   C["yaw"],   "Yaw",   "_sc_yaw",   "_ln_yaw"),
        ]:
            sc = self._ax_rpy.scatter(
                x, uw_arr, s=_SCATTER_PT, c=color, label=lbl, alpha=0.85, linewidths=0)
            sc.set_visible(btn.isChecked() and not sl)
            setattr(self, sc_attr, sc)

            ln, = self._ax_rpy.plot(
                x, _smooth(uw_arr, w), color=color, linewidth=1.4, alpha=0.9)
            ln.set_visible(btn.isChecked() and sl)
            setattr(self, ln_attr, ln)

        self._ax_rpy.set_ylabel("Angle (°)", fontsize=9)
        self._ax_rpy.set_xlabel("Time (s)", fontsize=9)
        self._ax_rpy.legend(loc="upper right", fontsize=8, markerscale=4)
        self._ax_rpy.grid(True, linestyle="--", alpha=0.3)
        self._ax_rpy.set_title(title_rpy, fontsize=9)
        self._vl_rpy = self._ax_rpy.axvline(
            x=float(self._x[0]), color="black", lw=1.5, ls="--", alpha=0.7, zorder=5)

        # ── Gráfico Kg ─────────────────────────────────────────────────────────
        self._ax_kg2.cla()
        self._sc_kg_line = self._ln_kg2 = None
        if "load_cell_est_kg" in df.columns:
            y_kg = df["load_cell_est_kg"].to_numpy(float)
            sc = self._ax_kg2.scatter(
                x, y_kg, s=_SCATTER_PT, c=C["kg"], label="Kg", alpha=0.85, linewidths=0)
            sc.set_visible(self._btn_kg.isChecked() and not sl)
            self._sc_kg_line = sc

            ln, = self._ax_kg2.plot(
                x, _smooth(y_kg, w), color=C["kg"], linewidth=1.4, alpha=0.9)
            ln.set_visible(self._btn_kg.isChecked() and sl)
            self._ln_kg2 = ln

        self._ax_kg2.set_ylabel("Weight (kg)", fontsize=9, color=C["kg"])
        self._ax_kg2.set_xlabel("Time (s)", fontsize=9)
        self._ax_kg2.tick_params(axis="y", labelcolor=C["kg"])
        self._ax_kg2.grid(True, linestyle="--", alpha=0.3)
        self._ax_kg2.set_title("Force (kg)", fontsize=10)
        self._vl_kg = self._ax_kg2.axvline(
            x=float(self._x[0]), color="black", lw=1.5, ls="--", alpha=0.7, zorder=5)

        # slider
        self._slider.setMaximum(len(self._x) - 1)
        self._slider.setValue(0)
        self._lbl_s1.setText(f"{float(self._x[-1]):.1f} s")

        self._cv_rpy.draw_idle()
        self._cv_kg2.draw_idle()
        self._update_3d(0)

    # ── mouse scrubber nos gráficos 2D ────────────────────────────────────────

    def _x_from_event(self, event) -> float | None:
        """Devolve o xdata se o evento foi num eixo 2D conhecido."""
        if event.inaxes in (self._ax_rpy, self._ax_kg2) and event.xdata is not None:
            return float(event.xdata)
        return None

    def _on_press(self, event):
        xd = self._x_from_event(event)
        if xd is None or self._x is None:
            return
        self._dragging = True
        idx = int(np.searchsorted(self._x, xd))
        self._slider.setValue(max(0, min(idx, len(self._x) - 1)))

    def _on_motion(self, event):
        if not self._dragging:
            return
        xd = self._x_from_event(event)
        if xd is None or self._x is None:
            return
        idx = int(np.searchsorted(self._x, xd))
        self._slider.setValue(max(0, min(idx, len(self._x) - 1)))

    def _on_release(self, event):
        self._dragging = False

    # ── slider ────────────────────────────────────────────────────────────────

    def _on_slider(self, idx: int):
        self._set_idx(idx)

    # ── play / pause ──────────────────────────────────────────────────────────

    def _toggle_play(self, checked: bool):
        self._start_play() if checked else self._stop_play()

    def _start_play(self):
        if self._x is None:
            self._btn_play.setChecked(False); return
        cur = self._slider.value()
        if cur >= len(self._x) - 1:
            cur = 0; self._slider.setValue(0)
        self._play_data0 = float(self._x[cur])
        self._play_wall0 = self._time_mod.monotonic()
        self._btn_play.setText("⏸  Pausa")
        self._timer.start()

    def _stop_play(self):
        self._timer.stop()
        self._btn_play.setChecked(False)
        self._btn_play.setText("▶  Play")

    def _on_tick(self):
        if self._x is None:
            self._stop_play(); return
        data_t = self._play_data0 + (self._time_mod.monotonic() - self._play_wall0) * self._speed
        if data_t >= float(self._x[-1]):
            self._slider.setValue(len(self._x) - 1); self._stop_play(); return
        idx = min(int(np.searchsorted(self._x, data_t)), len(self._x) - 1)
        self._slider.setValue(idx)

    def _set_speed(self, spd: float):
        self._speed = spd
        for b in self._speed_group:
            b.setChecked(False)
        if self._timer.isActive() and self._x is not None:
            cur = self._slider.value()
            self._play_data0 = float(self._x[cur])
            self._play_wall0 = self._time_mod.monotonic()

    def set_R(self, payload):
        """Recebe R (e sinais de flip) da calibração e recarrega com correcção aplicada.

        payload pode ser uma matriz R ou um tuplo (R, signs).
        """
        if isinstance(payload, tuple):
            R, signs = payload
        else:
            R, signs = payload, None
        self._R_calib = R
        if signs:
            self._calib_signs = dict(signs)
        if self._df is not None:
            self.load(self._df)



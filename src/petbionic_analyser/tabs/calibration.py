"""Tab – Calibração IMU."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider,
    QFileDialog, QMessageBox, QScrollArea,
)
from PyQt6.QtCore import Qt, pyqtSignal

from ..config import C, DOWNLOADS_DIR, _BASE
from ..imu import compute_R, apply_imu_correction, _turns_to_R
from ..viz3d import draw_prosthetic
from ..persistence import (
    _save_calib, _load_calib, _save_runs, _load_runs,
    _save_model_turns, _load_model_turns,
)
from ..gait import _smooth


class CalibrationTab(QWidget):
    R_computed = pyqtSignal(object)        # emite (R, signs) quando R é calculado
    model_base_changed = pyqtSignal(object)  # emite os "turns" da pose-base 3D

    _AXES    = ("roll", "pitch", "yaw")
    _GCOLS   = ("imu_gx", "imu_gy", "imu_gz")
    _GCOLORS = (C["x"], C["y"], C["z"])
    _RAW_COLS = ("imu_ax","imu_ay","imu_az","imu_gx","imu_gy","imu_gz",
                 "imu_mx","imu_my","imu_mz")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._paths   = {a: None for a in self._AXES}
        self._dfs     = {a: None for a in self._AXES}
        self._R       = None
        self._signs   = {a: 1 for a in self._AXES}
        self._test_df = None
        self._test_path_str: str | None = None

        # orientação-base do modelo 3D (movida do tab Main)
        self._model_turns = _load_model_turns()
        self._R_base = _turns_to_R(self._model_turns)
        # arrays suavizados p/ o 3D (calculados a partir do test CSV)
        self._t3_roll_s = self._t3_pitch_s = self._t3_yaw_s = None   # raw
        self._t3_roll_c = self._t3_pitch_c = self._t3_yaw_c = None   # corrigido

        # tenta carregar R guardado de sessão anterior
        R_saved, signs_saved = _load_calib()
        if R_saved is not None:
            self._R     = R_saved
            self._signs = signs_saved

        # Conteúdo dentro de um QScrollArea para não espremer (era caótico):
        # cada secção fica com a sua altura natural e há scroll se faltar espaço.
        lay = QVBoxLayout()
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # ── instruções ────────────────────────────────────────────────────
        instr = QLabel(
            "<b>IMU Calibration</b>  —  Load one CSV per physical axis of the prosthesis "
            "(pure rotation about ROLL, PITCH and YAW). The tool computes the matrix R "
            "that aligns the IMU frame with the actual prosthesis frame."
        )
        instr.setStyleSheet("color:#444;font-size:11px;padding:4px 2px;")
        instr.setWordWrap(True)
        lay.addWidget(instr)

        # ── selectores de ficheiro (1 por eixo) ───────────────────────────
        desc = {
            "roll":  "Prosthesis upright (vertical shaft). Tilt the foot side to side — rolling inward and outward, like the ankle rolling over the inner/outer edge.",
            "pitch": "Prosthesis upright (vertical shaft). Tilt the foot forward and backward — the ankle motion during the step.",
            "yaw":   "Prosthesis upright (vertical shaft). Rotate the whole leg left and right about the shaft — like the dog changing direction.",
        }
        picker_row = QHBoxLayout()
        picker_row.setSpacing(12)
        self._file_labels: dict[str, QLabel] = {}
        for axis in self._AXES:
            box = QVBoxLayout()
            color = C[axis]
            box.addWidget(QLabel(f"<b style='color:{color};font-size:12px;'>"
                                 f"{axis.upper()}</b>"))
            d = QLabel(desc[axis])
            d.setStyleSheet("color:#666;font-size:9px;")
            d.setWordWrap(True)
            box.addWidget(d)
            btn = QPushButton("Select CSV…")
            btn.setFixedHeight(26)
            btn.clicked.connect(lambda _, a=axis: self._pick_run(a))
            box.addWidget(btn)
            lbl = QLabel("—")
            lbl.setStyleSheet("color:#999;font-size:9px;")
            lbl.setWordWrap(True)
            box.addWidget(lbl)
            self._file_labels[axis] = lbl
            picker_row.addLayout(box)
        lay.addLayout(picker_row)

        # ── botão calcular + status ────────────────────────────────────────
        compute_row = QHBoxLayout()
        self._btn_compute = QPushButton("⚙  Compute R")
        self._btn_compute.setFixedHeight(30)
        self._btn_compute.setStyleSheet(
            "QPushButton{background:#3a7bd5;color:white;border-radius:5px;"
            "font-weight:bold;padding:4px 20px;}"
            "QPushButton:hover{background:#2a5fa5;}"
        )
        self._btn_compute.clicked.connect(self._compute)
        compute_row.addWidget(self._btn_compute)
        self._lbl_status = QLabel("Select the 3 runs and click Compute R")
        self._lbl_status.setStyleSheet("color:#888;font-size:11px;padding:0 8px;")
        compute_row.addWidget(self._lbl_status)
        compute_row.addStretch()
        lay.addLayout(compute_row)

        # ── gráfico: giroscópio de cada run (3 subplots lado a lado) ──────
        self._fig_gyro = Figure(figsize=(13, 2.4))
        self._fig_gyro.subplots_adjust(left=0.05, right=0.99, top=0.85,
                                       bottom=0.2, wspace=0.35)
        self._ax_gyro = {a: self._fig_gyro.add_subplot(1, 3, i+1)
                         for i, a in enumerate(self._AXES)}
        self._cv_gyro = FigureCanvasQTAgg(self._fig_gyro)
        self._cv_gyro.setMinimumHeight(190)
        self._cv_gyro.setMaximumHeight(220)
        self._decorate_gyro_titles()
        lay.addWidget(self._cv_gyro)

        # ── separador: CSV de teste + flips ───────────────────────────────
        test_row = QHBoxLayout()
        btn_test = QPushButton("Select test CSV…")
        btn_test.setFixedHeight(26)
        btn_test.clicked.connect(self._pick_test)
        test_row.addWidget(btn_test)
        self._lbl_test = QLabel("(none — load a CSV to see the corrected angles)")
        self._lbl_test.setStyleSheet("color:#888;font-size:10px;")
        test_row.addWidget(self._lbl_test)
        test_row.addStretch()
        sf = QLabel("  Flip sign:")
        sf.setStyleSheet("font-size:10px;color:#555;font-weight:bold;")
        test_row.addWidget(sf)
        self._sign_btns: dict[str, QPushButton] = {}
        for axis in self._AXES:
            b = QPushButton(f"{axis.upper()} ×+1")
            b.setFixedHeight(24)
            b.setStyleSheet(f"color:{C[axis]};font-weight:bold;font-size:10px;"
                            "border:1px solid #ccc;border-radius:3px;padding:0 6px;")
            b.clicked.connect(lambda _, a=axis: self._flip(a))
            test_row.addWidget(b)
            self._sign_btns[axis] = b
        lay.addLayout(test_row)

        # ── gráfico: ângulos corrigidos vs firmware ────────────────────────
        self._fig_corr = Figure(figsize=(13, 3.2))
        gs = self._fig_corr.add_gridspec(3, 1, hspace=0.07,
                                         left=0.06, right=0.99, top=0.91, bottom=0.1)
        self._ax_corr = {
            "roll":  self._fig_corr.add_subplot(gs[0]),
            "pitch": self._fig_corr.add_subplot(gs[1]),
            "yaw":   self._fig_corr.add_subplot(gs[2]),
        }
        self._ax_corr["pitch"].sharex(self._ax_corr["roll"])
        self._ax_corr["yaw"].sharex(self._ax_corr["roll"])
        self._cv_corr = FigureCanvasQTAgg(self._fig_corr)
        self._cv_corr.setMinimumHeight(230)
        self._cv_corr.setMaximumHeight(280)
        self._decorate_corr()
        lay.addWidget(self._cv_corr)

        # ── orientação-base do modelo 3D (botões movidos do Main) ──────────
        mrow = QHBoxLayout()
        lbl_m = QLabel("Orientar taco 3D:")
        lbl_m.setStyleSheet("font-size:10px;color:#555;font-weight:bold;")
        mrow.addWidget(lbl_m)
        for axis in ("x", "y", "z"):
            b = QPushButton(f"↻ {axis.upper()} 90°")
            b.setFixedHeight(24)
            b.setStyleSheet("font-size:10px;padding:0 8px;")
            b.clicked.connect(lambda _, a=axis: self._rotate_model3d(a))
            mrow.addWidget(b)
        btn_reset_m = QPushButton("Reset")
        btn_reset_m.setFixedHeight(24)
        btn_reset_m.setStyleSheet("font-size:10px;padding:0 8px;")
        btn_reset_m.clicked.connect(self._reset_model3d)
        mrow.addWidget(btn_reset_m)
        self._lbl_model = QLabel("")
        self._lbl_model.setStyleSheet("color:#888;font-size:10px;")
        mrow.addWidget(self._lbl_model)
        mrow.addStretch()
        lay.addLayout(mrow)
        self._update_model_label3d()

        # ── comparação 3D: firmware (raw) vs corrigido (R), do test CSV ────
        self._fig_3d = Figure(figsize=(13, 3.4))
        self._ax_3d_raw  = self._fig_3d.add_subplot(121, projection="3d")
        self._ax_3d_corr = self._fig_3d.add_subplot(122, projection="3d")
        self._cv_3d = FigureCanvasQTAgg(self._fig_3d)
        self._cv_3d.setMinimumHeight(300)
        self._cv_3d.setMaximumHeight(340)
        # mantém os cubos centrados (um por metade) e sem cortes ao redimensionar
        self._cv_3d.mpl_connect("resize_event", lambda _e: self._center_3d_calib())
        lay.addWidget(self._cv_3d)

        s_row = QHBoxLayout()
        s_row.addWidget(QLabel("3D:"))
        self._slider3d = QSlider(Qt.Orientation.Horizontal)
        self._slider3d.setMinimum(0); self._slider3d.setMaximum(0)
        self._slider3d.valueChanged.connect(self._update_3d_calib)
        s_row.addWidget(self._slider3d, stretch=1)
        lay.addLayout(s_row)

        # ── R display + botão export ───────────────────────────────────────
        bottom = QHBoxLayout()
        self._lbl_R = QLabel("R: not computed")
        self._lbl_R.setStyleSheet(
            "font-family:monospace;font-size:10px;color:#333;"
            "background:#f7f7f7;padding:6px 8px;"
            "border:1px solid #ddd;border-radius:4px;"
        )
        bottom.addWidget(self._lbl_R, stretch=1)
        btn_export = QPushButton("Export to firmware (C++)")
        btn_export.setFixedHeight(30)
        btn_export.setStyleSheet(
            "QPushButton{background:#2e7d32;color:white;border-radius:5px;"
            "font-weight:bold;padding:4px 16px;}"
            "QPushButton:hover{background:#1b5e20;}"
        )
        btn_export.clicked.connect(self._export)
        bottom.addWidget(btn_export)
        lay.addLayout(bottom)

        # aplica R guardado à UI (depois de todos os widgets existirem)
        if self._R is not None:
            det = np.linalg.det(self._R)
            self._lbl_status.setText(
                f"R loaded from a previous session  ·  det = {det:.5f}")
            self._lbl_status.setStyleSheet(
                "color:#2a5fa5;font-size:11px;padding:0 8px;font-style:italic;")
            self._update_R_label()
            for axis in self._AXES:
                s = self._signs[axis]
                self._sign_btns[axis].setText(
                    f"{axis.upper()} ×{'+' if s > 0 else '-'}1")

        # ── envolve todo o conteúdo num scroll area ────────────────────────
        content = QWidget()
        content.setLayout(lay)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # ── recarrega CSVs de calibração guardados (persistência entre sessões)
        self._restore_runs()

    # ── persistência dos runs de calibração ───────────────────────────────

    def _restore_runs(self):
        paths, test = _load_runs()
        for axis in self._AXES:
            p = paths.get(axis)
            if p and Path(p).exists():
                try:
                    df = pd.read_csv(p); df.columns = df.columns.str.strip()
                    self._paths[axis] = p
                    self._dfs[axis]   = df
                    self._file_labels[axis].setText(Path(p).name)
                    self._plot_gyro_run(axis, df)
                except Exception:
                    pass
        if test and Path(test).exists():
            try:
                df = pd.read_csv(test); df.columns = df.columns.str.strip()
                self._test_df = df
                self._test_path_str = test
                self._lbl_test.setText(Path(test).name)
                self._plot_corrected()
            except Exception:
                pass

    # ── decoração dos eixos ───────────────────────────────────────────────

    def _decorate_gyro_titles(self):
        for axis, ax in self._ax_gyro.items():
            ax.set_title(f"{axis.upper()} run", fontsize=9, color=C[axis], pad=3)
            ax.set_xlabel("Time (s)", fontsize=7)
            ax.set_ylabel("giro (counts)", fontsize=7)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.tick_params(labelsize=7)
        self._cv_gyro.draw_idle()

    def _decorate_corr(self):
        for key, ax in self._ax_corr.items():
            ax.cla()
            ax.set_ylabel(f"{key.upper()} (°)", color=C[key], fontsize=9)
            ax.tick_params(axis="y", labelcolor=C[key], labelsize=8)
            ax.tick_params(labelbottom=False)
            ax.grid(True, linestyle="--", alpha=0.3)
        self._ax_corr["yaw"].tick_params(labelbottom=True)
        self._ax_corr["yaw"].set_xlabel("Time (s)", fontsize=9)
        self._ax_corr["roll"].set_title(
            "Corrected angles ——  vs  firmware - - -  "
            "(load a test CSV to view)", fontsize=9, pad=4)
        self._cv_corr.draw_idle()

    # ── selectores de ficheiro ────────────────────────────────────────────

    def _pick_run(self, axis: str):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Calibration CSV — {axis.upper()}",
            str(DOWNLOADS_DIR), "CSV (*.csv);;All (*)")
        if not path:
            return
        try:
            df = pd.read_csv(path);  df.columns = df.columns.str.strip()
            self._paths[axis] = path
            self._dfs[axis]   = df
            self._file_labels[axis].setText(Path(path).name)
            self._plot_gyro_run(axis, df)
            _save_runs(self._paths, self._test_path_str)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    def _pick_test(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Test CSV", str(DOWNLOADS_DIR), "CSV (*.csv);;All (*)")
        if not path:
            return
        try:
            df = pd.read_csv(path);  df.columns = df.columns.str.strip()
            self._test_df = df
            self._test_path_str = path
            self._lbl_test.setText(Path(path).name)
            self._plot_corrected()
            _save_runs(self._paths, self._test_path_str)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    # ── plot do giroscópio de um run ──────────────────────────────────────

    def _plot_gyro_run(self, axis: str, df: pd.DataFrame):
        ax = self._ax_gyro[axis]
        ax.cla()
        if "sample_us" not in df.columns:
            return
        x = (df["sample_us"] - df["sample_us"].iloc[0]) / 1e6
        dominant = None
        if all(c in df.columns for c in self._GCOLS):
            g   = df[list(self._GCOLS)].values.astype(float)
            mag = np.linalg.norm(g, axis=1)
            thr = np.percentile(mag, 70)
            active = g[mag > thr]
            if len(active) > 3:
                _, _, Vt = np.linalg.svd(active - active.mean(axis=0))
                dominant = int(np.argmax(np.abs(Vt[0])))
            for i, (col, clr) in enumerate(zip(self._GCOLS, self._GCOLORS)):
                if col in df.columns:
                    is_dom = (i == dominant)
                    ax.plot(x, df[col].values, color=clr,
                            lw=1.5 if is_dom else 0.6,
                            alpha=0.95 if is_dom else 0.3,
                            label=col.replace("imu_", ""))
            ax.legend(fontsize=7, loc="upper right")
        dom_name = ["gx", "gy", "gz"][dominant] if dominant is not None else "?"
        ax.set_title(f"{axis.upper()} run  →  dominant axis: {dom_name}",
                     fontsize=9, color=C[axis], pad=3)
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_ylabel("counts", fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=7)
        self._cv_gyro.draw_idle()

    # ── cálculo de R ──────────────────────────────────────────────────────

    def _compute(self):
        missing = [a for a in self._AXES if self._paths[a] is None]
        if missing:
            QMessageBox.warning(self, "Missing runs",
                f"Select the 3 CSVs before computing R.\n"
                f"Missing: {', '.join(m.upper() for m in missing)}")
            return
        try:
            R   = compute_R(self._paths["roll"], self._paths["pitch"], self._paths["yaw"])
            det = np.linalg.det(R)
            self._R = R
            ok  = abs(det - 1.0) < 0.01
            self._lbl_status.setText(
                f"R computed  ✓   det = {det:.5f}  "
                f"{'(OK — valid rotation)' if ok else '⚠  det ≠ 1  (check the runs)'}")
            self._lbl_status.setStyleSheet(
                f"color:{'#2a7a2a' if ok else '#c05000'};"
                "font-size:11px;padding:0 8px;font-weight:bold;")
            self._update_R_label()
            _save_calib(R, self._signs)
            self.R_computed.emit((R, dict(self._signs)))
            self._plot_corrected()
        except Exception as e:
            QMessageBox.critical(self, "Error computing R", str(e))

    def _update_R_label(self):
        if self._R is None:
            self._lbl_R.setText("R: not computed")
            return
        R = self._R
        rows = "  ".join(
            "[ " + "  ".join(f"{v:+.4f}" for v in row) + " ]"
            for row in R
        )
        det = np.linalg.det(R)
        self._lbl_R.setText(
            f"R (IMU → prosthesis):   {rows}     det = {det:.5f}")

    # ── plot dos ângulos corrigidos ────────────────────────────────────────

    def _flip(self, axis: str):
        self._signs[axis] *= -1
        s = self._signs[axis]
        self._sign_btns[axis].setText(f"{axis.upper()} ×{'+' if s > 0 else '-'}1")
        if self._R is not None:
            _save_calib(self._R, self._signs)
            self.R_computed.emit((self._R, dict(self._signs)))
        self._plot_corrected()

    def _plot_corrected(self):
        self._decorate_corr()
        if self._test_df is None:
            return
        df = self._test_df
        if "sample_us" not in df.columns:
            return
        x = (df["sample_us"] - df["sample_us"].iloc[0]) / 1e6

        # ângulos raw do firmware (tracejado)
        for key, col in [("roll","roll_deg"), ("pitch","pitch_deg"), ("yaw","yaw_deg")]:
            if col in df.columns:
                self._ax_corr[key].plot(x, df[col].values, color=C[key],
                                        lw=0.8, ls="--", alpha=0.4, label="firmware")

        # ângulos corrigidos (cheio) — só se R estiver calculado e dados raw disponíveis
        if self._R is not None and all(c in df.columns for c in self._RAW_COLS):
            try:
                dfc = apply_imu_correction(df, self._R,
                                           sr=self._signs["roll"],
                                           sp=self._signs["pitch"],
                                           sy=self._signs["yaw"])
                for key, col in [("roll","pro_roll"),
                                  ("pitch","pro_pitch"),
                                  ("yaw","pro_yaw")]:
                    self._ax_corr[key].plot(x, dfc[col].values, color=C[key],
                                            lw=1.3, label="corrigido")
            except Exception:
                pass

        self._ax_corr["roll"].legend(fontsize=8, loc="upper right")
        for ax in self._ax_corr.values():
            ax.relim();  ax.autoscale_view()
        self._cv_corr.draw_idle()

        # ── prepara arrays p/ o modelo 3D (raw + corrigido) e slider ───────
        w = max(5, len(df) // 60)
        def _uw(a): return np.degrees(np.unwrap(np.radians(a)))
        def _col(c): return df[c].to_numpy(float) if c in df.columns else np.zeros(len(df))
        self._t3_roll_s  = _smooth(_uw(_col("roll_deg")),  w)
        self._t3_pitch_s = _smooth(_uw(_col("pitch_deg")), w)
        self._t3_yaw_s   = _smooth(_uw(_col("yaw_deg")),   w)
        if self._R is not None and all(c in df.columns for c in self._RAW_COLS):
            dfc = apply_imu_correction(df, self._R, sr=self._signs["roll"],
                                       sp=self._signs["pitch"], sy=self._signs["yaw"])
            self._t3_roll_c  = _smooth(_uw(dfc["pro_roll"].values),  w)
            self._t3_pitch_c = _smooth(_uw(dfc["pro_pitch"].values), w)
            self._t3_yaw_c   = _smooth(_uw(dfc["pro_yaw"].values),   w)
        else:
            self._t3_roll_c = self._t3_pitch_c = self._t3_yaw_c = None
        self._slider3d.setMaximum(max(0, len(df) - 1))
        self._slider3d.setValue(0)
        self._update_3d_calib(0)

    # ── modelo 3D (comparação firmware vs corrigido) ──────────────────────

    def _center_3d_calib(self):
        """Centra cada cubo na sua metade da figura, com folga p/ o título."""
        w = self._fig_3d.get_figwidth()  * self._fig_3d.dpi
        h = self._fig_3d.get_figheight() * self._fig_3d.dpi
        if w <= 1 or h <= 1:
            return
        # painel curto → reserva relativa maior no topo p/ o título de 2 linhas
        top_res = 0.30 if h < 220 else 0.16
        bot_res = 0.04
        avail = 1 - top_res - bot_res
        side = min((w / 2) * 0.92, h * avail)
        fw, fh = side / w, side / h
        y = 1 - top_res - fh
        self._ax_3d_raw.set_position([0.25 - fw / 2, y, fw, fh])
        self._ax_3d_corr.set_position([0.75 - fw / 2, y, fw, fh])
        self._cv_3d.draw_idle()

    def _update_3d_calib(self, idx: int):
        if self._t3_roll_s is None:
            return
        idx = max(0, min(idx, len(self._t3_roll_s) - 1))
        draw_prosthetic(self._ax_3d_raw,
                        float(self._t3_roll_s[idx]), float(self._t3_pitch_s[idx]),
                        float(self._t3_yaw_s[idx]), label="FIRMWARE (raw)",
                        R_base=self._R_base)
        if self._t3_roll_c is not None:
            draw_prosthetic(self._ax_3d_corr,
                            float(self._t3_roll_c[idx]), float(self._t3_pitch_c[idx]),
                            float(self._t3_yaw_c[idx]), label="CORRECTED (R)",
                            R_base=self._R_base)
        else:
            self._ax_3d_corr.cla()
            self._ax_3d_corr.set_title("CORRECTED (R)\n— no R calibration —",
                                       fontsize=8, pad=4)
            for lim in ("set_xlim", "set_ylim", "set_zlim"):
                getattr(self._ax_3d_corr, lim)(-0.75, 0.75)
        self._center_3d_calib()

    def _rotate_model3d(self, axis: str):
        self._model_turns[axis] = (self._model_turns[axis] + 1) % 4
        self._R_base = _turns_to_R(self._model_turns)
        _save_model_turns(self._model_turns)
        self._update_model_label3d()
        self._update_3d_calib(self._slider3d.value())
        self.model_base_changed.emit(dict(self._model_turns))

    def _reset_model3d(self):
        self._model_turns = {"x": 0, "y": 0, "z": 0}
        self._R_base = _turns_to_R(self._model_turns)
        _save_model_turns(self._model_turns)
        self._update_model_label3d()
        self._update_3d_calib(self._slider3d.value())
        self.model_base_changed.emit(dict(self._model_turns))

    def _update_model_label3d(self):
        t = self._model_turns
        self._lbl_model.setText(f"X:{t['x']*90}°  Y:{t['y']*90}°  Z:{t['z']*90}°")

    # ── exportar C++ ──────────────────────────────────────────────────────

    def _export(self):
        if self._R is None:
            QMessageBox.information(self, "Info", "Calcula R primeiro.")
            return
        R   = self._R
        det = np.linalg.det(R)
        lines = [
            "// Rotation matrix R: IMU frame → prosthesis frame",
            f"// det(R) = {det:.6f}  (generated by petBionic CSV Analyzer)",
            "// Usage: v_prosthesis = R * v_imu",
            "const float R_IMU_TO_PROSTHESIS[3][3] = {",
        ]
        for i, row in enumerate(R):
            comma = "," if i < 2 else ""
            lines.append(
                "    {{ {:+.6f}f, {:+.6f}f, {:+.6f}f }}{}".format(*row, comma))
        lines.append("};")
        code = "\n".join(lines)

        dlg = QMessageBox(self)
        dlg.setWindowTitle("C++ code — R for firmware")
        dlg.setText("Copy the code below into the firmware ('Show Details' button):")
        dlg.setDetailedText(code)
        dlg.setStandardButtons(QMessageBox.StandardButton.Ok |
                               QMessageBox.StandardButton.Save)
        dlg.button(QMessageBox.StandardButton.Save).setText("Save file…")
        ret = dlg.exec()
        if ret == QMessageBox.StandardButton.Save:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save R", str(_BASE / "imu_calibration.h"),
                "C++ header (*.h);;Text (*.txt);;All (*)")
            if path:
                Path(path).write_text(code + "\n")



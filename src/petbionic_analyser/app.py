"""Janela principal e arranque da aplicação."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("QtAgg")

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QVBoxLayout, QLabel,
    QTabWidget, QPushButton, QMessageBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette, QColor

from .config import _CALIB_FILE
from .persistence import _load_calib
from .widgets import FileBrowserPanel
from .tabs.load_cell import LoadCellTab
from .tabs.gait_features import GaitFeaturesTab
from .tabs.imu_tab import ImuTab
from .tabs.orientation3d import Orientation3DTab
from .tabs.calibration import CalibrationTab


# ══════════════════════════════════════════════════════════════════════════
#  Janela principal
# ══════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PetBionic Analyser")
        self.setMinimumSize(1280, 780)

        # splitter horizontal: browser | conteúdo
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # --- painel esquerdo: browser (escondido até o botão o mostrar) -----
        self._browser = FileBrowserPanel()
        self._browser.file_selected.connect(self._load)
        self._browser.setVisible(False)
        splitter.addWidget(self._browser)

        # --- painel direito: info + tabs ------------------------------------
        right = QWidget()
        r_lay = QVBoxLayout(right)
        r_lay.setContentsMargins(4, 4, 4, 4)
        r_lay.setSpacing(4)

        self._info = QLabel("Double-click a file to load")
        self._info.setStyleSheet(
            "color:#666;font-size:11px;padding:4px 6px;"
            "border-bottom:1px solid #ddd;")
        r_lay.addWidget(self._info)

        tabs = QTabWidget()

        # botão para mostrar/esconder o explorador de ficheiros (no canto das tabs)
        self._btn_files = QPushButton("📁  Files")
        self._btn_files.setCheckable(True)
        self._btn_files.setFixedHeight(26)
        self._btn_files.setStyleSheet(
            "QPushButton{padding:2px 12px;margin:2px 4px;border:1px solid #bbb;"
            "border-radius:4px;background:#f0f0f0;}"
            "QPushButton:checked{background:#3a7bd5;color:white;border-color:#2a5fa5;}")
        self._btn_files.toggled.connect(self._browser.setVisible)
        tabs.setCornerWidget(self._btn_files, Qt.Corner.TopRightCorner)

        self._tab_lc  = LoadCellTab()
        self._tab_gait = GaitFeaturesTab()
        self._tab_acc = ImuTab(
            cols=("imu_ax", "imu_ay", "imu_az"),
            labels=("ax", "ay", "az"),
            title="Accelerometer",
            ylabel="Acceleration (counts)",
        )
        self._tab_gyr = ImuTab(
            cols=("imu_gx", "imu_gy", "imu_gz"),
            labels=("gx", "gy", "gz"),
            title="Gyroscope",
            ylabel="Angular velocity (counts)",
        )
        self._tab_mag = ImuTab(
            cols=("imu_mx", "imu_my", "imu_mz"),
            labels=("mx", "my", "mz"),
            title="Magnetometer",
            ylabel="Magnetic field (counts)",
        )
        # Orientação — análise (zoom + unwrap, igual às outras tabs IMU)
        self._tab_ori_ana = ImuTab(
            cols=("roll_deg", "pitch_deg", "yaw_deg"),
            labels=("Roll", "Pitch", "Yaw"),
            title="Orientation — Analysis",
            ylabel="Angle (°)",
            show_magnitude=False,
            unwrap=True,
        )
        # Tab principal: orientação 3D + Kg + scrubber + play
        self._tab_ori_3d = Orientation3DTab()

        self._tab_calib = CalibrationTab()
        self._tab_calib.R_computed.connect(self._tab_ori_3d.set_R)
        self._tab_calib.model_base_changed.connect(self._tab_ori_3d.set_model_base)

        # "Main" é a primeira tab (interface principal de análise)
        tabs.addTab(self._tab_ori_3d,  "Main")
        tabs.addTab(self._tab_lc,      "Load Cell")
        tabs.addTab(self._tab_gait,    "Gait Features")
        tabs.addTab(self._tab_acc,     "Accelerometer")
        tabs.addTab(self._tab_gyr,     "Gyroscope")
        tabs.addTab(self._tab_mag,     "Magnetometer")
        tabs.addTab(self._tab_ori_ana, "Orientation — Analysis")
        tabs.addTab(self._tab_calib,   "IMU Calibration")

        r_lay.addWidget(tabs)
        splitter.addWidget(right)
        splitter.setSizes([260, 1020])

        # carrega R + sinais persistidos (se existirem) e aplica ao tab Main
        R_saved, signs_saved = _load_calib()
        if R_saved is not None:
            self._tab_ori_3d.set_R((R_saved, signs_saved))
            self.statusBar().showMessage(
                f"Calibration R loaded from {_CALIB_FILE.name}  ·  "
                f"det = {np.linalg.det(R_saved):.5f}")
        else:
            self.statusBar().showMessage("Ready")

    # ── carregamento ─────────────────────────────────────────────────────────

    def _load(self, path: str):
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()

            if "sample_us" not in df.columns:
                raise ValueError("Column 'sample_us' not found — invalid file.")

            fname = Path(path).name
            n     = len(df)
            dur   = (df["sample_us"].iloc[-1] - df["sample_us"].iloc[0]) / 1e6

            self._info.setText(
                f"<b>{fname}</b>  &nbsp;·&nbsp;  {n:,} samples  "
                f"&nbsp;·&nbsp;  {dur:.2f} s"
            )
            self._info.setStyleSheet(
                "color:#111;font-size:11px;padding:4px 6px;"
                "border-bottom:1px solid #ddd;")
            self.setWindowTitle(f"PetBionic Analyser — {fname}")

            self._tab_lc.load(df)
            self._tab_gait.load(df)
            self._tab_acc.load(df)
            self._tab_gyr.load(df)
            self._tab_mag.load(df)
            self._tab_ori_ana.load(df)
            self._tab_ori_3d.load(df)

            self.statusBar().showMessage(
                f"Loaded: {fname}  |  {n:,} samples  |  {dur:.2f} s")

        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))


# ══════════════════════════════════════════════════════════════════════════
#  Entrada
# ══════════════════════════════════════════════════════════════════════════

def _apply_light_palette(app):
    """Força um tema claro consistente (a UI foi desenhada para fundo claro).
    Evita o 'bug' do dark mode do macOS: texto escuro sobre fundo escuro."""
    p = QPalette()
    c = QColor
    p.setColor(QPalette.ColorRole.Window,          c("#ececec"))
    p.setColor(QPalette.ColorRole.WindowText,      c("#111111"))
    p.setColor(QPalette.ColorRole.Base,            c("#ffffff"))
    p.setColor(QPalette.ColorRole.AlternateBase,   c("#f5f5f5"))
    p.setColor(QPalette.ColorRole.Text,            c("#111111"))
    p.setColor(QPalette.ColorRole.Button,          c("#e6e6e6"))
    p.setColor(QPalette.ColorRole.ButtonText,      c("#111111"))
    p.setColor(QPalette.ColorRole.ToolTipBase,     c("#ffffff"))
    p.setColor(QPalette.ColorRole.ToolTipText,     c("#111111"))
    p.setColor(QPalette.ColorRole.PlaceholderText, c("#888888"))
    p.setColor(QPalette.ColorRole.Highlight,       c("#3a7bd5"))
    p.setColor(QPalette.ColorRole.HighlightedText, c("#ffffff"))
    app.setPalette(p)


def _fix_macos_menu_name(name: str = "PetBionic Analyser"):
    """No macOS, o nome na barra de menu vem do CFBundleName do bundle do
    executável (o Python do Homebrew → 'Python'). Reescrevemo-lo em runtime,
    antes de a NSApplication arrancar, para mostrar o nome da app."""
    if sys.platform != "darwin":
        return
    try:
        from Foundation import NSBundle
        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        if info is not None:
            info["CFBundleName"] = name
    except Exception:
        pass


def main():
    _fix_macos_menu_name()
    app = QApplication(sys.argv)
    app.setApplicationName("PetBionic Analyser")
    app.setApplicationDisplayName("PetBionic Analyser")
    app.setStyle("Fusion")
    _apply_light_palette(app)
    win = MainWindow()
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        win._load(sys.argv[1])
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

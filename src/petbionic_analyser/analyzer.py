#!/usr/bin/env python3
"""
petBionic CSV Analyzer v2
Visualizador interactivo de dados da prótese canina.

Tabs  : Célula de Carga | Acelerómetro | Giroscópio | Magnetómetro | Orientação 3D
Browse: Downloads + TestData pré-visualizados na barra lateral
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (activa projection='3d')

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider,
    QTabWidget, QSizePolicy, QMessageBox, QFileDialog,
    QTreeWidget, QTreeWidgetItem, QStyle, QScrollArea,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor


# ── caminhos ───────────────────────────────────────────────────────────────

# Esta app vive separada do projeto (~/Developer/PetBionic Analyser), mas lê os
# dados do projeto principal. O caminho do projeto pode ser configurado pela
# variável de ambiente PETBIONIC_ROOT; por omissão assume ~/Desktop/petBionic.
_BASE = Path(os.environ.get("PETBIONIC_ROOT", Path.home() / "Desktop" / "petBionic"))
TESTDATA_DIR  = _BASE / "TestData"
DOWNLOADS_DIR = Path.home() / "Downloads"
# estado de runtime guardado na raiz do repo (src/petbionic_analyser/ → parents[2])
_STATE_DIR    = Path(__file__).resolve().parents[2]
_CALIB_FILE   = _STATE_DIR / "imu_calibration_R.json"
_RUNS_FILE    = _STATE_DIR / "calib_runs.json"


def _save_runs(paths: dict[str, str | None], test: str | None) -> None:
    """Persiste os caminhos dos CSV de calibração (roll/pitch/yaw + teste)."""
    data = {"paths": {k: v for k, v in paths.items()}, "test": test}
    try:
        _RUNS_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def _load_runs() -> tuple[dict[str, str | None], str | None]:
    if not _RUNS_FILE.exists():
        return {}, None
    try:
        data = json.loads(_RUNS_FILE.read_text())
        return data.get("paths", {}) or {}, data.get("test")
    except Exception:
        return {}, None


def _save_calib(R: np.ndarray, signs: dict[str, int]) -> None:
    data = {
        "R":     R.tolist(),
        "signs": signs,
        "det":   float(np.linalg.det(R)),
    }
    _CALIB_FILE.write_text(json.dumps(data, indent=2))


def _load_calib() -> tuple[np.ndarray, dict[str, int]] | tuple[None, None]:
    if not _CALIB_FILE.exists():
        return None, None
    try:
        data  = json.loads(_CALIB_FILE.read_text())
        R     = np.array(data["R"])
        signs = data.get("signs", {"roll": 1, "pitch": 1, "yaw": 1})
        return R, signs
    except Exception:
        return None, None


# ── orientação base do MODELO 3D (pose de repouso do desenho) ────────────────
# Guardada como nº de voltas de 90° em cada eixo (x, y, z) → reorientar o taco
# para o pé/cabeça apontarem no sentido certo. Independente da calibração R.

_MODEL_FILE = _STATE_DIR / "model_orientation.json"


def _turns_to_R(turns: dict[str, int]) -> np.ndarray:
    """Constrói a rotação base a partir de voltas de 90° em x, y, z."""
    return _euler_to_R(90.0 * turns.get("x", 0),
                       90.0 * turns.get("y", 0),
                       90.0 * turns.get("z", 0))


def _save_model_turns(turns: dict[str, int]) -> None:
    _MODEL_FILE.write_text(json.dumps(turns, indent=2))


def _load_model_turns() -> dict[str, int]:
    if _MODEL_FILE.exists():
        try:
            d = json.loads(_MODEL_FILE.read_text())
            return {k: int(d.get(k, 0)) % 4 for k in ("x", "y", "z")}
        except Exception:
            pass
    return {"x": 0, "y": 0, "z": 0}


# ── paleta de cores ────────────────────────────────────────────────────────

C = {
    "kg":    "#1f77b4",
    "raw":   "#d95f02",
    "x":     "#e41a1c",
    "y":     "#4daf4a",
    "z":     "#984ea3",
    "roll":  "#e41a1c",
    "pitch": "#4daf4a",
    "yaw":   "#984ea3",
}

_SCATTER_PT = 3   # tamanho dos pontos do scatter (px)


# ══════════════════════════════════════════════════════════════════════════
#  Constantes do firmware + filtro quaternião (espelho de OrientationEstimator.h)
# ══════════════════════════════════════════════════════════════════════════

_GYRO_SCALE   = 1.0 / 131.0
_ACCEL_SCALE  = 1.0 / 16384.0
_MAG_SCALE    = 0.15
_GYRO_BIAS    = np.array([1.0938, 1.3516, 1.9059])
_MAG_OFFSET   = np.array([37.12,  14.78, -42.45])
_MAG_SCALE_C  = np.array([0.9979,  0.9896,  1.0128])
_ALPHA        = 0.98
_FAST_ROT_THR = 15.0
_D2R = np.pi / 180.0
_R2D = 180.0 / np.pi


def _qmul(q1, q2):
    w1, x1, y1, z1 = q1;  w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def _qnorm(q):
    m = np.linalg.norm(q)
    return q / m if m > 0 else q

def _slerp(q1, q2, t):
    dot = np.dot(q1, q2);  qa = q1.copy()
    if dot < 0.0: qa = -qa;  dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    th = np.arccos(dot);  s = np.sin(th)
    if s < 1e-3: return qa + (q2 - qa) * t
    return qa * (np.sin((1 - t) * th) / s) + q2 * (np.sin(t * th) / s)

def _from_euler(r_deg, p_deg, y_deg):
    r, p, y = r_deg*_D2R*0.5, p_deg*_D2R*0.5, y_deg*_D2R*0.5
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                     cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])

def _to_euler(q):
    w, x, y, z = q
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)) * _R2D
    sinp  = np.clip(2*(w*y - z*x), -1.0, 1.0)
    pitch = np.arcsin(sinp) * _R2D
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)) * _R2D
    if yaw < 0: yaw += 360.0
    return roll, pitch, yaw

def _ref_q(ax, ay, az, mx, my, mz):
    roll  = np.arctan2(ay, az) * _R2D
    pitch = np.arctan2(-ax, np.sqrt(ay*ay + az*az)) * _R2D
    rr, pr = roll*_D2R, pitch*_D2R
    mxc = mx*np.cos(pr) + mz*np.sin(pr)
    myc = mx*np.sin(rr)*np.sin(pr) + my*np.cos(rr) - mz*np.sin(rr)*np.cos(pr)
    yaw = np.arctan2(-myc, mxc) * _R2D
    if yaw < 0: yaw += 360.0
    return _from_euler(roll, pitch, -yaw)


def compute_R(roll_path: str, pitch_path: str, yaw_path: str,
              percentile: float = 70.0) -> np.ndarray:
    """Calcula R (IMU frame → frame da prótese) a partir de 3 runs de calibração."""
    gcols = ['imu_gx', 'imu_gy', 'imu_gz']
    specs = [('ROLL', roll_path), ('PITCH', pitch_path), ('YAW', yaw_path)]
    rows = []
    for _label, path in specs:
        df = pd.read_csv(path);  df.columns = df.columns.str.strip()
        g   = df[gcols].values.astype(float)
        mag = np.linalg.norm(g, axis=1)
        active = g[mag > np.percentile(mag, percentile)]
        _, _, Vt = np.linalg.svd(active - active.mean(axis=0))
        axis_dir = Vt[0]
        if np.dot(axis_dir, active.mean(axis=0)) < 0:
            axis_dir = -axis_dir
        rows.append(axis_dir / np.linalg.norm(axis_dir))
    R_raw = np.array(rows)
    U, _, Vt = np.linalg.svd(R_raw)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1;  R = U @ Vt
    return R


def apply_imu_correction(df: pd.DataFrame, R: np.ndarray,
                         sr: int = 1, sp: int = 1, sy: int = 1) -> pd.DataFrame:
    """Re-corre o filtro quaternião com os eixos rodados por R. Adiciona pro_roll/pitch/yaw."""
    ax = df['imu_ax'].values * _ACCEL_SCALE
    ay = df['imu_ay'].values * _ACCEL_SCALE
    az = df['imu_az'].values * _ACCEL_SCALE
    gx = df['imu_gx'].values * _GYRO_SCALE - _GYRO_BIAS[0]
    gy = df['imu_gy'].values * _GYRO_SCALE - _GYRO_BIAS[1]
    gz = df['imu_gz'].values * _GYRO_SCALE - _GYRO_BIAS[2]
    mx = (df['imu_mx'].values * _MAG_SCALE - _MAG_OFFSET[0]) * _MAG_SCALE_C[0]
    my = (df['imu_my'].values * _MAG_SCALE - _MAG_OFFSET[1]) * _MAG_SCALE_C[1]
    mz =-(df['imu_mz'].values * _MAG_SCALE - _MAG_OFFSET[2]) * _MAG_SCALE_C[2]

    ar = R @ np.vstack([ax, ay, az])
    gr = R @ np.vstack([gx, gy, gz])
    mr = R @ np.vstack([mx, my, mz])
    ax_r, ay_r, az_r = ar
    gx_r, gy_r, gz_r = gr
    mx_r, my_r, mz_r = mr

    elapsed = (df['sample_us'].values - df['sample_us'].values[0]) / 1e6
    n = len(df)
    rolls, pitches, yaws = np.zeros(n), np.zeros(n), np.zeros(n)
    q = None

    for i in range(n):
        dt = elapsed[i] - elapsed[i-1] if i > 0 else 0.0
        if q is None:
            q = _qnorm(_ref_q(ax_r[i], ay_r[i], az_r[i], mx_r[i], my_r[i], mz_r[i]))
            ro, pi, ya = _to_euler(q)
            rolls[i], pitches[i], yaws[i] = ro, pi, (360 - ya) % 360
            continue
        gxd, gyd, gzd = gx_r[i]*_D2R, gy_r[i]*_D2R, gz_r[i]*_D2R
        ang = np.sqrt(gxd*gxd + gyd*gyd + gzd*gzd) * dt
        if ang > 1e-4:
            s = np.sin(ang * 0.5) / ang
            q_g = np.array([np.cos(ang*0.5), gxd*dt*s, gyd*dt*s, gzd*dt*s])
        else:
            q_g = np.array([1., 0., 0., 0.])
        q_int  = _qnorm(_qmul(q, q_g))
        q_ref  = _qnorm(_ref_q(ax_r[i], ay_r[i], az_r[i], mx_r[i], my_r[i], mz_r[i]))
        gmag   = np.sqrt(gx_r[i]**2 + gy_r[i]**2 + gz_r[i]**2)
        q      = q_int if gmag > _FAST_ROT_THR else _qnorm(_slerp(q_ref, q_int, _ALPHA))
        ro, pi, ya = _to_euler(q)
        rolls[i], pitches[i], yaws[i] = ro, pi, (360 - ya) % 360

    out = df.copy()
    out['pro_roll']  = sr * rolls
    out['pro_pitch'] = sp * pitches
    out['pro_yaw']   = sy * yaws
    return out


# ══════════════════════════════════════════════════════════════════════════
#  3-D model da prótese (forma de taco de golfe)
# ══════════════════════════════════════════════════════════════════════════

def _euler_to_R(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """ZYX Euler → matriz de rotação 3×3."""
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _cyl(r: float, z0: float, z1: float, n: int = 22):
    """Superfície cilíndrica parametrizada."""
    t = np.linspace(0, 2 * np.pi, n)
    Z = np.array([z0, z1])
    T, Zm = np.meshgrid(t, Z)
    return r * np.cos(T), r * np.sin(T), Zm


def _rot(R, X, Y, Z):
    pts = R @ np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    return pts[0].reshape(X.shape), pts[1].reshape(Y.shape), pts[2].reshape(Z.shape)


def draw_prosthetic(ax, roll: float, pitch: float, yaw: float,
                    label: str | None = None,
                    R_base: np.ndarray | None = None) -> None:
    """Redesenha o modelo 3D da prótese com a orientação dada.

    R_base é uma rotação fixa da geometria (pose de repouso do modelo) aplicada
    ANTES da orientação dos sensores — serve para alinhar o desenho com a
    realidade (pé/cabeça no sentido certo).
    """
    ax.cla()
    R = _euler_to_R(roll, pitch, yaw)
    if R_base is not None:
        R = R @ R_base

    # -- haste (shaft) -------------------------------------------------------
    X, Y, Z = _cyl(0.038, -0.50, 0.44)
    ax.plot_surface(*_rot(R, X, Y, Z), color="#b0b0b0", alpha=0.92,
                    linewidth=0, shade=True)

    # -- punho (grip) — secção superior mais larga ---------------------------
    X, Y, Z = _cyl(0.065, 0.41, 0.50)
    ax.plot_surface(*_rot(R, X, Y, Z), color="#333333", alpha=1.0,
                    linewidth=0, shade=True)

    # -- cabeça (head / pad) — caixa achatada na base -----------------------
    #    x ∈ [-0.30, 0.08], y ∈ [-0.04, 0.04], z ∈ {-0.50, -0.44}
    for y0, y1 in [(-0.04, 0.04)]:
        for z_face in [-0.50, -0.44]:
            xv = np.array([[-0.30, 0.08], [-0.30, 0.08]])
            yv = np.array([[y0, y0], [y1, y1]])
            zv = np.full_like(xv, z_face)
            ax.plot_surface(*_rot(R, xv, yv, zv), color="#787878",
                            alpha=0.95, linewidth=0, shade=True)

    for x0, x1 in [(-0.30, -0.30), (0.08, 0.08)]:   # lados esquerdo/direito
        yv = np.array([[-0.04, -0.04], [0.04, 0.04]])
        xv = np.full_like(yv, x0)
        zv = np.array([[-0.50, -0.44], [-0.50, -0.44]])
        ax.plot_surface(*_rot(R, xv, yv, zv), color="#909090",
                        alpha=0.90, linewidth=0, shade=True)

    for y_face in [-0.04, 0.04]:                      # frente/trás
        xv = np.array([[-0.30, 0.08], [-0.30, 0.08]])
        zv = np.array([[-0.50, -0.50], [-0.44, -0.44]])
        yv = np.full_like(xv, y_face)
        ax.plot_surface(*_rot(R, xv, yv, zv), color="#a0a0a0",
                        alpha=0.88, linewidth=0, shade=True)

    # -- triedro de referência (X=vermelho, Y=verde, Z=azul) -----------------
    for vec, col in [([0.7, 0, 0], "r"), ([0, 0.7, 0], "g"), ([0, 0, 0.7], "b")]:
        vr = R @ np.array(vec)
        ax.plot([0, vr[0]], [0, vr[1]], [0, vr[2]], color=col,
                linewidth=1.0, alpha=0.45)

    lim = 0.75
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xlabel("X", fontsize=8, labelpad=0)
    ax.set_ylabel("Y", fontsize=8, labelpad=0)
    ax.set_zlabel("Z", fontsize=8, labelpad=0)
    angle_txt = f"roll {roll:+.1f}°   pitch {pitch:+.1f}°   yaw {yaw:+.1f}°"
    title = f"{label}\n{angle_txt}" if label else angle_txt
    ax.set_title(title, fontsize=8, pad=4)
    ax.tick_params(labelsize=7, pad=1)


# ══════════════════════════════════════════════════════════════════════════
#  Botão de toggle com cor
# ══════════════════════════════════════════════════════════════════════════

class ToggleBtn(QPushButton):
    def __init__(self, text: str, color: str, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.setFixedHeight(26)
        self._color = color
        self._refresh_style()
        self.toggled.connect(lambda _: self._refresh_style())

    def _refresh_style(self):
        if self.isChecked():
            self.setStyleSheet(
                f"QPushButton {{ background: {self._color}; color: white; "
                "border: none; border-radius: 4px; "
                "padding: 2px 10px; font-weight: bold; }"
            )
        else:
            self.setStyleSheet(
                "QPushButton { background: #ddd; color: #999; "
                "border: 1px solid #bbb; border-radius: 4px; padding: 2px 10px; }"
            )


# ══════════════════════════════════════════════════════════════════════════
#  Painel lateral – browser de ficheiros (árvore estilo VSCode)
# ══════════════════════════════════════════════════════════════════════════

def _fmt_size(path: Path) -> str:
    kb = path.stat().st_size // 1024
    return f"{kb} KB" if kb < 1024 else f"{kb / 1024:.1f} MB"


class FileBrowserPanel(QWidget):
    file_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(220)
        self.setMaximumWidth(300)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.setSpacing(4)

        hdr = QLabel("  Explorer")
        hdr.setFont(QFont("", 10, QFont.Weight.Bold))
        hdr.setStyleSheet(
            "color:#555; background:#f0f0f0; padding:4px 0px;"
            "border-bottom:1px solid #d0d0d0; letter-spacing:1px;"
        )
        lay.addWidget(hdr)

        # Árvore principal
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setAnimated(True)
        self._tree.setIndentation(16)
        self._tree.setUniformRowHeights(True)
        # padding apenas — seleção e cores ficam com o sistema (compatível dark/light)
        self._tree.setStyleSheet("QTreeWidget::item { padding: 3px 2px; }")
        self._tree.itemDoubleClicked.connect(self._on_double_click)
        lay.addWidget(self._tree, stretch=1)

        # Botões
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(4, 0, 4, 0)
        for label, slot in [("Refresh", self.refresh),
                             ("Other…",    self._open_dialog)]:
            btn = QPushButton(label)
            btn.setFixedHeight(24)
            btn.setStyleSheet("font-size:10px;")
            btn.clicked.connect(slot)
            btn_row.addWidget(btn)
        lay.addLayout(btn_row)

        self._build_tree()

    # ── ícones do sistema ───────────────────────────────────────────────────

    def _icon_dir(self):
        return self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)

    def _icon_file(self):
        return self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon)

    # ── construção da árvore ────────────────────────────────────────────────

    def _root_font(self) -> QFont:
        f = QFont()
        f.setBold(True)
        return f

    def _build_tree(self):
        self._tree.clear()

        # Downloads — só CSVs directamente na pasta (não recursivo)
        dl_item = QTreeWidgetItem(self._tree, ["Downloads"])
        dl_item.setIcon(0, self._icon_dir())
        dl_item.setData(0, Qt.ItemDataRole.UserRole, None)
        dl_item.setFont(0, self._root_font())
        try:
            if DOWNLOADS_DIR.exists():
                for f in sorted(DOWNLOADS_DIR.glob("*.csv"), key=lambda f: f.name):
                    self._add_file(dl_item, f)
        except PermissionError:
            self._add_permission_hint(dl_item)
        dl_item.setExpanded(False)

        # Test Data — recursivo com sub-pastas
        td_item = QTreeWidgetItem(self._tree, ["Test Data"])
        td_item.setIcon(0, self._icon_dir())
        td_item.setData(0, Qt.ItemDataRole.UserRole, None)
        td_item.setFont(0, self._root_font())
        try:
            if TESTDATA_DIR.exists():
                self._add_dir_children(td_item, TESTDATA_DIR)
        except PermissionError:
            self._add_permission_hint(td_item)
        td_item.setExpanded(False)

    def _add_permission_hint(self, parent: QTreeWidgetItem):
        """Mostra uma dica quando o macOS bloqueia o acesso à pasta (TCC)."""
        hint = QTreeWidgetItem(
            parent, ["⚠ macOS bloqueou o acesso — concede 'Full Disk Access'"])
        hint.setData(0, Qt.ItemDataRole.UserRole, None)

    def _add_dir_children(self, parent: QTreeWidgetItem, directory: Path):
        """Adiciona sub-pastas (expansíveis) e depois os CSVs desta pasta."""
        subdirs = sorted(
            (d for d in directory.iterdir() if d.is_dir()),
            key=lambda d: d.name,
        )
        for subdir in subdirs:
            folder = QTreeWidgetItem(parent, [subdir.name])
            folder.setIcon(0, self._icon_dir())
            folder.setData(0, Qt.ItemDataRole.UserRole, None)
            folder.setFont(0, self._root_font())
            self._add_dir_children(folder, subdir)
            folder.setExpanded(False)  # recolhido por defeito

        # CSVs desta pasta, ordenados pelo nome (= ordem temporal pelo nosso formato)
        csvs = sorted(
            (f for f in directory.iterdir() if f.is_file() and f.suffix == ".csv"),
            key=lambda f: f.name,
        )
        for f in csvs:
            self._add_file(parent, f)

    def _add_file(self, parent: QTreeWidgetItem, path: Path):
        size = _fmt_size(path)
        item = QTreeWidgetItem(parent, [f"{path.name}   {size}"])
        item.setIcon(0, self._icon_file())
        item.setData(0, Qt.ItemDataRole.UserRole, str(path))
        item.setToolTip(0, str(path))

    # ── interacção ──────────────────────────────────────────────────────────

    def _on_double_click(self, item: QTreeWidgetItem, _col: int):
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path:
            self.file_selected.emit(path)

    def _open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV", str(_BASE), "CSV (*.csv);;All (*)"
        )
        if path:
            self.file_selected.emit(path)

    def refresh(self):
        # guarda quais as pastas expandidas para restaurar depois
        expanded: set[str] = set()
        it = QTreeWidgetItem()
        root = self._tree.invisibleRootItem()
        stack = [root.child(i) for i in range(root.childCount())]
        while stack:
            node = stack.pop()
            if node and node.isExpanded():
                expanded.add(node.text(0))
            if node:
                stack += [node.child(i) for i in range(node.childCount())]

        self._build_tree()

        # restaura estado de expansão
        stack = [root.child(i) for i in range(root.childCount())]
        while stack:
            node = stack.pop()
            if node and node.text(0) in expanded:
                node.setExpanded(True)
            if node:
                stack += [node.child(i) for i in range(node.childCount())]


# ── botão de modo: Pontos ↔ Linha (partilhado por todos os tabs) ──────────

class _FitBtn(QPushButton):
    """Botão que alterna entre 'Linha' (mostrar fit suavizado) e 'Pontos' (scatter)."""
    def __init__(self, parent=None):
        super().__init__("Line", parent)
        self.setCheckable(True); self.setChecked(False)
        self.setFixedHeight(26); self.setFixedWidth(70)
        self.setStyleSheet(
            "QPushButton { padding:2px 6px; border:1px solid #999; "
            "border-radius:4px; background:#f5f5f5; }"
            "QPushButton:checked { background:#444; color:white; border-color:#333; }"
        )


# ══════════════════════════════════════════════════════════════════════════
#  Tab – Célula de Carga  (kg + raw ADC, scatter + fit, toggles)
# ══════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════
#  Extração de features de marcha (segmentação de passadas a partir da força)
# ══════════════════════════════════════════════════════════════════════════

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # np>=2 renomeou


def _regularity_index(t: np.ndarray, f: np.ndarray) -> float:
    """Índice de regularidade do passo: altura do primeiro pico dominante da
    autocorrelação do sinal de força (reamostrado uniforme). Valor em [0,1];
    quanto maior, mais periódica/repetível é a marcha."""
    if len(f) < 20:
        return float("nan")
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        return float("nan")
    tu = np.arange(t[0], t[-1], dt)
    fu = np.interp(tu, t, f)
    fu = fu - fu.mean()
    if np.allclose(fu, 0):
        return float("nan")
    ac = np.correlate(fu, fu, mode="full")[len(fu) - 1:]
    ac = ac / ac[0]
    dac = np.diff(ac)
    up = np.flatnonzero((dac[:-1] < 0) & (dac[1:] >= 0)) + 1  # 1.º mínimo local
    if len(up) == 0:
        return float("nan")
    return float(np.clip(ac[up[0]:].max(), 0.0, 1.0))


def _sparc(speed: np.ndarray, fs: float, fc: float = 10.0,
           amp_th: float = 0.05) -> float:
    """Spectral Arc Length — métrica de suavidade do movimento, invariante de
    escala (Balasubramanian et al. 2015). Valores menos negativos = mais suave.
    `speed` é um perfil de velocidade (aqui a magnitude do giroscópio)."""
    speed = np.asarray(speed, float)
    if len(speed) < 10 or not np.any(speed):
        return float("nan")
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + 2))
    freq = np.arange(nfft) * (fs / nfft)
    mag = np.abs(np.fft.fft(speed, nfft))
    if mag.max() == 0:
        return float("nan")
    mag = mag / mag.max()
    sel = freq <= fc
    freq, mag = freq[sel], mag[sel]
    inx = np.flatnonzero(mag >= amp_th)
    if len(inx) < 2:
        return float("nan")
    freq, mag = freq[inx[0]:inx[-1] + 1], mag[inx[0]:inx[-1] + 1]
    df_ = np.diff(freq) / (freq[-1] - freq[0])
    dm = np.diff(mag)
    return float(-np.sum(np.sqrt(df_ ** 2 + dm ** 2)))


def compute_gait_features(df: pd.DataFrame) -> dict | None:
    """Segmenta passadas pelos picos da célula de carga e calcula as features
    biomecânicas (força / temporais / variabilidade). Devolve dict com valores
    e dados para visualização, ou None se não houver sinal de força."""
    if "load_cell_est_kg" not in df.columns or "sample_us" not in df.columns:
        return None
    t = (df["sample_us"].to_numpy(float) - df["sample_us"].iloc[0]) / 1e6
    f = df["load_cell_est_kg"].to_numpy(float)
    if len(f) < 20:
        return None

    w = max(3, len(f) // 200)
    fs = pd.Series(f).rolling(w, center=True, min_periods=1).mean().to_numpy()
    base = float(np.percentile(fs, 10))
    top = float(np.percentile(fs, 90))
    rng = top - base
    if rng < 1e-6:
        return None
    contact_thr = base + 0.30 * rng     # limiar de contacto (foot-strike/toe-off)
    peak_thr = base + 0.50 * rng        # uma passada válida tem de o ultrapassar

    above = fs > contact_thr
    raw_segs: list[tuple[int, int]] = []
    in_c, s0 = False, 0
    for i in range(len(fs)):
        if above[i] and not in_c:
            in_c, s0 = True, i
        elif not above[i] and in_c:
            in_c = False
            raw_segs.append((s0, i - 1))
    if in_c:
        raw_segs.append((s0, len(fs) - 1))

    stances: list[dict] = []
    for a, b in raw_segs:
        seg = fs[a:b + 1]
        dur = t[b] - t[a]
        if seg.max() < peak_thr or dur < 0.05 or dur > 2.0:
            continue
        pk = a + int(np.argmax(seg))
        impulse = float(_trapz(np.clip(fs[a:b + 1] - base, 0, None), t[a:b + 1]))
        lr = float((fs[pk] - contact_thr) / max(t[pk] - t[a], 1e-3))
        stances.append({"a": a, "b": b, "pk": pk,
                        "t0": t[a], "t1": t[b], "tp": t[pk],
                        "pvf": float(fs[pk]), "impulse": impulse,
                        "stance": dur, "lr": lr})

    dur_total = float(t[-1] - t[0])
    out: dict = {"t": t, "f": f, "fs": fs, "base": base,
                 "contact_thr": contact_thr, "stances": stances,
                 "dur": dur_total, "ok": len(stances) >= 2}
    if not out["ok"]:
        return out

    pvf = np.array([s["pvf"] for s in stances])
    stance_t = np.array([s["stance"] for s in stances])
    peaks_t = np.array([s["tp"] for s in stances])
    impulse = np.array([s["impulse"] for s in stances])
    lr = np.array([s["lr"] for s in stances])
    stride_t = np.diff(peaks_t)                 # intervalo pico-a-pico
    swing_t = stride_t - stance_t[:-1]
    duty = stance_t[:-1] / stride_t

    def _cv(a: np.ndarray) -> float:
        a = np.asarray(a, float)
        m = a.mean()
        return float(a.std() / m) if m else float("nan")

    out.update({
        "pvf": pvf, "stride_t": stride_t, "stance_t": stance_t,
        "peaks_t": peaks_t, "n_strides": len(stances),
        # força (cinética)
        "PVF_mean": float(pvf.mean()),
        "impulse_mean": float(impulse.mean()),
        "lr_mean": float(lr.mean()),
        "wbr": float(np.mean(pvf > 0.5 * np.median(pvf))),
        # temporais
        "stride_mean": float(stride_t.mean()),
        "stance_mean": float(stance_t.mean()),
        "swing_mean": float(swing_t.mean()),
        "duty_mean": float(duty.mean()),
        "cadence": (len(stances) / dur_total) if dur_total > 0 else float("nan"),
        # variabilidade
        "cv_stride": _cv(stride_t),
        "cv_pvf": _cv(pvf),
        "reg": _regularity_index(t, fs),
        "pvf_ac1": float(np.corrcoef(pvf[:-1], pvf[1:])[0, 1]) if len(pvf) > 2 else float("nan"),
    })

    out.update(_imu_features(df, t, stances, _cv))
    return out


def _imu_features(df: pd.DataFrame, t: np.ndarray, stances: list[dict],
                  _cv) -> dict:
    """Features cinemáticas a partir do IMU, segmentadas pelas mesmas passadas
    (janela pico-a-pico). Devolve {} se não houver colunas IMU."""
    has_rpy = {"roll_deg", "pitch_deg", "yaw_deg"}.issubset(df.columns)
    has_acc = {"imu_ax", "imu_ay", "imu_az"}.issubset(df.columns)
    has_gyr = {"imu_gx", "imu_gy", "imu_gz"}.issubset(df.columns)
    if not (has_rpy or has_acc or has_gyr):
        return {}

    def _uw(col):  # unwrap para evitar saltos de ±180° no cálculo de ROM
        return np.degrees(np.unwrap(np.radians(df[col].to_numpy(float))))
    roll = _uw("roll_deg") if has_rpy else None
    pitch = _uw("pitch_deg") if has_rpy else None
    am = (np.linalg.norm(df[["imu_ax", "imu_ay", "imu_az"]].to_numpy(float), axis=1)
          if has_acc else None)
    gm = (np.linalg.norm(df[["imu_gx", "imu_gy", "imu_gz"]].to_numpy(float), axis=1)
          if has_gyr else None)

    pk_idx = [s["pk"] for s in stances]
    pitch_rom, roll_rom, gpeak, arms, pitch_pk = [], [], [], [], []
    for i in range(len(stances) - 1):
        lo, hi = pk_idx[i], pk_idx[i + 1]      # janela de uma passada
        if hi - lo < 3:
            continue
        if pitch is not None:
            pitch_rom.append(float(np.ptp(pitch[lo:hi])))
            roll_rom.append(float(np.ptp(roll[lo:hi])))
            pitch_pk.append(float(pitch[stances[i]["pk"]]))
        if gm is not None:
            gpeak.append(float(gm[lo:hi].max()))
        if am is not None:
            arms.append(float(np.sqrt(np.mean(am[lo:hi] ** 2))))

    res: dict = {"has_imu": True}
    arr = lambda L: np.array(L, float) if L else np.array([])
    if pitch is not None and pitch_rom:
        res.update({"pitch_rom": arr(pitch_rom), "roll_rom": arr(roll_rom),
                    "pitch_pk": arr(pitch_pk),
                    "pitch_rom_mean": float(np.mean(pitch_rom)),
                    "roll_rom_mean": float(np.mean(roll_rom)),
                    "cv_pitch_rom": _cv(pitch_rom),
                    "pitch_pk_mean": float(np.mean(pitch_pk)),
                    "cv_pitch_pk": _cv(pitch_pk)})
    if gm is not None and gpeak:
        res.update({"gpeak": arr(gpeak), "gpeak_mean": float(np.mean(gpeak)),
                    "cv_gpeak": _cv(gpeak)})
    if am is not None and arms:
        res.update({"arms_mean": float(np.mean(arms)), "cv_arms": _cv(arms)})

    # suavidade do movimento (SPARC) sobre a magnitude do giroscópio
    if gm is not None and len(t) > 20:
        dt = np.median(np.diff(t))
        if np.isfinite(dt) and dt > 0:
            tu = np.arange(t[0], t[-1], dt)
            gu = np.interp(tu, t, gm)
            res["sparc"] = _sparc(gu, fs=1.0 / dt)
    return res


# ══════════════════════════════════════════════════════════════════════════
#  Tab – Features de Marcha (segmentação + overlays + resumo)
# ══════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════
#  Tab – IMU genérico (acelerómetro | giroscópio | magnetómetro | orientação)
# ══════════════════════════════════════════════════════════════════════════

def _smooth(series, w: int) -> np.ndarray:
    return pd.Series(series).rolling(w, center=True, min_periods=1).mean().to_numpy()


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


# ══════════════════════════════════════════════════════════════════════════
#  Tab – Orientação 3D
#    RPY (sem zoom, só scrub) + Kg sincronizado + modelo 3D + play
# ══════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════
#  Tab – Calibração IMU  (calcula R, visualiza runs, exporta para firmware)
# ══════════════════════════════════════════════════════════════════════════

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

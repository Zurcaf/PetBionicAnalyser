"""Widgets reutilizáveis: botões de toggle/fit e browser de ficheiros."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QPushButton, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QSizePolicy,
    QTreeWidget, QTreeWidgetItem, QStyle, QFileDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from .config import C, DOWNLOADS_DIR, TESTDATA_DIR, _BASE


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



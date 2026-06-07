"""Constantes de configuração: caminhos, cores e estilo."""
from __future__ import annotations

import os
from pathlib import Path

# A app vive separada do projeto, mas lê os dados do projeto principal. O caminho
# pode ser configurado por PETBIONIC_ROOT; por omissão assume ~/Desktop/petBionic.
_BASE = Path(os.environ.get("PETBIONIC_ROOT", Path.home() / "Desktop" / "petBionic"))
TESTDATA_DIR  = _BASE / "TestData"
DOWNLOADS_DIR = Path.home() / "Downloads"

# estado de runtime guardado na raiz do repo (src/petbionic_analyser/ → parents[2])
_STATE_DIR    = Path(__file__).resolve().parents[2]
_CALIB_FILE   = _STATE_DIR / "imu_calibration_R.json"
_RUNS_FILE    = _STATE_DIR / "calib_runs.json"
_MODEL_FILE   = _STATE_DIR / "model_orientation.json"

# paleta de cores partilhada
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

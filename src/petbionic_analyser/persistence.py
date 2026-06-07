"""Persistência de estado: calibração R, runs de calibração, pose do modelo 3D."""
from __future__ import annotations

import json
import numpy as np

from .config import _CALIB_FILE, _RUNS_FILE, _MODEL_FILE


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

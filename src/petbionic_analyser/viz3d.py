"""Desenho do modelo 3D da prótese (forma de taco de golfe)."""
from __future__ import annotations

import numpy as np

from .imu import _euler_to_R


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



#!/usr/bin/env python3
"""
Gera figuras ilustrativas das features biomecânicas para o capítulo de
Metodologia da tese.

Estilo misto:
  A) feat_gait_cycle      — diagrama didático (ciclo de marcha idealizado)
  B) feat_force_real      — features de força anotadas sobre marcha real
  C) feat_variability_real— variabilidade (PVF / tempo de passada / regularidade)

Reutiliza compute_gait_features() do csv_analyzer. Correr:
    QT_QPA_PLATFORM=offscreen python scripts/make_feature_figures.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from csv_analyzer import compute_gait_features  # noqa: E402

# ── caminhos ────────────────────────────────────────────────────────────────
# A app vive em ~/Developer/PetBionic Analyser, mas as figuras vão para a tese e
# os dados vêm do projeto principal. Configurável por PETBIONIC_ROOT.
BASE = Path(os.environ.get("PETBIONIC_ROOT", Path.home() / "Desktop" / "petBionic"))
OUT = (BASE / "PIC2" / "Relatorio_PIC2_102815_Afonso_Cruz"
       / "05_figures" / "05_methodology")
DATA_DIR = BASE / "TestData" / "Round1dia28_cleaned" / "AndamentoTestes28"
DEFAULT_RUN = DATA_DIR / "20260528_12h39_run02.csv"   # marcha regular

# ── estilo ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8, "axes.grid": True, "grid.linestyle": "--",
    "grid.alpha": 0.3, "figure.dpi": 110,
})
CK, CR, CG, CB = "#1f77b4", "#e41a1c", "#4daf4a", "#555555"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}.pdf / .png")


# ════════════════════════════════════════════════════════════════════════════
#  A) Diagrama didático do ciclo de marcha
# ════════════════════════════════════════════════════════════════════════════
def fig_gait_cycle():
    t = np.linspace(0, 2.0, 800)

    def pulse(t, t0, width, amp):
        x = (t - t0) / width
        y = amp * np.exp(-0.5 * (x / 0.42) ** 2)        # corpo do stance
        y += 0.18 * amp * np.exp(-0.5 * ((x - 0.55) / 0.28) ** 2)  # 2.º pico
        return np.where(np.abs(x) < 1.15, y, 0.0)

    stride = 1.0
    f = pulse(t, 0.45, 0.32, 4.0) + pulse(t, 0.45 + stride, 0.32, 4.0)

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.plot(t, f, color=CK, lw=2, zorder=3)

    # first cycle: annotations
    fs1, to1 = 0.18, 0.78          # approximate foot-strike / toe-off
    pk1 = 0.45
    fs2 = fs1 + stride
    ax.axvspan(fs1, to1, color=CG, alpha=0.15, zorder=0)
    ax.text((fs1 + to1) / 2, 0.35, "STANCE", ha="center", va="bottom",
            fontsize=9, color="#2e7d32")
    ax.axvspan(to1, fs2, color="#bbbbbb", alpha=0.18, zorder=0)
    ax.text((to1 + fs2) / 2, 0.35, "SWING", ha="center", va="bottom",
            fontsize=9, color="#555")

    for x, lab in [(fs1, "foot-strike"), (to1, "toe-off"), (fs2, "foot-strike")]:
        ax.axvline(x, color=CB, ls=":", lw=1.2, zorder=2)
        ax.text(x, 4.55, lab, rotation=90, ha="right", va="top",
                fontsize=7.5, color=CB)

    # PVF
    pvf = float(f[np.argmin(np.abs(t - pk1))])
    ax.plot(pk1, pvf, "o", color=CR, ms=8, zorder=4)
    ax.annotate("PVF\n(peak force)", (pk1, pvf), (pk1 + 0.18, pvf + 0.25),
                fontsize=8.5, color=CR,
                arrowprops=dict(arrowstyle="->", color=CR))

    # impulse = area under the stance phase
    m = (t >= fs1) & (t <= to1)
    ax.fill_between(t[m], 0, f[m], color=CR, alpha=0.12, zorder=1)
    ax.text(0.40, 1.5, "Impulse\n= ∫ F dt", ha="center", fontsize=8.5, color=CR)

    # loading rate = FS→peak slope
    ax.annotate("", (pk1, pvf), (fs1, 0.05),
                arrowprops=dict(arrowstyle="-", color="#ff7f0e", lw=2))
    ax.text(fs1 + 0.02, pvf * 0.62, " loading\n rate", fontsize=8, color="#d35400")

    # brackets: stride time and stance time
    def bracket(x0, x1, y, label, color):
        ax.annotate("", (x0, y), (x1, y),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=1.3))
        ax.text((x0 + x1) / 2, y - 0.32, label, ha="center", va="top",
                fontsize=8.5, color=color)
    bracket(fs1, fs2, -0.8, "stride time", CB)
    bracket(fs1, to1, -1.7, "stance time", "#2e7d32")
    ax.text((to1 + fs2) / 2, -1.7 - 0.32, "duty factor = stance / stride",
            ha="center", va="top", fontsize=7.5, color="#777", style="italic")

    ax.set_xlabel("Time")
    ax.set_ylabel("Vertical force")
    ax.set_title("Gait cycle and biomechanical features (schematic)")
    ax.set_ylim(-2.6, 5.2)
    ax.set_yticks([0]); ax.set_xticks([])
    ax.grid(False)
    save(fig, "feat_gait_cycle")


# ════════════════════════════════════════════════════════════════════════════
#  B) Features de força sobre dados reais
# ════════════════════════════════════════════════════════════════════════════
def fig_force_real(F: dict, window=(0, 6)):
    t, fs = F["t"], F["fs"]
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    ax.plot(t, fs, color=CK, lw=1.3, label="Force (kg)", zorder=3)
    ax.axhline(F["contact_thr"], color="#999", ls="--", lw=1,
               label="contact threshold")
    ax.axhline(F["PVF_mean"], color=CR, ls=":", lw=1, label="mean PVF")

    first = True
    for s in F["stances"]:
        ax.axvspan(s["t0"], s["t1"], color=CG, alpha=0.13,
                   label="stance" if first else None, zorder=0)
        first = False
    ax.scatter(F["peaks_t"], F["pvf"], s=34, color=CR, zorder=5,
               label="PVF (peak)")

    # highlight impulse + loading rate on one stride within the window
    target = next((s for s in F["stances"] if window[0] < s["t0"] < window[1] - 1), None)
    if target:
        m = (t >= target["t0"]) & (t <= target["t1"])
        ax.fill_between(t[m], F["base"], fs[m], color=CR, alpha=0.10, zorder=1)
        ax.annotate("impulse", ((target["t0"] + target["t1"]) / 2, F["base"] + 0.4),
                    fontsize=8, color=CR, ha="center")
        ax.annotate("", (target["tp"], fs[np.argmin(np.abs(t - target["tp"]))]),
                    (target["t0"], F["contact_thr"]),
                    arrowprops=dict(arrowstyle="-", color="#ff7f0e", lw=2))
        ax.text(target["t0"], F["PVF_mean"] * 0.6, " loading\n rate",
                fontsize=7.5, color="#d35400")

    ax.set_xlim(*window)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Force (kg)")
    ax.set_title("Force features on real gait (run02, 28 May 2026)")
    ax.legend(ncol=3, loc="upper right", framealpha=0.9)
    save(fig, "feat_force_real")


# ════════════════════════════════════════════════════════════════════════════
#  C) Variabilidade sobre dados reais
# ════════════════════════════════════════════════════════════════════════════
def fig_variability_real(F: dict):
    fig = plt.figure(figsize=(9.2, 3.4))
    gs = fig.add_gridspec(1, 3, wspace=0.32)
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))

    def bars(ax, vals, color, title, cv, unit):
        idx = np.arange(1, len(vals) + 1)
        ax.bar(idx, vals, color=color, alpha=0.75, width=0.82)
        m, sd = float(np.mean(vals)), float(np.std(vals))
        ax.axhspan(m - sd, m + sd, color=color, alpha=0.13)
        ax.axhline(m, color=color, ls="--", lw=1.2)
        ax.set_title(f"{title}\nCV = {100 * cv:.1f} %", fontsize=9.5)
        ax.set_xlabel("stride"); ax.set_ylabel(unit)

    bars(ax1, F["pvf"], CR, "PVF per stride", F["cv_pvf"], "kg")
    bars(ax2, F["stride_t"], CK, "Stride time", F["cv_stride"], "s")

    # regularidade: autocorrelação do sinal de força
    t, fsig = F["t"], F["fs"]
    dt = np.median(np.diff(t))
    tu = np.arange(t[0], t[-1], dt)
    fu = np.interp(tu, t, fsig); fu -= fu.mean()
    ac = np.correlate(fu, fu, "full")[len(fu) - 1:]
    ac /= ac[0]
    lags = np.arange(len(ac)) * dt
    keep = lags <= 3.0
    ax3.plot(lags[keep], ac[keep], color=CG, lw=1.6)
    ax3.axhline(0, color="#aaa", lw=0.8)
    # marca o pico de regularidade
    dac = np.diff(ac)
    up = np.flatnonzero((dac[:-1] < 0) & (dac[1:] >= 0)) + 1
    if len(up):
        pk = up[0] + int(np.argmax(ac[up[0]:up[0] + int(1.5 / dt)]))
        ax3.plot(lags[pk], ac[pk], "o", color=CR, ms=7)
        ax3.annotate(f"regularity\n= {ac[pk]:.2f}", (lags[pk], ac[pk]),
                     (lags[pk] + 0.3, ac[pk] + 0.05), fontsize=8, color=CR,
                     arrowprops=dict(arrowstyle="->", color=CR))
    ax3.set_title("Regularity\n(autocorrelation)", fontsize=9.5)
    ax3.set_xlabel("lag (s)"); ax3.set_ylabel("autocorr.")

    fig.suptitle("Variability features on real gait (run02, 28 May 2026)",
                 fontsize=10.5, y=1.04)
    save(fig, "feat_variability_real")


# ════════════════════════════════════════════════════════════════════════════
#  D) Features cinemáticas (IMU) sobre dados reais
# ════════════════════════════════════════════════════════════════════════════
def fig_kinematic_real(F: dict, df: pd.DataFrame, window=(0, 6)):
    if not F.get("has_imu"):
        print("  (sem IMU — figura cinemática ignorada)")
        return
    t = F["t"]
    pitch = np.degrees(np.unwrap(np.radians(df["pitch_deg"].to_numpy(float))))
    gm = np.linalg.norm(df[["imu_gx", "imu_gy", "imu_gz"]].to_numpy(float), axis=1)

    fig = plt.figure(figsize=(9.2, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.5, wspace=0.3)
    axT = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    # painel superior: pitch + magnitude do giroscópio
    axT.plot(t, pitch, color=CG, lw=1.4, label="Pitch (°)", zorder=3)
    axg = axT.twinx()
    axg.plot(t, gm, color="#ff7f0e", lw=0.9, alpha=0.55, label="Gyro magnitude")
    axg.set_ylabel("angular rate (counts)", color="#cc6600", fontsize=9)
    axg.tick_params(axis="y", labelcolor="#cc6600")
    first = True
    for s in F["stances"]:
        axT.axvspan(s["t0"], s["t1"], color=CB, alpha=0.08,
                    label="stance" if first else None, zorder=0)
        first = False

    # anotação de ROM do pitch numa passada dentro da janela
    target = next((s for s in F["stances"] if window[0] < s["t0"] < window[1] - 1), None)
    if target:
        m = (t >= target["t0"]) & (t <= target["t1"])
        pmin, pmax = float(pitch[m].min()), float(pitch[m].max())
        xa = (target["t0"] + target["t1"]) / 2
        axT.annotate("", (xa, pmax), (xa, pmin),
                     arrowprops=dict(arrowstyle="<->", color=CR, lw=1.6))
        axT.text(xa + 0.12, (pmin + pmax) / 2, "pitch ROM",
                 fontsize=8.5, color=CR, va="center")

    axT.set_xlim(*window)
    axT.set_xlabel("Time (s)"); axT.set_ylabel("Pitch (°)", color="#2e7d32")
    axT.tick_params(axis="y", labelcolor="#2e7d32")
    axT.set_title("Kinematic (IMU) features on real gait (run02, 28 May 2026)")
    h1, l1 = axT.get_legend_handles_labels()
    h2, l2 = axg.get_legend_handles_labels()
    axT.legend(h1 + h2, l1 + l2, fontsize=7.5, ncol=3, loc="upper right",
               framealpha=0.9)

    # paineis inferiores: variabilidade cinemática por passada
    def bars(ax, vals, color, title, cv, unit):
        idx = np.arange(1, len(vals) + 1)
        ax.bar(idx, vals, color=color, alpha=0.75, width=0.82)
        m, sd = float(np.mean(vals)), float(np.std(vals))
        ax.axhspan(m - sd, m + sd, color=color, alpha=0.13)
        ax.axhline(m, color=color, ls="--", lw=1.2)
        ax.set_title(f"{title}   CV = {100 * cv:.1f} %", fontsize=9.5)
        ax.set_xlabel("stride"); ax.set_ylabel(unit)

    bars(ax1, F["pitch_rom"], CG, "Pitch ROM per stride", F["cv_pitch_rom"], "°")
    bars(ax2, F["gpeak"], "#ff7f0e", "Peak angular rate per stride",
         F["cv_gpeak"], "counts")
    sp = F.get("sparc")
    if sp is not None and np.isfinite(sp):
        fig.text(0.5, -0.02,
                 f"Movement smoothness (SPARC) = {sp:.2f}   "
                 "(less negative = smoother; rises with adaptation)",
                 ha="center", fontsize=8.5, color="#444", style="italic")
    save(fig, "feat_kinematic_real")


def main():
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RUN
    df = pd.read_csv(run); df.columns = df.columns.str.strip()
    F = compute_gait_features(df)
    if F is None or not F["ok"]:
        sys.exit(f"Sem passadas suficientes em {run.name}")
    print(f"A gerar figuras a partir de {run.name} "
          f"({F['n_strides']} passadas) -> {OUT}")
    fig_gait_cycle()
    fig_force_real(F)
    fig_variability_real(F)
    fig_kinematic_real(F, df)
    print("Concluído.")


if __name__ == "__main__":
    main()

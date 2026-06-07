"""Extração de features de marcha (cinética / temporais / variabilidade / IMU)."""
from __future__ import annotations

import numpy as np
import pandas as pd


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




def _smooth(series, w: int) -> np.ndarray:
    return pd.Series(series).rolling(w, center=True, min_periods=1).mean().to_numpy()


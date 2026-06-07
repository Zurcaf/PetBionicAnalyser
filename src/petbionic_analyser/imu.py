"""Matemática de orientação: filtro quaternião, Euler ↔ matriz, matriz R."""
from __future__ import annotations

import numpy as np
import pandas as pd

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


def _euler_to_R(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """ZYX Euler → matriz de rotação 3×3."""
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx



def _turns_to_R(turns: dict[str, int]) -> np.ndarray:
    """Constrói a rotação base a partir de voltas de 90° em x, y, z."""
    return _euler_to_R(90.0 * turns.get("x", 0),
                       90.0 * turns.get("y", 0),
                       90.0 * turns.get("z", 0))

"""Synthetic ground-truth generators.

Produces a known trajectory + the *exact* MARG readings a perfect sensor at that
trajectory would report (plus configurable bias & noise). Because we hold the
ground truth, we can compute real error metrics (speed RMSE, attitude RMSE,
drift) instead of just "does it run".

Two placement models:
  * make_walking_sequence  -> foot-worn: clean stance phases (ZUPT-friendly).
  * make_torso_sequence    -> sternum / upper-arm: never truly still; forward
    walking speed with vertical bob + lateral sway and an upright, gently
    oscillating orientation. This is the Polar-H10-style placement and the one
    we care about for accurate *speed*.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from ekf9dof.sensors import Sample
from ekf9dof.filter import GRAVITY

MAG_REF = np.array([0.0, 0.5, -0.8660254])
MAG_REF = MAG_REF / np.linalg.norm(MAG_REF)


def _finalize(t, dt, pos, R_list, rng, gyro_bias, accel_bias,
              gyro_noise, accel_noise, mag_noise, device_id, placement):
    """Given ground-truth position + orientation, derive the exact sensor
    stream a perfect MARG would report and package the ground truth."""
    n = len(t)
    g_n = np.array([0.0, 0.0, -GRAVITY])
    gyro_bias = np.asarray(gyro_bias, float)
    accel_bias = np.asarray(accel_bias, float)

    vel = np.gradient(pos, dt, axis=0)
    acc_nav = np.gradient(vel, dt, axis=0)

    # body-frame angular velocity implied exactly by the attitude profile
    omega = np.zeros((n, 3))
    for k in range(n - 1):
        omega[k] = (R_list[k].inv() * R_list[k + 1]).as_rotvec() / dt
    omega[-1] = omega[-2]

    quat = np.array([r.as_quat() for r in R_list])
    euler = np.array([r.as_euler("xyz", degrees=True) for r in R_list])

    samples = []
    for k in range(n):
        Rm = R_list[k].as_matrix()
        gyro_meas = omega[k] + gyro_bias + rng.normal(0, gyro_noise, 3)
        f_b = Rm.T @ (acc_nav[k] - g_n) + accel_bias + rng.normal(0, accel_noise, 3)
        m_b = Rm.T @ MAG_REF + rng.normal(0, mag_noise, 3)
        samples.append(
            Sample(t=t[k], device_id=device_id, accel=f_b, gyro=gyro_meas, mag=m_b)
        )

    truth = {
        "t": t,
        "pos": pos,
        "vel": vel,
        "quat": quat,
        "euler": euler,
        "speed": np.linalg.norm(vel, axis=1),
        "ground_speed": np.linalg.norm(vel[:, :2], axis=1),
        "gyro_bias": gyro_bias,
        "accel_bias": accel_bias,
        "placement": placement,
    }
    return samples, truth


def make_walking_sequence(
    duration=12.0,
    fs=200.0,
    step_period=1.0,
    stance_frac=0.4,
    stride_speed=1.4,
    lift_height=0.12,
    turn_rate=0.33,
    pitch_amp=0.35,
    roll_amp=0.12,
    seed=0,
    gyro_bias=(0.01, -0.008, 0.005),
    accel_bias=(0.05, -0.03, 0.02),
    gyro_noise=0.004,
    accel_noise=0.03,
    mag_noise=0.01,
    device_id="foot",
):
    """Foot-worn gait: stance (planted, still) then swing (stride + rotation).
    The stance phase is a genuine zero-velocity window, so ZUPT applies.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / fs
    t = np.arange(0.0, duration, dt)
    n = len(t)

    phase = np.mod(t, step_period) / step_period
    stance = phase < stance_frac
    swing = np.clip((phase - stance_frac) / (1.0 - stance_frac), 0.0, 1.0)
    swing_env = np.where(stance, 0.0, (1 - np.cos(2 * np.pi * swing)) / 2)

    psi = np.cumsum(turn_rate * swing_env) * dt
    fwd = np.stack([np.cos(psi), np.sin(psi)], axis=1)

    speed = stride_speed * swing_env
    pos = np.zeros((n, 3))
    pos[:, :2] = np.cumsum(speed[:, None] * fwd, axis=0) * dt
    pos[:, 2] = lift_height * swing_env

    pitch = pitch_amp * swing_env
    roll = roll_amp * np.sin(2 * np.pi * 1.0 * t) * swing_env
    R_list = [Rotation.from_euler("ZYX", [psi[k], pitch[k], roll[k]]) for k in range(n)]

    return _finalize(t, dt, pos, R_list, rng, gyro_bias, accel_bias,
                     gyro_noise, accel_noise, mag_noise, device_id, "foot")


def make_torso_sequence(
    duration=24.0,
    fs=100.0,
    placement="sternum",
    mean_speed=1.30,
    cadence=1.9,          # steps per second
    fore_aft=0.18,        # fore-aft speed modulation (fraction of mean_speed)
    bob_amp=0.030,        # vertical bob amplitude (m)
    sway_amp=0.045,       # lateral sway amplitude (m)
    turn_rate=0.10,       # gentle steady heading change (rad/s)
    rest=3.0,             # seconds standing still at start and end
    ramp=1.5,             # seconds to accelerate/decelerate
    pitch_amp_deg=4.0,    # fore-aft trunk pitch
    roll_amp_deg=3.0,     # lateral trunk lean
    yaw_amp_deg=2.0,      # trunk rotation about vertical
    seed=0,
    gyro_bias=(0.008, -0.006, 0.004),
    accel_bias=(0.04, -0.03, 0.02),
    gyro_noise=0.004,
    accel_noise=0.03,
    mag_noise=0.01,
    device_id="sternum",
):
    """Sternum / upper-arm placement: forward walking with the characteristic
    torso vertical bob (2x/stride) and lateral sway (1x/stride), upright with a
    small oscillating orientation. Never reaches zero velocity -> needs the
    body-planar velocity constraint instead of ZUPT.

    `upper_arm` reuses the same model with larger orientation oscillation to
    approximate arm swing.
    """
    if placement in ("upper_arm", "arm"):
        pitch_amp_deg *= 4.0
        yaw_amp_deg *= 2.0
        fore_aft *= 1.3

    rng = np.random.default_rng(seed)
    dt = 1.0 / fs
    t = np.arange(0.0, duration, dt)
    n = len(t)

    f_step = cadence / 2.0            # stride frequency (Hz); 2 steps per stride
    w_step = 2 * np.pi * cadence      # per-step angular freq
    w_stride = 2 * np.pi * f_step     # per-stride angular freq

    # gait envelope: stand still -> ramp up -> walk -> ramp down -> stand still.
    # Standing periods let ZUPT lock v=0 and calibrate biases, which is what
    # makes the absolute walking speed observable from inertial data alone.
    up = np.clip((t - rest) / ramp, 0.0, 1.0)
    down = np.clip((t - (duration - rest)) / ramp, 0.0, 1.0)
    env = up - down                                  # trapezoid 0 -> 1 -> 0

    # heading turns only while walking
    psi = np.cumsum(turn_rate * env) * dt
    fwd = np.stack([np.cos(psi), np.sin(psi)], axis=1)
    lat = np.stack([-np.sin(psi), np.cos(psi)], axis=1)

    # forward speed oscillates twice per stride around the (enveloped) mean speed
    v_fwd = env * mean_speed * (1.0 + fore_aft * np.sin(w_step * t))
    lateral = env * sway_amp * np.sin(w_stride * t)  # side-to-side offset

    pos = np.zeros((n, 3))
    # integrate the forward velocity vector (NOT distance x heading, which spirals)
    pos[:, :2] = np.cumsum(v_fwd[:, None] * fwd, axis=0) * dt + lateral[:, None] * lat
    pos[:, 2] = env * bob_amp * np.sin(w_step * t)   # vertical bob, twice per stride

    yaw = psi + env * np.deg2rad(yaw_amp_deg) * np.sin(w_stride * t)
    pitch = env * np.deg2rad(pitch_amp_deg) * np.sin(w_step * t)
    roll = env * np.deg2rad(roll_amp_deg) * np.sin(w_stride * t)
    R_list = [Rotation.from_euler("ZYX", [yaw[k], pitch[k], roll[k]]) for k in range(n)]

    return _finalize(t, dt, pos, R_list, rng, gyro_bias, accel_bias,
                     gyro_noise, accel_noise, mag_noise, device_id, placement)


def make_cadence_walk(
    fs=100.0,
    placement="sternum",
    speed_knots=((0, 0.0), (3, 0.0), (5, 1.0), (10, 1.0), (12, 1.7),
                 (18, 1.7), (20, 1.0), (24, 1.0), (26, 0.0), (29, 0.0)),
    cad0=1.50, cad1=0.45,          # cadence(steps/s) = cad0 + cad1*speed
    bob0=0.012, bob1=0.020,        # vertical bob amplitude (m) vs step length
    sway0=0.020, sway1=0.040,      # lateral sway amplitude (m) vs step length
    turn_rate=0.08,
    pitch_amp_deg=4.0, roll_amp_deg=3.0, yaw_amp_deg=2.0,
    seed=0,
    gyro_bias=(0.008, -0.006, 0.004),
    accel_bias=(0.04, -0.03, 0.02),
    gyro_noise=0.004, accel_noise=0.03, mag_noise=0.01,
    device_id="sternum",
):
    """Torso walk whose speed *varies* (rest -> 1.0 -> 1.7 -> 1.0 -> rest).

    Cadence, step length and vertical bounce are coupled to speed the way real
    gait is, so a cadence x step_length estimator has a genuine, non-trivial
    signal to recover. truth additionally carries `cadence` and `step_length`.
    """
    if placement in ("upper_arm", "arm"):
        pitch_amp_deg *= 4.0
        yaw_amp_deg *= 2.0

    rng = np.random.default_rng(seed)
    dt = 1.0 / fs
    duration = speed_knots[-1][0]
    t = np.arange(0.0, duration, dt)
    n = len(t)

    kt = np.array([k[0] for k in speed_knots], float)
    kv = np.array([k[1] for k in speed_knots], float)
    v = np.interp(t, kt, kv)
    # light smoothing to round the speed transitions
    w = max(1, int(0.4 * fs))
    v = np.convolve(v, np.ones(w) / w, mode="same")

    walking = v > 0.05
    cad = np.where(walking, cad0 + cad1 * v, 0.0)        # steps/s
    step_len = np.where(cad > 0, v / np.maximum(cad, 1e-6), 0.0)
    phase = np.cumsum(2 * np.pi * cad) * dt              # step-phase (vertical bob freq)

    bob = np.where(walking, bob0 + bob1 * step_len, 0.0)
    sway = np.where(walking, sway0 + sway1 * step_len, 0.0)

    psi = np.cumsum(turn_rate * walking) * dt
    fwd = np.stack([np.cos(psi), np.sin(psi)], axis=1)
    lat = np.stack([-np.sin(psi), np.cos(psi)], axis=1)

    pos = np.zeros((n, 3))
    pos[:, :2] = np.cumsum(v[:, None] * fwd, axis=0) * dt + (sway * np.sin(phase / 2))[:, None] * lat
    pos[:, 2] = bob * np.sin(phase)                      # vertical bob, once per step

    env = walking.astype(float)
    yaw = psi + env * np.deg2rad(yaw_amp_deg) * np.sin(phase / 2)
    pitch = env * np.deg2rad(pitch_amp_deg) * np.sin(phase)
    roll = env * np.deg2rad(roll_amp_deg) * np.sin(phase / 2)
    R_list = [Rotation.from_euler("ZYX", [yaw[k], pitch[k], roll[k]]) for k in range(n)]

    samples, truth = _finalize(t, dt, pos, R_list, rng, gyro_bias, accel_bias,
                               gyro_noise, accel_noise, mag_noise, device_id, placement)
    truth["cadence"] = cad
    truth["step_length"] = step_len
    truth["target_speed"] = v
    return samples, truth


def attitude_error_deg(est_quat, gt_quat):
    """Per-sample geodesic angle (deg) between estimated and true orientation."""
    est_quat = np.atleast_2d(est_quat)
    gt_quat = np.atleast_2d(gt_quat)
    errs = []
    for qe, qg in zip(est_quat, gt_quat):
        re = Rotation.from_quat(qe)
        rg = Rotation.from_quat(qg)
        errs.append((re * rg.inv()).magnitude())
    return np.degrees(np.asarray(errs))

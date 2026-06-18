"""15-state Error-State Kalman Filter (ESKF) for a single 9-DOF MARG device.

Conventions
-----------
* Navigation frame: ENU (x=East, y=North, z=Up), gravity g_n = [0, 0, -g].
* Attitude R maps body -> nav:  v_nav = R @ v_body.
* Error state (15) ordered [dp(3), dv(3), dtheta(3), dba(3), dbg(3)] with a
  *local* (body-frame) attitude error: R_true = R_nominal * exp([dtheta]_x).

The nominal state is propagated with strapdown integration; the EKF estimates the
small error state and injects it back after each update (multiplicative for
attitude), which keeps the covariance a clean 15x15 and avoids gimbal lock /
quaternion-normalization issues.

Measurements:
* accelerometer  -> gravity-direction aiding (corrects roll & pitch, accel bias)
* magnetometer   -> field-direction aiding   (corrects heading / yaw)
* zero-velocity  -> ZUPT pseudo-measurement   (bounds velocity & position drift)
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from .math_utils import skew

GRAVITY = 9.81


@dataclass
class FilterConfig:
    """Tunable parameters. Defaults are reasonable for a consumer foot-worn IMU."""

    gravity: float = GRAVITY

    # --- process noise (continuous PSD, scaled by dt during prediction) ---
    sigma_gyro: float = 0.01          # rad/s/sqrt(Hz)
    sigma_accel: float = 0.08         # m/s^2/sqrt(Hz)
    sigma_gyro_bias: float = 1e-5     # rad/s^2/sqrt(Hz) random walk
    sigma_accel_bias: float = 1e-4    # m/s^3/sqrt(Hz) random walk

    # --- measurement noise ---
    sigma_acc_grav: float = 0.35      # m/s^2, base noise of gravity update
    acc_grav_gate: float = 8.0        # inflates accel R when |a| deviates from g
    sigma_mag: float = 0.03           # unit-vector noise for heading update
    sigma_zupt: float = 0.02          # m/s, zero-velocity pseudo-measurement
    sigma_body_vel: float = 0.18      # m/s, body-planar velocity constraint (torso/arm)

    # --- reference magnetic field in nav frame (unit, ENU) ---
    # default: pointing North with downward inclination (z-Up so down is -z)
    mag_ref: Tuple[float, float, float] = (0.0, 0.5, -0.8660254)

    # --- initial 1-sigma uncertainties ---
    p0_pos: float = 1e-3              # m
    p0_vel: float = 0.1              # m/s
    p0_att: float = np.deg2rad(10.0)  # rad
    p0_accel_bias: float = 0.2       # m/s^2
    p0_gyro_bias: float = 0.05       # rad/s


class ESKF9DOF:
    def __init__(
        self,
        config: Optional[FilterConfig] = None,
        q_init=None,
        p_init=None,
        v_init=None,
        ba_init=None,
        bg_init=None,
    ):
        self.cfg = config or FilterConfig()
        self.g_n = np.array([0.0, 0.0, -self.cfg.gravity])

        # nominal state
        self.p = np.zeros(3) if p_init is None else np.asarray(p_init, float).copy()
        self.v = np.zeros(3) if v_init is None else np.asarray(v_init, float).copy()
        self.R = Rotation.identity() if q_init is None else Rotation.from_quat(q_init)
        self.b_a = np.zeros(3) if ba_init is None else np.asarray(ba_init, float).copy()
        self.b_g = np.zeros(3) if bg_init is None else np.asarray(bg_init, float).copy()

        # error-state covariance (15x15)
        c = self.cfg
        self.P = np.diag(
            [c.p0_pos**2] * 3
            + [c.p0_vel**2] * 3
            + [c.p0_att**2] * 3
            + [c.p0_accel_bias**2] * 3
            + [c.p0_gyro_bias**2] * 3
        ).astype(float)

        self.mag_ref = np.asarray(self.cfg.mag_ref, float)
        self.mag_ref = self.mag_ref / np.linalg.norm(self.mag_ref)
        self.last_acc_nav = np.zeros(3)
        self.initialized = False

    # ------------------------------------------------------------------ #
    # convenience accessors
    # ------------------------------------------------------------------ #
    @property
    def quat(self):
        """Orientation quaternion in scipy order [x, y, z, w]."""
        return self.R.as_quat()

    @property
    def euler_deg(self):
        """Roll/pitch/yaw in degrees (intrinsic xyz)."""
        return self.R.as_euler("xyz", degrees=True)

    @property
    def speed(self):
        return float(np.linalg.norm(self.v))

    @property
    def ground_speed(self):
        """Horizontal speed magnitude -- the subject's over-ground speed."""
        return float(np.linalg.norm(self.v[:2]))

    def state_dict(self, t=None):
        return {
            "t": t,
            "pos": self.p.copy(),
            "vel": self.v.copy(),
            "speed": self.speed,
            "ground_speed": self.ground_speed,
            "quat": self.quat.copy(),
            "euler": self.euler_deg.copy(),
            "accel_bias": self.b_a.copy(),
            "gyro_bias": self.b_g.copy(),
        }

    # ------------------------------------------------------------------ #
    # initialization from a static reading (gravity + magnetometer)
    # ------------------------------------------------------------------ #
    def initialize_from_measurement(self, accel, mag, weights=(1.0, 0.3)):
        """Set initial attitude by aligning measured gravity & field with refs.

        Solves a small Wahba problem (Kabsch) so the filter starts near the true
        orientation instead of relying on convergence from identity.
        """
        accel = np.asarray(accel, float)
        mag = np.asarray(mag, float)
        up_b = accel / np.linalg.norm(accel)            # body 'up' (specific force ~ -g)
        m_b = mag / np.linalg.norm(mag)
        up_n = np.array([0.0, 0.0, 1.0])
        R_est, _ = Rotation.align_vectors(
            [up_n, self.mag_ref], [up_b, m_b], weights=list(weights)
        )
        self.R = R_est
        self.initialized = True
        return self

    # ------------------------------------------------------------------ #
    # prediction (strapdown integration + error-state propagation)
    # ------------------------------------------------------------------ #
    def predict(self, dt, gyro, accel):
        if dt <= 0:
            return
        c = self.cfg
        gyro = np.asarray(gyro, float)
        accel = np.asarray(accel, float)

        omega = gyro - self.b_g          # bias-corrected angular rate (body)
        f_b = accel - self.b_a           # bias-corrected specific force (body)
        Rm = self.R.as_matrix()
        acc_nav = Rm @ f_b + self.g_n    # true linear acceleration in nav frame
        self.last_acc_nav = acc_nav      # gravity-compensated; feeds step detection

        # nominal kinematics
        self.p = self.p + self.v * dt + 0.5 * acc_nav * dt * dt
        self.v = self.v + acc_nav * dt
        self.R = self.R * Rotation.from_rotvec(omega * dt)

        # error-state transition F = I + A*dt
        I3 = np.eye(3)
        F = np.eye(15)
        F[0:3, 3:6] = I3 * dt
        F[3:6, 6:9] = -Rm @ skew(f_b) * dt
        F[3:6, 9:12] = -Rm * dt
        F[6:9, 6:9] = I3 - skew(omega) * dt
        F[6:9, 12:15] = -I3 * dt

        # discrete process noise (diagonal block approximation)
        Qd = np.zeros((15, 15))
        Qd[3:6, 3:6] = I3 * (c.sigma_accel**2) * dt
        Qd[6:9, 6:9] = I3 * (c.sigma_gyro**2) * dt
        Qd[9:12, 9:12] = I3 * (c.sigma_accel_bias**2) * dt
        Qd[12:15, 12:15] = I3 * (c.sigma_gyro_bias**2) * dt

        self.P = F @ self.P @ F.T + Qd
        self.P = 0.5 * (self.P + self.P.T)

    # ------------------------------------------------------------------ #
    # generic measurement update (Joseph form for numerical stability)
    # ------------------------------------------------------------------ #
    def _update(self, H, y, R_meas):
        S = H @ self.P @ H.T + R_meas
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y
        self._inject(dx)
        IKH = np.eye(15) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_meas @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def _inject(self, dx):
        self.p = self.p + dx[0:3]
        self.v = self.v + dx[3:6]
        self.R = self.R * Rotation.from_rotvec(dx[6:9])   # multiplicative attitude
        self.b_a = self.b_a + dx[9:12]
        self.b_g = self.b_g + dx[12:15]

    # ------------------------------------------------------------------ #
    # measurement models
    # ------------------------------------------------------------------ #
    def update_accel(self, accel):
        """Gravity-direction aiding. Corrects roll/pitch and accel bias.

        Assumes specific force ~ gravity only; the measurement noise is inflated
        adaptively when |accel| departs from g (i.e. the body is accelerating),
        so dynamic phases are automatically down-weighted.
        """
        accel = np.asarray(accel, float)
        Rm = self.R.as_matrix()
        v_n = -self.g_n                      # expected accel direction (Up * g)
        h = Rm.T @ v_n + self.b_a
        y = accel - h
        H = np.zeros((3, 15))
        H[:, 6:9] = skew(Rm.T @ v_n)
        H[:, 9:12] = np.eye(3)
        dev = abs(np.linalg.norm(accel) - self.cfg.gravity)
        r = self.cfg.sigma_acc_grav**2 + (self.cfg.acc_grav_gate * dev) ** 2
        self._update(H, y, np.eye(3) * r)

    def update_mag(self, mag):
        """Field-direction aiding. Corrects heading (yaw)."""
        mag = np.asarray(mag, float)
        n = np.linalg.norm(mag)
        if n < 1e-9:
            return
        mag_u = mag / n
        Rm = self.R.as_matrix()
        h = Rm.T @ self.mag_ref
        y = mag_u - h
        H = np.zeros((3, 15))
        H[:, 6:9] = skew(Rm.T @ self.mag_ref)
        self._update(H, y, np.eye(3) * (self.cfg.sigma_mag**2))

    def update_zupt(self):
        """Zero-velocity pseudo-measurement: pin velocity to zero while stationary."""
        H = np.zeros((3, 15))
        H[:, 3:6] = np.eye(3)
        y = -self.v
        self._update(H, y, np.eye(3) * (self.cfg.sigma_zupt**2))

    def update_body_velocity_zero(self, axes=(1, 2), sigma=None):
        """Soft constraint: drift of velocity along the given BODY axes is ~zero.

        For a torso/arm sensor there is no stance phase, but the body sways and
        bobs about a path it only travels *forward* along (in its own frame).
        Constraining body-lateral (y) and body-vertical (z) velocity toward zero
        -- with loose noise so the real sway/bob still shows -- removes the slow
        drift that would ruin the speed estimate, while leaving body-forward (x)
        velocity free as the measured walking speed.

        We deliberately omit the attitude (dtheta) coupling term: orientation is
        already well constrained by the gravity + magnetometer updates, and
        letting an oscillating velocity perturb attitude only degrades it.
        """
        sigma = self.cfg.sigma_body_vel if sigma is None else sigma
        Rm = self.R.as_matrix()
        vb = Rm.T @ self.v                  # velocity in body frame
        axes = list(axes)
        H = np.zeros((len(axes), 15))
        for i, j in enumerate(axes):
            H[i, 3:6] = Rm[:, j]            # d(vb_j)/dv = (R^T row j) = R[:, j]
        y = -vb[axes]
        self._update(H, y, np.eye(len(axes)) * (sigma**2))

    # ------------------------------------------------------------------ #
    # one full step
    # ------------------------------------------------------------------ #
    def step(self, dt, gyro, accel, mag=None, zupt=False):
        self.predict(dt, gyro, accel)
        self.update_accel(accel)
        if mag is not None:
            self.update_mag(mag)
        if zupt:
            self.update_zupt()
        return self.state_dict()

"""Cadence-based walking-speed estimation for torso / arm placements.

Absolute speed is *not* observable by integrating a single torso IMU (gravity
and sustained linear acceleration are indistinguishable). Real chest/wrist
wearables instead estimate speed biomechanically:

    speed = cadence x step_length

This module detects steps from the gravity-compensated vertical acceleration
(produced by the EKF), measures cadence from the step timing, and maps cadence
to step length with a small calibrated model. The EKF supplies clean,
gravity-removed vertical acceleration and orientation; this supplies speed.

Pipeline: StepDetector -> step events -> SpeedEstimator -> speed.
"""

import numpy as np


class StepDetector:
    """Streaming peak detector on a (near-zero-mean) vertical-acceleration signal.

    Emits one event per step: a peak that rises above `accel_thresh` with at
    least `min_interval` since the previous step. Reports the step interval and
    the peak-to-trough acceleration amplitude over the step.
    """

    def __init__(self, min_interval=0.30, accel_thresh=0.8):
        self.min_interval = min_interval
        self.accel_thresh = accel_thresh
        self._prev_a = None
        self._prev_t = None
        self._rising = False
        self._last_step_t = None
        self._cur_min = np.inf
        self._cur_max = -np.inf

    def update(self, t, a):
        """Feed one (time, vertical-accel) sample; return an event dict or None."""
        self._cur_min = min(self._cur_min, a)
        self._cur_max = max(self._cur_max, a)
        event = None
        if self._prev_a is not None:
            falling = a < self._prev_a
            # a peak = was rising, now falling, and the peak cleared the threshold
            if self._rising and falling and self._prev_a > self.accel_thresh:
                t_pk = self._prev_t
                if self._last_step_t is None or (t_pk - self._last_step_t) >= self.min_interval:
                    interval = None if self._last_step_t is None else t_pk - self._last_step_t
                    event = {
                        "t": t_pk,
                        "interval": interval,
                        "amp": self._cur_max - self._cur_min,
                    }
                    self._last_step_t = t_pk
                    self._cur_min = np.inf
                    self._cur_max = -np.inf
            self._rising = a > self._prev_a
        self._prev_a = a
        self._prev_t = t
        return event


class SpeedEstimator:
    """Turns step events into a walking-speed estimate via cadence x step_length.

    Step length is modelled as a linear function of cadence -- step_length =
    a + b * cadence -- which captures that people lengthen their stride as they
    speed up. `a` and `b` are per-subject calibration constants (fit once from a
    short walk over a known distance, or from labelled speed).
    """

    def __init__(self, a=0.30, b=0.12, smooth=0.4, min_interval=0.30,
                 accel_thresh=0.8, stop_timeout=0.9, stop_tau=0.4):
        self.a = a
        self.b = b
        self.smooth = smooth
        self.stop_timeout = stop_timeout   # s without a step before we treat it as stopped
        self.stop_tau = stop_tau           # s decay time constant once stopped
        self.detector = StepDetector(min_interval=min_interval, accel_thresh=accel_thresh)
        self.speed = 0.0
        self.cadence = 0.0
        self.step_length = 0.0
        self.steps = []           # recorded (t, cadence, amp) for calibration/analysis
        self._last_step_t = None
        self._prev_t = None

    def update(self, t, a_vert):
        """Feed (time, gravity-compensated vertical accel). Returns current speed."""
        if self._prev_t is None:
            self._prev_t = t
        dt = max(0.0, t - self._prev_t)
        self._prev_t = t

        ev = self.detector.update(t, a_vert)
        if ev is not None and ev["interval"] and ev["interval"] > 1e-3:
            cad = 1.0 / ev["interval"]                  # steps per second
            self.cadence = cad
            self.step_length = max(0.0, self.a + self.b * cad)
            inst_speed = self.step_length * cad
            # hold-between-steps + EMA smoothing across steps
            self.speed = (1 - self.smooth) * self.speed + self.smooth * inst_speed
            self.steps.append((ev["t"], cad, ev["amp"]))
            self._last_step_t = ev["t"]
        elif self._last_step_t is not None and (t - self._last_step_t) > self.stop_timeout:
            # no steps for a while -> subject has stopped; decay toward zero
            self.speed *= np.exp(-dt / self.stop_tau)
            self.cadence = 0.0
        # otherwise: hold the current speed between steps
        return self.speed

    # ------------------------------------------------------------------ #
    @staticmethod
    def fit(cadences, true_speeds):
        """Least-squares fit of step_length = a + b*cadence.

        Uses step_length = speed / cadence as the regression target.
        Returns (a, b).
        """
        cad = np.asarray(cadences, float)
        spd = np.asarray(true_speeds, float)
        good = cad > 1e-3
        cad, spd = cad[good], spd[good]
        step_len = spd / cad
        A = np.column_stack([np.ones_like(cad), cad])
        coef, *_ = np.linalg.lstsq(A, step_len, rcond=None)
        return float(coef[0]), float(coef[1])

    def calibrate(self, cadences, true_speeds):
        self.a, self.b = self.fit(cadences, true_speeds)
        return self.a, self.b

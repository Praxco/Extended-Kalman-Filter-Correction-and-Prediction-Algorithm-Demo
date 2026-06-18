"""Zero-velocity (stance-phase) detection for pedestrian dead reckoning.

A windowed SHOE-style detector: when the device is near-stationary the measured
acceleration magnitude sits at ~g with low variance and the gyro magnitude is
small. It self-gates by placement -- a foot satisfies this every stance phase, a
torso almost never does -- so we can run the same detector on every device.
"""

from collections import deque
import numpy as np


class ZeroVelocityDetector:
    def __init__(
        self,
        g=9.81,
        accel_var_thresh=0.5,
        accel_bias_thresh=0.7,
        gyro_thresh=0.35,
        window=10,
    ):
        """
        Parameters
        ----------
        g : float
            Local gravity magnitude, m/s^2.
        accel_var_thresh : float
            Max std-dev of |accel| over the window to be considered still.
        accel_bias_thresh : float
            Max |mean(|accel|) - g| over the window to be considered still.
        gyro_thresh : float
            Max mean |gyro| (rad/s) over the window to be considered still.
        window : int
            Number of samples in the sliding decision window.
        """
        self.g = g
        self.accel_var_thresh = accel_var_thresh
        self.accel_bias_thresh = accel_bias_thresh
        self.gyro_thresh = gyro_thresh
        self.window = window
        self._acc = deque(maxlen=window)
        self._gyr = deque(maxlen=window)

    def reset(self):
        self._acc.clear()
        self._gyr.clear()

    def update(self, accel, gyro):
        """Push one sample, return True if the device is currently stationary."""
        self._acc.append(float(np.linalg.norm(accel)))
        self._gyr.append(float(np.linalg.norm(gyro)))
        if len(self._acc) < self.window:
            return False
        acc = np.asarray(self._acc)
        gyr = np.asarray(self._gyr)
        still_level = abs(acc.mean() - self.g) < self.accel_bias_thresh
        still_var = acc.std() < self.accel_var_thresh
        still_gyro = gyr.mean() < self.gyro_thresh
        return bool(still_level and still_var and still_gyro)

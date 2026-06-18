"""Sensor data boundary.

`Sample` is the single contract between a data *source* (synthetic generator now,
live multi-device stream later) and the filter. Keeping this thin and explicit is
what lets us swap sources without touching the estimator.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Sample:
    """One synchronized MARG reading from one device.

    Attributes
    ----------
    t : float
        Timestamp in seconds (monotonic; used to derive dt).
    device_id : str
        Identifies which body-worn device produced this reading.
    accel : np.ndarray, shape (3,)
        Specific force in the sensor/body frame, m/s^2.
    gyro : np.ndarray, shape (3,)
        Angular rate in the sensor/body frame, rad/s.
    mag : np.ndarray, shape (3,)
        Magnetic field in the sensor/body frame (only the direction is used).
    """

    t: float
    device_id: str
    accel: np.ndarray
    gyro: np.ndarray
    mag: np.ndarray

    def __post_init__(self):
        self.accel = np.asarray(self.accel, dtype=float).reshape(3)
        self.gyro = np.asarray(self.gyro, dtype=float).reshape(3)
        self.mag = np.asarray(self.mag, dtype=float).reshape(3)

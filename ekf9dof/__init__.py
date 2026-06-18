"""ekf9dof - portable 9-DOF MARG Error-State Kalman Filter.

Core (filter, manager, zupt, sensors) depends only on numpy + scipy so it can be
dropped into the HIL optimization toolkit without dragging in any UI deps.
"""

from .filter import ESKF9DOF, FilterConfig, GRAVITY
from .zupt import ZeroVelocityDetector
from .sensors import Sample
from .manager import MultiDeviceFilter, DeviceFilter

__all__ = [
    "ESKF9DOF",
    "FilterConfig",
    "GRAVITY",
    "ZeroVelocityDetector",
    "Sample",
    "MultiDeviceFilter",
    "DeviceFilter",
]

__version__ = "0.1.0"

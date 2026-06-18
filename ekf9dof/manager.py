"""Multi-device orchestration.

Each body-worn device gets its own independent ESKF + ZUPT detector. The manager
routes incoming `Sample`s by device_id, derives dt per device, and records a
history for analysis/visualization. This is the seam where inter-segment
biomechanical constraints could later be added without touching the core filter.
"""

from typing import Dict, Iterable, List, Optional

from .filter import ESKF9DOF, FilterConfig
from .zupt import ZeroVelocityDetector
from .sensors import Sample

# Placement -> aiding strategy. A foot has true stance phases (ZUPT); a torso or
# arm never stops, so we use the body-planar velocity constraint instead.
_FOOT_PLACEMENTS = {"foot", "ankle", "shoe"}


def infer_aiding(placement: Optional[str]) -> str:
    if placement is None:
        return "zupt"
    return "zupt" if placement.lower() in _FOOT_PLACEMENTS else "body_planar"


class DeviceFilter:
    """A single device's estimator + placement-appropriate aiding + history."""

    def __init__(
        self,
        device_id: str,
        config: Optional[FilterConfig] = None,
        placement: Optional[str] = None,
        use_zupt: bool = True,
        aiding: Optional[str] = None,
    ):
        self.device_id = device_id
        self.placement = placement
        self.use_aiding = use_zupt           # toggle aiding on/off (name kept for compat)
        self.aiding = aiding or infer_aiding(placement)
        self.ekf = ESKF9DOF(config)
        self.zupt = ZeroVelocityDetector(g=self.ekf.cfg.gravity)
        self.last_t: Optional[float] = None
        self.history: List[dict] = []

    def process(self, sample: Sample) -> dict:
        # First sample: level the filter from gravity + field, don't integrate.
        if self.last_t is None:
            self.ekf.initialize_from_measurement(sample.accel, sample.mag)
            self.last_t = sample.t
            return self._record(sample.t, stationary=True)

        dt = sample.t - self.last_t
        self.last_t = sample.t
        if dt <= 0:
            dt = 1e-3

        self.ekf.predict(dt, sample.gyro, sample.accel)
        self.ekf.update_accel(sample.accel)
        self.ekf.update_mag(sample.mag)

        stationary = False
        if self.use_aiding and self.aiding == "zupt":
            stationary = self.zupt.update(sample.accel, sample.gyro)
            if stationary:
                self.ekf.update_zupt()
        elif self.use_aiding and self.aiding == "body_planar":
            # While genuinely still (e.g. standing at the start) a full ZUPT
            # locks velocity to zero and calibrates the biases -- which is what
            # makes the subsequent walking speed observable. Once moving, fall
            # back to the body-planar (no sideways/vertical drift) constraint.
            stationary = self.zupt.update(sample.accel, sample.gyro)
            if stationary:
                self.ekf.update_zupt()
            else:
                self.ekf.update_body_velocity_zero()

        return self._record(sample.t, stationary)

    def _record(self, t, stationary) -> dict:
        snap = self.ekf.state_dict(t)
        snap["device_id"] = self.device_id
        snap["placement"] = self.placement
        snap["aiding"] = self.aiding
        snap["stationary"] = bool(stationary)
        self.history.append(snap)
        return snap


class MultiDeviceFilter:
    """Routes a multiplexed sample stream to per-device filters."""

    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config
        self.devices: Dict[str, DeviceFilter] = {}

    def add_device(
        self,
        device_id: str,
        placement: Optional[str] = None,
        use_zupt: bool = True,
        config: Optional[FilterConfig] = None,
        aiding: Optional[str] = None,
    ) -> DeviceFilter:
        dev = DeviceFilter(
            device_id,
            config=config or self.config,
            placement=placement,
            use_zupt=use_zupt,
            aiding=aiding,
        )
        self.devices[device_id] = dev
        return dev

    def process(self, sample: Sample) -> dict:
        if sample.device_id not in self.devices:
            self.add_device(sample.device_id)
        return self.devices[sample.device_id].process(sample)

    def process_stream(self, samples: Iterable[Sample]) -> List[dict]:
        return [self.process(s) for s in samples]

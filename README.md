# ekf9dof — 9-DOF MARG Extended Kalman Filter

A compact, portable sensor-fusion module that turns raw **9-axis IMU data**
(3-axis accelerometer + gyroscope + magnetometer) into **orientation, speed, and
trajectory** for a moving subject. Designed as a self-contained algorithm to be
patched into a larger HIL optimization toolkit.

The estimator is a **15-state Error-State Kalman Filter (ESKF)** with
quaternion attitude, online gyro/accel bias estimation, gravity + magnetometer
aiding, and **Zero-Velocity Updates (ZUPT)** for pedestrian dead reckoning.

## Layout
```
ekf9dof/        portable core — depends ONLY on numpy + scipy
  filter.py       the 15-state ESKF (predict / accel / mag / zupt updates)
  zupt.py         stance-phase (zero-velocity) detector
  manager.py      multi-device orchestration (one filter per body segment)
  sensors.py      Sample dataclass — the source <-> filter boundary
  math_utils.py   small helpers
tests/          pytest suite + synthetic ground-truth generator
demo/           run_demo.py (matplotlib) + TALKING_POINTS.md
```

The `ekf9dof/` package has **no UI dependencies** — ideally should be able to 
just drop into one of the HIL toolkits once the suite is ready

## Quick start
```bash
python demo/run_demo.py     # runs the filter, prints metrics, shows the figure
python -m pytest -q         # 5 tests validate it against ground truth
```

## Minimal API
```python
from ekf9dof import MultiDeviceFilter, Sample

mdf = MultiDeviceFilter()
mdf.add_device("foot", placement="foot")     # add one per body-worn device

snap = mdf.process(Sample(t, "foot", accel=a, gyro=g, mag=m))
# snap -> {"pos", "vel", "speed", "euler", "quat", "gyro_bias", "stationary", ...}
```

Same interface for synthetic data today and a live multi-device stream later.

## Status
- [x] Core ESKF + ZUPT, multi-device manager
- [x] Synthetic ground-truth simulator + 5 passing tests
- [x] Matplotlib proof-of-concept demo
- [ ] Web UI (Streamlit + Plotly) — planned
- [ ] Real-device ingestion + live streaming
- [ ] Performance pass (currently ~1.6k full steps/s/core)

Requires: `numpy`, `scipy` (core); `matplotlib` (just for initial testing of filter's accuracy, I would rather use HTML or streamlit_app going forward).

"""Functional tests that prove the filter actually estimates correctly.

These assert against ground truth from the synthetic generator: orientation
tracking accuracy, ZUPT bounding position drift, and gyro-bias convergence.
"""

import numpy as np
import pytest

from ekf9dof.manager import MultiDeviceFilter
from ekf9dof.sensors import Sample
from tests.synthetic import make_walking_sequence, attitude_error_deg


def _run(use_zupt, seed=1):
    samples, truth = make_walking_sequence(seed=seed)
    mdf = MultiDeviceFilter()
    dev = mdf.add_device("foot", placement="foot", use_zupt=use_zupt)
    snaps = [dev.process(s) for s in samples]
    est = {
        "quat": np.array([s["quat"] for s in snaps]),
        "pos": np.array([s["pos"] for s in snaps]),
        "vel": np.array([s["vel"] for s in snaps]),
        "gyro_bias": np.array([s["gyro_bias"] for s in snaps]),
        "stationary": np.array([s["stationary"] for s in snaps]),
    }
    return truth, est, dev


def test_attitude_tracks_ground_truth():
    truth, est, _ = _run(use_zupt=True)
    # skip the first second (initial convergence)
    start = 200
    err = attitude_error_deg(est["quat"][start:], truth["quat"][start:])
    rmse = np.sqrt(np.mean(err**2))
    assert rmse < 5.0, f"attitude RMSE too high: {rmse:.2f} deg"


def test_zupt_bounds_position_drift():
    truth_z, est_z, _ = _run(use_zupt=True)
    truth_n, est_n, _ = _run(use_zupt=False)
    err_z = np.linalg.norm(est_z["pos"][-1, :2] - truth_z["pos"][-1, :2])
    err_n = np.linalg.norm(est_n["pos"][-1, :2] - truth_n["pos"][-1, :2])
    # ZUPT must keep drift bounded and dramatically beat the un-aided integrator
    assert err_z < 1.0, f"ZUPT position error too high: {err_z:.2f} m"
    assert err_z < 0.25 * err_n, (
        f"ZUPT ({err_z:.2f} m) should be far better than no-ZUPT ({err_n:.2f} m)"
    )


def test_zupt_detects_stance_phases():
    _, est, _ = _run(use_zupt=True)
    frac_still = est["stationary"].mean()
    # gait is ~40% stance; detector should fire a meaningful but not absurd amount
    assert 0.1 < frac_still < 0.7, f"stance detection fraction implausible: {frac_still:.2f}"


def test_gyro_bias_converges():
    truth, est, _ = _run(use_zupt=True)
    est_bias = est["gyro_bias"][-1]
    err = np.linalg.norm(est_bias - truth["gyro_bias"])
    assert err < 0.02, f"gyro bias estimate off by {err:.4f} rad/s"


def test_multi_device_routing_is_independent():
    """Two devices in one stream must keep separate state."""
    samples_a, _ = make_walking_sequence(seed=2, device_id="foot")
    samples_b, _ = make_walking_sequence(seed=7, device_id="wrist", stride_speed=0.6)
    mdf = MultiDeviceFilter()
    mdf.add_device("foot", placement="foot")
    mdf.add_device("wrist", placement="wrist")
    # interleave the two streams
    for sa, sb in zip(samples_a, samples_b):
        mdf.process(sa)
        mdf.process(sb)
    assert set(mdf.devices) == {"foot", "wrist"}
    foot_hist = mdf.devices["foot"].history
    wrist_hist = mdf.devices["wrist"].history
    assert len(foot_hist) == len(samples_a)
    assert len(wrist_hist) == len(samples_b)
    # distinct trajectories
    foot_end = foot_hist[-1]["pos"]
    wrist_end = wrist_hist[-1]["pos"]
    assert not np.allclose(foot_end, wrist_end)

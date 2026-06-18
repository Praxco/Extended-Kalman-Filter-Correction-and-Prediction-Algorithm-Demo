"""Proof-of-concept demo for the 9-DOF MARG Error-State Kalman Filter.

Run:
    python demo/run_demo.py

Generates a synthetic walking trajectory, feeds the raw 9-axis sensor stream
through the filter, and produces a single figure that shows, left-to-right:

  * the raw accelerometer / gyroscope / magnetometer data going in,
  * the recovered orientation vs. ground truth,
  * the recovered 3-D walking trajectory (with ZUPT) vs. ground truth and vs.
    the un-aided integrator that drifts away.

It also prints the headline numbers (orientation error, drift, throughput) to
the console for talking points.
"""

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ekf9dof.manager import MultiDeviceFilter
from tests.synthetic import make_walking_sequence, attitude_error_deg


def run(samples, use_zupt):
    mdf = MultiDeviceFilter()
    dev = mdf.add_device("foot", placement="foot", use_zupt=use_zupt)
    t0 = time.perf_counter()
    snaps = [dev.process(s) for s in samples]
    elapsed = time.perf_counter() - t0
    return {
        "pos": np.array([s["pos"] for s in snaps]),
        "quat": np.array([s["quat"] for s in snaps]),
        "euler": np.array([s["euler"] for s in snaps]),
        "stationary": np.array([s["stationary"] for s in snaps]),
        "elapsed": elapsed,
        "n": len(snaps),
    }


def main(show=True):
    # --- generate one synthetic walk and run the filter two ways ---
    fs = 100.0
    samples, truth = make_walking_sequence(duration=20.0, fs=fs, seed=1)
    est = run(samples, use_zupt=True)
    raw_int = run(samples, use_zupt=False)

    t = truth["t"]
    acc = np.array([s.accel for s in samples])
    gyr = np.array([s.gyro for s in samples])
    mag = np.array([s.mag for s in samples])

    # --- headline metrics ---
    att_rmse = np.sqrt(np.mean(attitude_error_deg(est["quat"][100:], truth["quat"][100:]) ** 2))
    drift_zupt = np.linalg.norm(est["pos"][-1, :2] - truth["pos"][-1, :2])
    drift_raw = np.linalg.norm(raw_int["pos"][-1, :2] - truth["pos"][-1, :2])
    path_len = np.sum(np.linalg.norm(np.diff(truth["pos"], axis=0), axis=1))
    rate = est["n"] / est["elapsed"]

    print("=" * 60)
    print(" 9-DOF MARG Extended Kalman Filter - demo summary")
    print("=" * 60)
    print(f" samples processed      : {est['n']} @ {fs:.0f} Hz  ({t[-1]:.0f} s walk)")
    print(f" true path length       : {path_len:5.1f} m")
    print(f" orientation RMSE       : {att_rmse:5.2f} deg")
    print(f" final drift WITH ZUPT  : {drift_zupt:5.2f} m   ({drift_zupt/path_len*100:.1f}% of path)")
    print(f" final drift NO aiding  : {drift_raw:5.2f} m   ({drift_raw/path_len*100:.1f}% of path)")
    print(f" throughput             : {rate:,.0f} steps/s  ({1e6/rate:.0f} us/step)")
    print("=" * 60)

    # --- figure ---
    win = (t <= 6.0)  # show first 6 s of raw data so it stays readable
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("9-DOF MARG Extended Kalman Filter - Pedestrian Tracking Demo", fontsize=15, fontweight="bold")
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.26)

    labels = ["x", "y", "z"]
    for col, (data, name, unit) in enumerate(
        [(acc, "Accelerometer", "m/s$^2$"), (gyr, "Gyroscope", "rad/s"), (mag, "Magnetometer", "norm.")]
    ):
        ax = fig.add_subplot(gs[0, col])
        for i in range(3):
            ax.plot(t[win], data[win, i], lw=0.8, label=labels[i])
        ax.set_title(f"Raw {name} (input)", fontsize=11)
        ax.set_xlabel("t (s)")
        ax.set_ylabel(unit)
        ax.legend(loc="upper right", fontsize=8, ncol=3)
        ax.grid(alpha=0.3)

    # orientation est vs truth
    ax = fig.add_subplot(gs[1, 0])
    names = ["roll", "pitch", "yaw"]
    colors = ["C0", "C1", "C2"]
    for i in range(3):
        ax.plot(t, truth["euler"][:, i], color=colors[i], lw=2.2, alpha=0.35)
        ax.plot(t, est["euler"][:, i], color=colors[i], lw=1.0, label=names[i])
    ax.set_title(f"Orientation: estimate vs truth (RMSE {att_rmse:.2f} deg)", fontsize=11)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("angle (deg)")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.04, "thick = truth   thin = estimate", transform=ax.transAxes, fontsize=8, alpha=0.7)

    # 3D trajectory
    ax = fig.add_subplot(gs[1, 1:3], projection="3d")
    tp, ep, rp = truth["pos"], est["pos"], raw_int["pos"]
    ax.plot(tp[:, 0], tp[:, 1], tp[:, 2], color="k", lw=3, alpha=0.35, label="ground truth")
    ax.plot(ep[:, 0], ep[:, 1], ep[:, 2], color="C2", lw=1.6, label=f"EKF + ZUPT  (drift {drift_zupt:.2f} m)")
    ax.plot(rp[:, 0], rp[:, 1], rp[:, 2], color="C3", lw=1.2, ls="--", label=f"no aiding  (drift {drift_raw:.1f} m)")

    # body-frame triads along the estimated path (the "vector frames")
    from scipy.spatial.transform import Rotation
    idx = np.linspace(0, est["n"] - 1, 7, dtype=int)
    L = 0.4
    for k in idx:
        R = Rotation.from_quat(est["quat"][k]).as_matrix()
        o = ep[k]
        for ax_i, c in zip(range(3), ["#d62728", "#2ca02c", "#1f77b4"]):
            ax.quiver(o[0], o[1], o[2], R[0, ax_i], R[1, ax_i], R[2, ax_i], length=L, color=c, lw=1.2)
    ax.set_title("3-D trajectory + body-frame orientation (red/green/blue = body x/y/z)", fontsize=11)
    ax.set_xlabel("E (m)")
    ax.set_ylabel("N (m)")
    ax.set_zlabel("U (m)")
    ax.legend(loc="upper left", fontsize=8)
    try:
        ax.set_box_aspect((np.ptp(tp[:, 0]) + 1, np.ptp(tp[:, 1]) + 1, max(np.ptp(tp[:, 2]) * 4, 1)))
    except Exception:
        pass

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_output.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f" figure saved -> {out}")
    if show:
        plt.show()


if __name__ == "__main__":
    main(show="--no-show" not in sys.argv)

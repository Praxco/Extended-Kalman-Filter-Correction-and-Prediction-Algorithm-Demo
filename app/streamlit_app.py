"""Streamlit dashboard for the 9-DOF MARG EKF.

Launch:
    streamlit run app/streamlit_app.py

TODO: Shows, for a sternum / upper-arm placement: the recovered orientation
(pitch/roll/yaw) vs ground truth, the cadence-based walking-speed estimate vs
ground truth, an interactive 3-D body-frame orientation, and the raw sensor data.
"""

import os
import sys

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ekf9dof.filter import ESKF9DOF
from ekf9dof.cadence import SpeedEstimator
from tests.synthetic import make_cadence_walk, attitude_error_deg

st.set_page_config(page_title="9-DOF MARG EKF", layout="wide")


@st.cache_data
def run_pipeline(placement, seed):
    samples, truth = make_cadence_walk(placement=placement, seed=seed)
    t = truth["t"]
    tg = truth["ground_speed"]

    ekf = ESKF9DOF()
    ekf.initialize_from_measurement(samples[0].accel, samples[0].mag)
    last = samples[0].t
    eul = [ekf.euler_deg]
    quat = [ekf.quat]
    av = [(samples[0].t, 0.0)]
    for s in samples[1:]:
        dt = s.t - last
        last = s.t
        ekf.predict(dt, s.gyro, s.accel)
        ekf.update_accel(s.accel)
        ekf.update_mag(s.mag)
        eul.append(ekf.euler_deg)
        quat.append(ekf.quat)
        av.append((s.t, ekf.last_acc_nav[2]))
    eul = np.array(eul)
    quat = np.array(quat)

    # per-subject calibration of the cadence -> step-length model
    det = SpeedEstimator()
    cads, spds = [], []
    for tt, a in av:
        ev = det.detector.update(tt, a)
        if ev and ev["interval"]:
            cads.append(1.0 / ev["interval"])
            spds.append(tg[np.argmin(np.abs(t - ev["t"]))])
    A, B = SpeedEstimator.fit(cads, spds) if len(cads) > 2 else (0.30, 0.12)
    est = SpeedEstimator(a=A, b=B)
    spd = np.array([est.update(tt, a) for tt, a in av])

    return {
        "t": t,
        "true_speed": tg,
        "est_speed": spd,
        "eul": eul,
        "true_eul": truth["euler"],
        "quat": quat,
        "true_quat": truth["quat"],
        "acc": np.array([s.accel for s in samples]),
        "gyr": np.array([s.gyro for s in samples]),
        "mag": np.array([s.mag for s in samples]),
        "A": A,
        "B": B,
    }


st.title("9-DOF MARG Extended Kalman Filter")
st.caption("Pedestrian orientation + walking speed from a sternum / arm IMU "
           "(3-axis accel + gyro + mag). Synthetic ground-truth demo.")

with st.sidebar:
    st.header("Scenario")
    placement = st.selectbox("Sensor placement", ["sternum", "upper_arm"])
    seed = st.slider("Random seed (noise)", 0, 20, 3)
    show_raw = st.checkbox("Show raw sensor data", value=False)

d = run_pipeline(placement, seed)
t = d["t"]
walk = d["true_speed"] > 0.2

sp_rmse = float(np.sqrt(np.mean((d["est_speed"][walk] - d["true_speed"][walk]) ** 2)))
sp_mape = float(np.mean(np.abs(d["est_speed"][walk] - d["true_speed"][walk]) / d["true_speed"][walk]) * 100)
att_rmse = float(np.sqrt(np.mean(attitude_error_deg(d["quat"], d["true_quat"]) ** 2)))

c1, c2, c3 = st.columns(3)
c1.metric("Orientation RMSE", f"{att_rmse:.2f} deg")
c2.metric("Speed RMSE", f"{sp_rmse:.2f} m/s")
c3.metric("Speed error", f"{sp_mape:.0f} %")

# ---- speed ----
st.subheader("Walking speed — estimate vs ground truth")
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=d["true_speed"], name="true", line=dict(color="#888", width=4)))
fig.add_trace(go.Scatter(x=t, y=d["est_speed"], name="EKF + cadence model", line=dict(color="#2ca02c", width=2)))
fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                  xaxis_title="t (s)", yaxis_title="speed (m/s)", legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

# ---- orientation ----
st.subheader("Orientation — estimate vs ground truth")
names = ["roll", "pitch", "yaw"]
colors = ["#1f77b4", "#ff7f0e", "#d62728"]
figo = make_subplots(rows=1, cols=3, subplot_titles=names)
for i in range(3):
    figo.add_trace(go.Scatter(x=t, y=d["true_eul"][:, i], name="true", legendgroup="t",
                              showlegend=(i == 0), line=dict(color="#aaa", width=4)), row=1, col=i + 1)
    figo.add_trace(go.Scatter(x=t, y=d["eul"][:, i], name="estimate", legendgroup="e",
                              showlegend=(i == 0), line=dict(color=colors[i], width=1.5)), row=1, col=i + 1)
figo.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h"))
st.plotly_chart(figo, use_container_width=True)

# ---- interactive 3D body frame ----
st.subheader("Body-frame orientation (drag to rotate)")
k = st.slider("time (s)", float(t[0]), float(t[-1]), float(t[len(t) // 2]), 0.1)
ki = int(np.argmin(np.abs(t - k)))
R = Rotation.from_quat(d["quat"][ki]).as_matrix()
fig3 = go.Figure()
for i, (c, nm) in enumerate(zip(["#d62728", "#2ca02c", "#1f77b4"], ["x (forward)", "y (left)", "z (up)"])):
    fig3.add_trace(go.Scatter3d(x=[0, R[0, i]], y=[0, R[1, i]], z=[0, R[2, i]],
                                mode="lines", line=dict(color=c, width=10), name=nm))
fig3.update_layout(height=420, margin=dict(l=0, r=0, t=0, b=0),
                   scene=dict(xaxis=dict(range=[-1, 1]), yaxis=dict(range=[-1, 1]),
                              zaxis=dict(range=[-1, 1]), aspectmode="cube"))
st.plotly_chart(fig3, use_container_width=True)

# ---- raw sensors ----
if show_raw:
    st.subheader("Raw sensor data (input)")
    figr = make_subplots(rows=1, cols=3, subplot_titles=["accelerometer", "gyroscope", "magnetometer"])
    for col, arr in enumerate([d["acc"], d["gyr"], d["mag"]]):
        for i, ax in enumerate("xyz"):
            figr.add_trace(go.Scatter(x=t, y=arr[:, i], name=ax, showlegend=(col == 0),
                                      legendgroup=ax), row=1, col=col + 1)
    figr.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(figr, use_container_width=True)

st.caption("Orientation comes from the EKF (gravity + magnetometer aiding). Absolute "
           "speed is not observable by integrating a single torso IMU, so speed uses "
           "cadence x step-length — the method real chest straps use.")

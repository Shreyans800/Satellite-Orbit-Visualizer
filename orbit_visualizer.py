import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

R_EARTH = 6371
MU = 398600

def classify_orbit(altitude_km, inclination_deg):
    if abs(inclination_deg - 90) < 5:
        if 600 <= altitude_km <= 800:
            return "Sun-Synchronous Orbit (SSO)"
        return "Polar Orbit"
    elif abs(inclination_deg) < 5 and abs(altitude_km - 35786) < 200:
        return "Geostationary Orbit (GEO)"
    elif 160 <= altitude_km <= 2000:
        return "Low Earth Orbit (LEO)"
    elif 2000 < altitude_km < 35786:
        return "Medium Earth Orbit (MEO)"
    elif altitude_km > 35786:
        return "High Earth Orbit (HEO)"
    elif 200 <= altitude_km <= 35786:
        return "Geostationary Transfer Orbit (GTO)"
    else:
        return "Unclassified"

def generate_orbit(apoapsis_km, periapsis_km, inclination_deg, steps=200):
    a = (apoapsis_km + periapsis_km + 2 * R_EARTH) / 2
    e = (apoapsis_km - periapsis_km) / (apoapsis_km + periapsis_km + 2 * R_EARTH)
    inc_rad = np.radians(inclination_deg)

    theta = np.linspace(0, 2*np.pi, steps)
    theta = 2 * np.pi - theta  # Reverse direction

    r = (a * (1 - e**2)) / (1 + e * np.cos(theta))

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = y * np.sin(inc_rad)
    y = y * np.cos(inc_rad)

    period = 2 * np.pi * np.sqrt(a**3 / MU)
    min_alt = min(periapsis_km, apoapsis_km)
    max_alt = max(periapsis_km, apoapsis_km)
    orbit_type = classify_orbit((min_alt + max_alt) / 2, inclination_deg)

    return x, y, z, period / 60, (min_alt, max_alt), orbit_type

def create_earth_mesh():
    u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
    xs = R_EARTH * np.cos(u) * np.sin(v)
    ys = R_EARTH * np.sin(u) * np.sin(v)
    zs = R_EARTH * np.cos(v)
    return xs, ys, zs

def create_3d_orbit_figure(x, y, z, position_index=0):
    max_range = np.max(np.abs(np.concatenate([x, y, z]))) * 1.2
    xs, ys, zs = create_earth_mesh()

    default_camera = dict(
        eye=dict(x=1.25, y=1.25, z=1.25),
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0)
    )

    fig = go.Figure(
        data=[
            go.Surface(x=xs, y=ys, z=zs, colorscale='Blues', opacity=0.6, showscale=False, name='Earth'),
            go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='red', width=3), name='Orbit'),
            go.Scatter3d(x=[x[position_index]], y=[y[position_index]], z=[z[position_index]], mode='markers',
                         marker=dict(size=6, color='red', symbol='square'), name='Satellite')
        ],
        layout=go.Layout(
            title="3D Orbit Visualization",
            scene=dict(
                xaxis=dict(range=[-max_range, max_range], autorange=False, title='X (km)'),
                yaxis=dict(range=[-max_range, max_range], autorange=False, title='Y (km)'),
                zaxis=dict(range=[-max_range, max_range], autorange=False, title='Z (km)'),
                aspectmode='data',
                camera=default_camera,
            )
        )
    )
    return fig

# --- Streamlit app ---
st.set_page_config(page_title="Satellite Orbit Visualizer", layout="wide")
st.title("🛰️ Satellite Orbit Visualizer (2D & 3D)")

with st.sidebar:
    st.header("Input Parameters")
    periapsis = st.number_input("Periapsis Altitude (km)", min_value=0.0, value=200.0, step=10.0)
    apoapsis = st.number_input("Apoapsis Altitude (km)", min_value=0.0, value=300.0, step=10.0)
    inclination = st.slider("Inclination (°)", 0, 180, 0)
    show_3d = st.checkbox("Show 3D Orbit", value=True)

# Initialize session state storage
if "orbit_data" not in st.session_state:
    st.session_state.orbit_data = None

if st.button("Generate Orbit"):
    x, y, z, period_min, alt_range, orbit_type = generate_orbit(apoapsis, periapsis, inclination)
    st.session_state.orbit_data = {
        "x": x,
        "y": y,
        "z": z,
        "period_min": period_min,
        "alt_range": alt_range,
        "orbit_type": orbit_type
    }

if st.session_state.orbit_data:
    od = st.session_state.orbit_data
    st.subheader("🛰️ Orbit Summary")
    st.markdown(f"**Orbit Type:** {od['orbit_type']}")
    st.markdown(f"**Orbital Period:** {od['period_min']:.2f} minutes")
    st.markdown(f"**Altitude Range:** {od['alt_range'][0]} km to {od['alt_range'][1]} km")

    pos_idx = st.slider("Satellite Position on Orbit", 0, len(od["x"]) - 1, 0, step=1)

    if show_3d:
        fig3d = create_3d_orbit_figure(od["x"], od["y"], od["z"], pos_idx)
        st.plotly_chart(fig3d, use_container_width=True)
else:
    st.info("Please generate an orbit first using the inputs above.")

# Orbit reference table
st.subheader("📋 Orbit Type Reference Table")
orbit_table = pd.DataFrame({
    "Orbit Type": ["LEO", "MEO", "HEO", "GEO", "SSO", "Polar", "GTO", "Unclassified"],
    "Periapsis (km)": ["160-2000", "2000-35786", "Above 35786", "Approx 35786", "600-800", "Varies", "200-35786", "-"],
    "Apoapsis (km)": ["160-2000", "2000-35786", "Above 35786", "Approx 35786", "600-800", "Varies", "200-35786", "-"],
    "Inclination (deg)": ["0-90", "10-60", "Varies", "0", "Near 90", "Near 90", "Varies", "-"],
    "Description": [
        "Low Earth Orbit",
        "Medium Earth Orbit",
        "High Earth Orbit",
        "Geostationary Orbit",
        "Sun-Synchronous Orbit",
        "Polar Orbit",
        "Geostationary Transfer Orbit",
        "Outside standard classifications"
    ]
})
st.dataframe(orbit_table)

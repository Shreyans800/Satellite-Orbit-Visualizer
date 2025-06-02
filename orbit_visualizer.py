import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import warnings

warnings.filterwarnings("ignore", message="Thread 'MainThread': missing ScriptRunContext!")
st.set_page_config(page_title="Satellite Orbit Visualizer", layout="wide")

# Hide sidebar toggle button so sidebar can't be closed by user
st.markdown(
    """
    <style>
    button[data-testid="collapsed-control"] {
        display: none;
    }
    .css-1d391kg {
        min-width: 300px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# your existing sidebar and rest of app below...
with st.sidebar:
    st.header("Input Parameters")
    periapsis = st.number_input("Periapsis Altitude (km)", min_value=0.0, value=200.0, step=10.0)
    apoapsis = st.number_input("Apoapsis Altitude (km)", min_value=0.0, value=300.0, step=10.0)
    inclination = st.slider("Inclination (°)", 0, 180, 0)
    show_2d = st.checkbox("Show 2D Orbit", value=True)
    show_3d = st.checkbox("Show 3D Orbit", value=True)

R_EARTH = 6371  # Earth radius in km
MU = 398600     # Earth gravitational parameter km^3/s^2

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
    r = (a * (1 - e**2)) / (1 + e * np.cos(theta))

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = y * np.sin(inc_rad)
    y = y * np.cos(inc_rad)

    period = 2 * np.pi * np.sqrt(a**3 / MU)  # seconds
    min_alt = min(periapsis_km, apoapsis_km)
    max_alt = max(periapsis_km, apoapsis_km)
    orbit_type = classify_orbit((min_alt + max_alt) / 2, inclination_deg)

    return x, y, z, period/60, (min_alt, max_alt), orbit_type

def create_earth_mesh():
    u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
    xs = R_EARTH * np.cos(u) * np.sin(v)
    ys = R_EARTH * np.sin(u) * np.sin(v)
    zs = R_EARTH * np.cos(v)
    return xs, ys, zs

def create_3d_orbit_figure(x, y, z, sat_idx=0):
    max_range = np.max(np.abs(np.concatenate([x, y, z]))) * 1.2
    xs, ys, zs = create_earth_mesh()

    camera = dict(
        eye=dict(x=1.5, y=1.5, z=1.2)
    )

    fig = go.Figure(
        data=[
            go.Surface(x=xs, y=ys, z=zs, colorscale='Blues', opacity=0.6, showscale=False, name='Earth'),
            go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='red', width=3), name='Orbit'),
            go.Scatter3d(x=[x[sat_idx]], y=[y[sat_idx]], z=[z[sat_idx]], mode='markers',
                         marker=dict(size=6, color='red', symbol='square'), name='Satellite')
        ],
        layout=go.Layout(
            title="3D Orbit Visualization",
            scene=dict(
                xaxis=dict(range=[-max_range, max_range], autorange=False, title='X (km)'),
                yaxis=dict(range=[-max_range, max_range], autorange=False, title='Y (km)'),
                zaxis=dict(range=[-max_range, max_range], autorange=False, title='Z (km)'),
                aspectmode='data',
                camera=camera
            ),
            margin=dict(l=0, r=0, t=40, b=0)
        )
    )
    return fig

st.set_page_config(page_title="Satellite Orbit Visualizer", layout="wide")
st.title("🛰️ Satellite Orbit Visualizer (2D & 3D)")

# Sidebar inputs always visible, no expander
with st.sidebar:
    st.header("Input Parameters")
    periapsis = st.number_input("Periapsis Altitude (km)", min_value=0.0, value=200.0, step=10.0)
    apoapsis = st.number_input("Apoapsis Altitude (km)", min_value=0.0, value=300.0, step=10.0)
    inclination = st.slider("Inclination (°)", 0, 180, 0)
    show_2d = st.checkbox("Show 2D Orbit", value=True)
    show_3d = st.checkbox("Show 3D Orbit", value=True)

def plot_2d(x, y):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(6, 6))
    earth = Circle((0, 0), R_EARTH, color='blue', alpha=0.3)
    ax.add_patch(earth)
    ax.plot(x, y, 'r-', label='Orbit Path')
    ax.set_aspect('equal')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title("2D Orbit Visualization")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

if 'orbit_data' not in st.session_state:
    st.session_state.orbit_data = None

if st.button("Generate Orbit"):
    x, y, z, period_min, alt_range, orbit_type = generate_orbit(apoapsis, periapsis, inclination)
    st.session_state.orbit_data = {
        "x": x,
        "y": y,
        "z": z,
        "period_min": period_min,
        "alt_range": alt_range,
        "orbit_type": orbit_type,
    }

if st.session_state.orbit_data:
    od = st.session_state.orbit_data
    st.subheader("🛰️ Orbit Summary")
    st.markdown(f"**Orbit Type:** {od['orbit_type']}")
    st.markdown(f"**Orbital Period:** {od['period_min']:.2f} minutes")
    st.markdown(f"**Altitude Range:** {od['alt_range'][0]} km to {od['alt_range'][1]} km")

    if show_2d:
        plot_2d(od["x"], od["y"])

    # Slider BELOW 2D plot for satellite position from 0 to 360 degrees
    pos_deg = st.slider("Satellite Position (degrees)", 0, 360, 0, step=1)
    pos_idx = int((pos_deg / 360) * (len(od["x"]) - 1))

    if show_3d:
        fig3d = create_3d_orbit_figure(od["x"], od["y"], od["z"], pos_idx)
        st.plotly_chart(fig3d, use_container_width=True)

else:
    st.info("Please generate an orbit first using the inputs above.")

st.subheader("📋 Orbit Type Reference Table")
orbit_table = pd.DataFrame({
    "Orbit Type": ["LEO", "MEO", "HEO", "GEO", "SSO", "Polar", "GTO", "Unclassified"],
    "Periapsis (km)": [160, 2000, 35786, 35786, 600, "Varies (~90° inclination)", 200, "-"],
    "Apoapsis (km)": [2000, 35786, "100000", 35786, 800, "Varies (~90° inclination)", 35786, "-"],
})
st.table(orbit_table)

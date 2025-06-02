import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import warnings
warnings.filterwarnings("ignore", message="Thread 'MainThread': missing ScriptRunContext!")

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
    # Rotate orbit plane by inclination
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

def create_3d_orbit_animation(x, y, z):
    max_range = np.max(np.abs(np.concatenate([x, y, z]))) * 200

    xs, ys, zs = create_earth_mesh()

    # Fixed camera position and no zoom changes
    camera = dict(
        eye=dict(x=200 * max_range, y=200 * max_range, z=200 * max_range)
    )

    fig = go.Figure(
        data=[
            go.Surface(x=xs, y=ys, z=zs, colorscale='Blues', opacity=0.6, showscale=False, name='Earth'),
            go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='red', width=3), name='Orbit'),
            go.Scatter3d(x=[x[0]], y=[y[0]], z=[z[0]], mode='markers',
                         marker=dict(size=6, color='red', symbol='square'), name='Satellite')
        ],
        layout=go.Layout(
            title="3D Orbit Animation",
            scene=dict(
                xaxis=dict(range=[-max_range, max_range], autorange=False, title='X (km)'),
                yaxis=dict(range=[-max_range, max_range], autorange=False, title='Y (km)'),
                zaxis=dict(range=[-max_range, max_range], autorange=False, title='Z (km)'),
                aspectmode='data',
                camera=camera
            ),
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                y=1,
                x=0.8,
                xanchor='left',
                yanchor='bottom',
                buttons=[dict(label='▶ Play',
                              method='animate',
                              args=[None, {"frame": {"duration": 50, "redraw": True},
                                           "fromcurrent": True, "mode": "immediate"}])],
            )]
        )
    )

    frames = []
    for i in range(len(x)):
        frames.append(go.Frame(
            data=[go.Scatter3d(x=[x[i]], y=[y[i]], z=[z[i]], mode='markers',
                               marker=dict(size=6, color='red', symbol='square'))]
        ))
    # Add last frame to reset satellite position back to start (optional)
    frames.append(go.Frame(
        data=[go.Scatter3d(x=[x[0]], y=[y[0]], z=[z[0]], mode='markers',
                           marker=dict(size=6, color='red', symbol='square'))]
        def create_earth_mesh():
    u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
    xs = R_EARTH * np.cos(u) * np.sin(v)
    ys = R_EARTH * np.sin(u) * np.sin(v)
    zs = R_EARTH * np.cos(v)
    return xs, ys, zs
    ))

    fig.frames = frames
    return fig

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

# Streamlit UI
st.set_page_config(page_title="Satellite Orbit Visualizer", layout="wide")
st.title("🛰️ Satellite Orbit Visualizer (2D & 3D)")

with st.sidebar:
    st.header("Input Parameters")
    periapsis = st.number_input("Periapsis Altitude (km)", min_value=0.0, value=200.0, step=10.0)
    apoapsis = st.number_input("Apoapsis Altitude (km)", min_value=0.0, value=300.0, step=10.0)
    inclination = st.slider("Inclination (°)", 0, 180, 0)
    show_2d = st.checkbox("Show 2D Orbit", value=True)
    show_3d = st.checkbox("Show 3D Orbit with Animation", value=True)

if st.button("Generate Orbit"):
    x, y, z, period_min, alt_range, orbit_type = generate_orbit(apoapsis, periapsis, inclination)
    st.subheader("🛰️ Orbit Summary")
    st.markdown(f"**Orbit Type:** {orbit_type}")
    st.markdown(f"**Orbital Period:** {period_min:.2f} minutes")
    st.markdown(f"**Altitude Range:** {alt_range[0]} km to {alt_range[1]} km")

    df = pd.DataFrame({'X (km)': x, 'Y (km)': y, 'Z (km)': z})
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Orbit Data (CSV)", data=csv, file_name="orbit_data.csv", mime='text/csv')

    if show_2d:
        plot_2d(x, y)
    if show_3d:
        fig3d = create_3d_orbit_animation(x, y, z)
        st.plotly_chart(fig3d, use_container_width=True)

st.subheader("📋 Orbit Type Reference Table")
orbit_table = pd.DataFrame({
    "Orbit Type": ["LEO", "MEO", "HEO", "GEO", "SSO", "Polar", "GTO", "Unclassified"],
    "Periapsis (km)": [160, 2000, 35786, 35786, 600, "Varies ~ 90°", 200, "-"],
    "Apoapsis (km)": [2000, 35786, "100000", 35786, 800, "Varies ~ 90°", 35786, "-"],
})
st.table(orbit_table)

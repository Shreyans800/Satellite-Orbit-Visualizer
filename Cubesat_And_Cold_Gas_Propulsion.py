import streamlit as st
import numpy as np
import plotly.graph_objects as go
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy.time import TimeDelta
from astropy import units as u
import pandas as pd
import warnings

warnings.filterwarnings("ignore", message="Thread 'MainThread': missing ScriptRunContext!")

# --- Cold gas propulsion calculation ---
def cold_gas_thrust(m_dot, ve):
    thrust = m_dot * ve  # Newtons
    isp = ve / 9.80665  # seconds
    return thrust, isp

# --- Orbit simulation and 3D plot ---
def orbit_simulation(orbit_type, altitude_km=None, inclination_deg=None, periapsis_km=None, apoapsis_km=None):
    earth_radius_km = Earth.R.to(u.km).value

    if orbit_type == 'circular':
        orbit = Orbit.circular(Earth, alt=altitude_km * u.km, inc=inclination_deg * u.deg)
    elif orbit_type == 'elliptical':
        r_p = (earth_radius_km + periapsis_km) * u.km
        r_a = (earth_radius_km + apoapsis_km) * u.km
        a = (r_p + r_a) / 2
        ecc = (r_a - r_p) / (r_a + r_p)
        orbit = Orbit.from_classical(Earth, a, ecc, inclination_deg * u.deg, 0 * u.deg, 0 * u.deg, 0 * u.deg)
    else:
        st.error("Unsupported orbit type.")
        return None, None

    num_points = 100
    times = [orbit.epoch + TimeDelta(t) for t in np.linspace(0, orbit.period.to_value(u.s), num_points) * u.s]
    r_vectors = np.array([orbit.propagate(t - orbit.epoch).r.to_value(u.km) for t in times])

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=r_vectors[:, 0], y=r_vectors[:, 1], z=r_vectors[:, 2],
        mode='lines', line=dict(color='blue', width=3), name='Orbit Path'
    ))

    u_sphere = np.linspace(0, 2 * np.pi, 50)
    v_sphere = np.linspace(0, np.pi, 50)
    x_sphere = earth_radius_km * np.outer(np.cos(u_sphere), np.sin(v_sphere))
    y_sphere = earth_radius_km * np.outer(np.sin(u_sphere), np.sin(v_sphere))
    z_sphere = earth_radius_km * np.outer(np.ones_like(u_sphere), np.cos(v_sphere))

    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        colorscale='Blues',
        opacity=0.7,
        showscale=False,
        name='Earth'
    ))

    sat_pos = r_vectors[0]
    size = earth_radius_km * 0.02

    cube_verts = np.array([
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1]
    ]) * size + sat_pos

    cube_edges = [
        (0,1),(0,2),(0,4),
        (1,3),(1,5),
        (2,3),(2,6),
        (3,7),
        (4,5),(4,6),
        (5,7),
        (6,7)
    ]

    for edge in cube_edges:
        fig.add_trace(go.Scatter3d(
            x=cube_verts[list(edge),0],
            y=cube_verts[list(edge),1],
            z=cube_verts[list(edge),2],
            mode='lines',
            line=dict(color='red', width=5),
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        title="3D Orbit and CubeSat Visualization",
        scene=dict(
            xaxis_title='X (km)', yaxis_title='Y (km)', zaxis_title='Z (km)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    return fig, orbit.period.to(u.min).value

# --- Solar power calculation ---
def solar_power_calc(panel_area, efficiency, solar_irradiance=1361):
    power = panel_area * efficiency * solar_irradiance
    return power

# --- Power vs Time plot ---
def plot_power_vs_time(panel_area, efficiency, orbit_period_min):
    times_min = np.linspace(0, orbit_period_min, 100)
    eclipse_fraction = 0.35
    power = solar_power_calc(panel_area, efficiency)

    power_over_time = []
    for t in times_min:
        if orbit_period_min * (0.5 - eclipse_fraction / 2) <= t <= orbit_period_min * (0.5 + eclipse_fraction / 2):
            power_over_time.append(0)
        else:
            power_over_time.append(power)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times_min,
        y=power_over_time,
        mode='lines',
        line=dict(color='green'),
        name='Solar Power Output'
    ))

    fig.update_layout(
        title=f"Solar Power Output vs Time (Orbit Period {orbit_period_min:.2f} min)",
        xaxis_title="Time (minutes)",
        yaxis_title="Power Output (Watts)",
        height=500,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig

# --- Streamlit app starts here ---
st.title("CubeSat Design and Cold Gas Propulsion Simulator")

orbit_type = st.selectbox("Orbit Type", options=["circular", "elliptical"])

valid_orbit_params = True

if orbit_type == "circular":
    altitude = st.number_input("Orbit Altitude (km)", min_value=100, value=500)
    inclination = st.number_input("Orbit Inclination (deg)", min_value=0.0, max_value=180.0, value=97.6)
else:
    periapsis = st.number_input("Periapsis Altitude (km)", min_value=100, value=300)
    apoapsis = st.number_input("Apoapsis Altitude (km)", min_value=100, value=800)
    inclination = st.number_input("Orbit Inclination (deg)", min_value=0.0, max_value=180.0, value=97.6)
    if periapsis >= apoapsis:
        st.error("Periapsis must be less than Apoapsis.")
        valid_orbit_params = False

# Cold gas propulsion inputs
st.subheader("Cold Gas Propulsion Inputs")
default_gases = ["Hydrogen", "Nitrogen", "Argon", "Carbon Dioxide"]

gas_data = []
for gas_name in default_gases:
    st.markdown(f"**{gas_name}**")
    m_dot = st.number_input(f"Mass Flow Rate (kg/s) for {gas_name}", min_value=0.0, format="%.6f", value=0.0001, key=f"mdot_{gas_name}")
    ve = st.number_input(f"Exhaust Velocity (m/s) for {gas_name}", min_value=0.0, value=1000.0, key=f"ve_{gas_name}")
    gas_data.append({"name": gas_name, "m_dot": m_dot, "ve": ve})

# Solar panel inputs
st.subheader("Solar Panel Parameters")
panel_area = st.number_input("Solar Panel Area (m²)", min_value=0.0, value=0.1)
efficiency = st.number_input("Solar Panel Efficiency (0 to 1)", min_value=0.0, max_value=1.0, value=0.28)

# Run simulation
if st.button("Run Simulation"):
    if not valid_orbit_params:
        st.error("Cannot run simulation: periapsis must be less than apoapsis.")
    else:
        # Propulsion table
        table_data = []
        for gas in gas_data:
            thrust, isp = cold_gas_thrust(gas["m_dot"], gas["ve"])
            gas["thrust"] = thrust
            gas["isp"] = isp
            table_data.append([
                gas["name"], gas["m_dot"], gas["ve"], thrust, isp
            ])

        df = pd.DataFrame(table_data, columns=["Gas", "Mass Flow Rate (kg/s)", "Exhaust Velocity (m/s)", "Thrust (N)", "Specific Impulse (s)"])
        st.subheader("Cold Gas Propulsion Data Table")
        st.dataframe(df)

        # Orbit simulation
        try:
            if orbit_type == "circular":
                orbit_fig, orbit_period = orbit_simulation(orbit_type, altitude_km=altitude, inclination_deg=inclination)
            else:
                orbit_fig, orbit_period = orbit_simulation(orbit_type, periapsis_km=periapsis, apoapsis_km=apoapsis, inclination_deg=inclination)
        except Exception as e:
            st.error(f"Orbit simulation error: {e}")
            orbit_fig, orbit_period = None, None

        if orbit_fig is not None:
            st.plotly_chart(orbit_fig, use_container_width=True)
        else:
            st.warning("Orbit plot could not be generated.")

        if orbit_period is None:
            orbit_period = 90  # fallback value

        # Solar power plot
        power_fig = plot_power_vs_time(panel_area, efficiency, orbit_period)
        st.plotly_chart(power_fig, use_container_width=True)

        # CSV download
        csv = df.to_csv(index=False)
        st.download_button("Download Propulsion Data as CSV", data=csv, file_name="propulsion_data.csv", mime="text/csv")

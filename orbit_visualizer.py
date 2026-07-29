import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Satellite Orbit Visualizer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONSTANTS
# ============================================================

R_EARTH = 6371.0       # Earth radius in km
MU = 398600.4418       # Earth's gravitational parameter km^3/s^2


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Hide Streamlit sidebar collapse buttons */
    button[data-testid="collapsedControl"] {
        display: none !important;
    }

    button[data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }

    /* Keep sidebar visible */
    [data-testid="stSidebar"] {
        min-width: 300px !important;
        width: 300px !important;
        transform: none !important;
    }

    /* Main page */
    .main {
        padding-top: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ORBIT CLASSIFICATION
# ============================================================

def classify_orbit(
    periapsis_km,
    apoapsis_km,
    inclination_deg
):

    average_altitude = (
        periapsis_km + apoapsis_km
    ) / 2

    altitude_difference = (
        apoapsis_km - periapsis_km
    )

    # GEO
    if (
        abs(inclination_deg) <= 5
        and abs(periapsis_km - 35786) <= 200
        and abs(apoapsis_km - 35786) <= 200
    ):
        return "Geostationary Orbit (GEO)"

    # SSO
    if (
        96 <= inclination_deg <= 100
        and 600 <= periapsis_km <= 800
        and 600 <= apoapsis_km <= 800
    ):
        return "Sun-Synchronous Orbit (SSO)"

    # Polar
    if (
        85 <= inclination_deg <= 95
        and not (
            600 <= periapsis_km <= 800
            and 600 <= apoapsis_km <= 800
        )
    ):
        return "Polar Orbit"

    # GTO
    if (
        periapsis_km <= 1000
        and apoapsis_km >= 30000
        and apoapsis_km <= 40000
    ):
        return "Geostationary Transfer Orbit (GTO)"

    # LEO
    if (
        160 <= average_altitude <= 2000
        and altitude_difference <= 500
    ):
        return "Low Earth Orbit (LEO)"

    # MEO
    if (
        2000 < average_altitude < 35786
        and altitude_difference <= 10000
    ):
        return "Medium Earth Orbit (MEO)"

    # HEO
    if (
        apoapsis_km > 35786
        and altitude_difference > 10000
    ):
        return "High Earth Orbit (HEO)"

    return "Unclassified"


# ============================================================
# GENERATE ORBIT
# ============================================================

def generate_orbit(
    apoapsis_km,
    periapsis_km,
    inclination_deg,
    steps=360
):

    # Ensure apoapsis is larger
    if apoapsis_km < periapsis_km:
        apoapsis_km, periapsis_km = (
            periapsis_km,
            apoapsis_km
        )

    # Distances from Earth's center
    r_a = R_EARTH + apoapsis_km
    r_p = R_EARTH + periapsis_km

    # Semi-major axis
    a = (r_a + r_p) / 2

    # Eccentricity
    e = (r_a - r_p) / (r_a + r_p)

    # Inclination
    inc = np.radians(inclination_deg)

    # True anomaly
    theta = np.linspace(
        0,
        2 * np.pi,
        steps,
        endpoint=False
    )

    # Orbital radius
    r = (
        a * (1 - e**2)
        / (1 + e * np.cos(theta))
    )

    # Orbit in orbital plane
    x = r * np.cos(theta)
    y_orbital = r * np.sin(theta)

    # Rotate orbit by inclination
    y = y_orbital * np.cos(inc)
    z = y_orbital * np.sin(inc)

    # Orbital period
    period_seconds = (
        2
        * np.pi
        * np.sqrt(a**3 / MU)
    )

    period_minutes = (
        period_seconds / 60
    )

    orbit_type = classify_orbit(
        periapsis_km,
        apoapsis_km,
        inclination_deg
    )

    return {
        "x": x,
        "y": y,
        "z": z,
        "theta": theta,
        "period_min": float(period_minutes),
        "periapsis": float(periapsis_km),
        "apoapsis": float(apoapsis_km),
        "inclination": float(inclination_deg),
        "orbit_type": orbit_type
    }


# ============================================================
# EARTH FOR 3D
# ============================================================

def create_earth_mesh():

    u, v = np.mgrid[
        0:2 * np.pi:50j,
        0:np.pi:25j
    ]

    x = (
        R_EARTH
        * np.cos(u)
        * np.sin(v)
    )

    y = (
        R_EARTH
        * np.sin(u)
        * np.sin(v)
    )

    z = (
        R_EARTH
        * np.cos(v)
    )

    return x, y, z


# ============================================================
# 2D ORBIT
# ============================================================

def create_2d_plot(
    x,
    y,
    sat_idx
):

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    # Earth
    earth = Circle(
        (0, 0),
        R_EARTH,
        alpha=0.7
    )

    ax.add_patch(earth)

    # Orbit
    ax.plot(
        x,
        y,
        linewidth=2,
        label="Orbit"
    )

    # Satellite
    ax.scatter(
        x[sat_idx],
        y[sat_idx],
        s=100,
        marker="s",
        label="Satellite"
    )

    # Calculate fixed range
    max_range = max(
        np.max(np.abs(x)),
        np.max(np.abs(y))
    ) * 1.15

    ax.set_xlim(
        -max_range,
        max_range
    )

    ax.set_ylim(
        -max_range,
        max_range
    )

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    ax.set_xlabel(
        "X Position (km)"
    )

    ax.set_ylabel(
        "Y Position (km)"
    )

    ax.set_title(
        "2D Satellite Orbit"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    return fig


# ============================================================
# 3D ORBIT
# ============================================================

def create_3d_plot(
    x,
    y,
    z,
    sat_idx
):

    earth_x, earth_y, earth_z = (
        create_earth_mesh()
    )

    # Fixed axis range
    max_radius = max(
        np.max(np.abs(x)),
        np.max(np.abs(y)),
        np.max(np.abs(z))
    )

    axis_range = (
        max_radius * 1.15
    )

    # Fixed camera
    camera = dict(
        eye=dict(
            x=1.5,
            y=1.5,
            z=1.2
        ),
        center=dict(
            x=0,
            y=0,
            z=0
        )
    )

    fig = go.Figure()

    # --------------------------------------------------------
    # EARTH
    # --------------------------------------------------------

    fig.add_trace(
        go.Surface(
            x=earth_x,
            y=earth_y,
            z=earth_z,
            colorscale="Blues",
            opacity=0.85,
            showscale=False,
            name="Earth",
            hoverinfo="skip"
        )
    )

    # --------------------------------------------------------
    # ORBIT
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line=dict(
                width=4
            ),
            name="Orbit"
        )
    )

    # --------------------------------------------------------
    # RED CUBE SATELLITE
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter3d(
            x=[x[sat_idx]],
            y=[y[sat_idx]],
            z=[z[sat_idx]],
            mode="markers",
            marker=dict(
                size=9,
                color="red",
                symbol="square"
            ),
            name="Satellite"
        )
    )

    # --------------------------------------------------------
    # LAYOUT
    # --------------------------------------------------------

    fig.update_layout(

        title="3D Satellite Orbit",

        margin=dict(
            l=0,
            r=0,
            t=50,
            b=0
        ),

        scene=dict(

            # Fixed camera
            camera=camera,

            # Fixed axis ranges
            xaxis=dict(
                title="X (km)",
                range=[
                    -axis_range,
                    axis_range
                ],
                autorange=False
            ),

            yaxis=dict(
                title="Y (km)",
                range=[
                    -axis_range,
                    axis_range
                ],
                autorange=False
            ),

            zaxis=dict(
                title="Z (km)",
                range=[
                    -axis_range,
                    axis_range
                ],
                autorange=False
            ),

            # Prevent automatic zooming
            aspectmode="cube",

            dragmode="orbit"
        )
    )

    return fig


# ============================================================
# SESSION STATE
# ============================================================

if "orbit_data" not in st.session_state:

    st.session_state.orbit_data = None


if "simulation_running" not in st.session_state:

    st.session_state.simulation_running = False


if "satellite_position" not in st.session_state:

    st.session_state.satellite_position = 0


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🛰️ Input Parameters"
    )

    periapsis = st.number_input(
        "Periapsis Altitude (km)",
        min_value=0.0,
        value=200.0,
        step=10.0
    )

    apoapsis = st.number_input(
        "Apoapsis Altitude (km)",
        min_value=0.0,
        value=300.0,
        step=10.0
    )

    inclination = st.slider(
        "Orbit Inclination (°)",
        min_value=0,
        max_value=180,
        value=0,
        step=1
    )

    show_2d = st.checkbox(
        "Show 2D Orbit",
        value=True
    )

    show_3d = st.checkbox(
        "Show 3D Orbit",
        value=True
    )

    generate_button = st.button(
        "Generate Orbit",
        use_container_width=True
    )


# ============================================================
# TITLE
# ============================================================

st.title(
    "🛰️ Satellite Orbit Visualizer"
)

st.write(
    "Visualize circular and elliptical satellite "
    "orbits around Earth in 2D and 3D."
)


# ============================================================
# GENERATE ORBIT
# ============================================================

if generate_button:

    if periapsis > apoapsis:

        st.warning(
            "Periapsis was greater than apoapsis. "
            "The values have been automatically swapped."
        )

    st.session_state.orbit_data = generate_orbit(
        apoapsis,
        periapsis,
        inclination
    )

    # Reset satellite
    st.session_state.satellite_position = 0

    # Stop previous simulation
    st.session_state.simulation_running = False


# ============================================================
# DISPLAY ORBIT
# ============================================================

if st.session_state.orbit_data is not None:

    orbit = st.session_state.orbit_data

    x = orbit["x"]
    y = orbit["y"]
    z = orbit["z"]

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "🛰️ Orbit Summary"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Orbit Type",
            orbit["orbit_type"]
        )

    with col2:

        st.metric(
            "Orbital Period",
            f"{orbit['period_min']:.2f} min"
        )

    with col3:

        st.metric(
            "Altitude",
            f"{float(orbit['periapsis']):.0f} - "
            f"{float(orbit['apoapsis']):.0f} km"
        )


    # --------------------------------------------------------
    # POSITION
    # --------------------------------------------------------

    position_deg = st.slider(
        "Satellite Position (°)",
        min_value=0,
        max_value=360,
        value=st.session_state.satellite_position,
        step=1,
        key="position_slider"
    )

    # Update session state
    st.session_state.satellite_position = (
        position_deg
    )

    # Convert 0-360 degrees to orbit index
    if position_deg == 360:

        sat_idx = 0

    else:

        sat_idx = int(
            (
                position_deg
                / 360
            )
            * len(x)
        )

        sat_idx = min(
            sat_idx,
            len(x) - 1
        )


    # --------------------------------------------------------
    # 2D
    # --------------------------------------------------------

    if show_2d:

        st.subheader(
            "🌍 2D Orbit Visualization"
        )

        fig_2d = create_2d_plot(
            x,
            y,
            sat_idx
        )

        st.pyplot(
            fig_2d,
            clear_figure=True
        )


    # --------------------------------------------------------
    # SIMULATION CONTROLS
    # --------------------------------------------------------

    st.subheader(
        "🎮 Simulation Controls"
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "▶ Start Simulation",
            use_container_width=True
        ):

            st.session_state.simulation_running = True


    with col2:

        if st.button(
            "⏹ Stop Simulation",
            use_container_width=True
        ):

            st.session_state.simulation_running = False


    # --------------------------------------------------------
    # SIMULATION STATUS
    # --------------------------------------------------------

    if st.session_state.simulation_running:

        st.success(
            "🟢 Simulation Running"
        )

    else:

        st.info(
            "🔴 Simulation Stopped"
        )


    # --------------------------------------------------------
    # 3D
    # --------------------------------------------------------

    if show_3d:

        st.subheader(
            "🌍 3D Orbit Visualization"
        )

        fig_3d = create_3d_plot(
            x,
            y,
            z,
            sat_idx
        )

        st.plotly_chart(
            fig_3d,
            use_container_width=True,
            key="orbit_3d"
        )


    # --------------------------------------------------------
    # CSV DOWNLOAD
    # --------------------------------------------------------

    csv_df = pd.DataFrame(
        {
            "Angle (degrees)": np.arange(
                0,
                360
            ),

            "X (km)": x,

            "Y (km)": y,

            "Z (km)": z
        }
    )

    csv_data = csv_df.to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        "📥 Download Orbit Data (CSV)",
        data=csv_data,
        file_name="satellite_orbit_data.csv",
        mime="text/csv"
    )


    # --------------------------------------------------------
    # AUTOMATIC SIMULATION
    # --------------------------------------------------------

    if st.session_state.simulation_running:

        # Increase position
        next_position = (
            st.session_state.satellite_position
            + 2
        )

        # Complete revolution
        if next_position >= 360:

            next_position = 0

        st.session_state.satellite_position = (
            next_position
        )

        # Animation speed
        time.sleep(
            0.05
        )

        # Rerun application
        st.rerun()


else:

    st.info(
        "Enter your orbit parameters in the sidebar "
        "and click 'Generate Orbit'."
    )


# ============================================================
# ORBIT REFERENCE TABLE
# ============================================================

st.subheader(
    "📋 Orbit Type Reference Table"
)

orbit_table = pd.DataFrame(
    {
        "Orbit Type": [
            "LEO",
            "MEO",
            "HEO",
            "GEO",
            "SSO",
            "Polar",
            "GTO",
            "Unclassified"
        ],

        "Periapsis (km)": [
            "160 - 2,000",
            "2,000 - 35,786",
            "> 2,000",
            "35,786",
            "600 - 800",
            "Varies",
            "≈ 200",
            "-"
        ],

        "Apoapsis (km)": [
            "160 - 2,000",
            "2,000 - 35,786",
            "> 35,786",
            "35,786",
            "600 - 800",
            "Varies",
            "≈ 35,786",
            "-"
        ]
    }
)

st.table(
    orbit_table
)

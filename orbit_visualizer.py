import numpy as np
import pandas as pd
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
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Hide sidebar collapse buttons */
    button[data-testid="collapsedControl"] {
        display: none !important;
    }

    button[data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }

    /* Keep sidebar width */
    [data-testid="stSidebar"] {
        min-width: 300px !important;
        width: 300px !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

R_EARTH = 6371.0
MU = 398600.4418


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
    ):
        return "Polar Orbit"

    # GTO
    if (
        periapsis_km <= 1000
        and 30000 <= apoapsis_km <= 40000
    ):
        return "Geostationary Transfer Orbit (GTO)"

    # LEO
    if (
        160 <= average_altitude <= 2000
    ):
        return "Low Earth Orbit (LEO)"

    # MEO
    if (
        2000 < average_altitude < 35786
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

    apoapsis_km = float(apoapsis_km)
    periapsis_km = float(periapsis_km)
    inclination_deg = float(inclination_deg)

    # Swap if necessary
    if periapsis_km > apoapsis_km:

        periapsis_km, apoapsis_km = (
            apoapsis_km,
            periapsis_km
        )

    # Distance from Earth's center
    r_p = R_EARTH + periapsis_km
    r_a = R_EARTH + apoapsis_km

    # Semi-major axis
    a = (
        r_p + r_a
    ) / 2

    # Eccentricity
    e = (
        r_a - r_p
    ) / (
        r_a + r_p
    )

    # Inclination
    inc = np.radians(
        inclination_deg
    )

    # 360 complete positions
    theta = np.linspace(
        0,
        2 * np.pi,
        steps,
        endpoint=False
    )

    # Radius
    r = (
        a * (1 - e ** 2)
        / (
            1
            + e * np.cos(theta)
        )
    )

    # Orbital plane
    x = r * np.cos(theta)

    y_orbital = (
        r * np.sin(theta)
    )

    # Rotate by inclination
    y = (
        y_orbital
        * np.cos(inc)
    )

    z = (
        y_orbital
        * np.sin(inc)
    )

    # Orbital period
    period_seconds = (
        2
        * np.pi
        * np.sqrt(
            a ** 3 / MU
        )
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
        "period_min": float(
            period_minutes
        ),
        "periapsis": float(
            periapsis_km
        ),
        "apoapsis": float(
            apoapsis_km
        ),
        "inclination": float(
            inclination_deg
        ),
        "orbit_type": orbit_type
    }


# ============================================================
# EARTH
# ============================================================

def create_earth():

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
# 2D PLOT
# ============================================================

def create_2d_plot(
    x,
    y,
    sat_idx
):

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    # Earth
    earth = Circle(
        (0, 0),
        R_EARTH,
        alpha=0.7
    )

    ax.add_patch(
        earth
    )

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

    max_range = max(
        np.max(
            np.abs(x)
        ),
        np.max(
            np.abs(y)
        )
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
        "equal"
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
# 3D ANIMATED PLOT
# ============================================================

def create_animated_3d_plot(
    x,
    y,
    z
):

    earth_x, earth_y, earth_z = (
        create_earth()
    )

    # Fixed camera / fixed zoom
    max_radius = max(
        np.max(
            np.abs(x)
        ),
        np.max(
            np.abs(y)
        ),
        np.max(
            np.abs(z)
        )
    )

    axis_limit = (
        max_radius * 1.20
    )

    fixed_camera = dict(
        eye=dict(
            x=1.5,
            y=1.5,
            z=1.2
        ),
        center=dict(
            x=0,
            y=0,
            z=0
        ),
        up=dict(
            x=0,
            y=0,
            z=1
        )
    )

    # ========================================================
    # INITIAL FIGURE
    # ========================================================

    fig = go.Figure(

        data=[

            # EARTH
            go.Surface(
                x=earth_x,
                y=earth_y,
                z=earth_z,
                colorscale="Blues",
                opacity=0.85,
                showscale=False,
                name="Earth",
                hoverinfo="skip"
            ),

            # ORBIT
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(
                    width=4,
                    color="red"
                ),
                name="Orbit"
            ),

            # SATELLITE
            go.Scatter3d(
                x=[x[0]],
                y=[y[0]],
                z=[z[0]],
                mode="markers",
                marker=dict(
                    size=10,
                    color="red",
                    symbol="square"
                ),
                name="Satellite"
            )
        ]
    )

    # ========================================================
    # ANIMATION FRAMES
    # ========================================================

    frames = []

    for i in range(
        len(x)
    ):

        frame = go.Frame(

            name=str(i),

            data=[

                # Satellite only
                # Earth and orbit stay fixed
                go.Scatter3d(
                    x=[x[i]],
                    y=[y[i]],
                    z=[z[i]],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color="red",
                        symbol="square"
                    )
                )
            ],

            traces=[2]
        )

        frames.append(
            frame
        )

    fig.frames = frames

    # ========================================================
    # PLAY / STOP BUTTONS
    # ========================================================

    fig.update_layout(

        title=(
            "3D Satellite Orbit Simulation"
        ),

        margin=dict(
            l=0,
            r=0,
            t=50,
            b=0
        ),

        scene=dict(

            camera=fixed_camera,

            # IMPORTANT:
            # Fixed ranges prevent automatic zoom
            xaxis=dict(
                title="X (km)",
                range=[
                    -axis_limit,
                    axis_limit
                ],
                autorange=False
            ),

            yaxis=dict(
                title="Y (km)",
                range=[
                    -axis_limit,
                    axis_limit
                ],
                autorange=False
            ),

            zaxis=dict(
                title="Z (km)",
                range=[
                    -axis_limit,
                    axis_limit
                ],
                autorange=False
            ),

            aspectmode="cube"
        ),

        # ====================================================
        # ANIMATION CONTROLS
        # ====================================================

        updatemenus=[

            dict(

                type="buttons",

                showactive=False,

                direction="left",

                x=0.1,

                y=1.12,

                buttons=[

                    # START
                    dict(

                        label="▶ Start Simulation",

                        method="animate",

                        args=[

                            None,

                            dict(

                                frame=dict(
                                    duration=50,
                                    redraw=True
                                ),

                                transition=dict(
                                    duration=0
                                ),

                                fromcurrent=True,

                                mode="immediate"
                            )
                        ]
                    ),

                    # STOP
                    dict(

                        label="⏹ Stop Simulation",

                        method="animate",

                        args=[

                            [None],

                            dict(

                                frame=dict(
                                    duration=0,
                                    redraw=False
                                ),

                                transition=dict(
                                    duration=0
                                ),

                                mode="immediate"
                            )
                        ]
                    )
                ]
            )
        ],

        # ====================================================
        # SLIDER
        # ====================================================

        sliders=[

            dict(

                active=0,

                x=0.1,

                y=0.02,

                len=0.8,

                currentvalue=dict(
                    prefix="Position: "
                ),

                steps=[

                    dict(

                        label=f"{i}°",

                        method="animate",

                        args=[

                            [str(i)],

                            dict(

                                mode="immediate",

                                frame=dict(
                                    duration=0,
                                    redraw=True
                                ),

                                transition=dict(
                                    duration=0
                                )
                            )
                        ]
                    )

                    for i in range(
                        360
                    )
                ]
            )
        ]
    )

    return fig


# ============================================================
# SESSION STATE
# ============================================================

if "orbit_data" not in st.session_state:

    st.session_state.orbit_data = None


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
        value=0
    )

    show_2d = st.checkbox(
        "Show 2D Orbit",
        value=True
    )

    show_3d = st.checkbox(
        "Show 3D Orbit",
        value=True
    )

    generate = st.button(
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
# GENERATE
# ============================================================

if generate:

    st.session_state.orbit_data = (
        generate_orbit(
            apoapsis,
            periapsis,
            inclination
        )
    )


# ============================================================
# MAIN APP
# ============================================================

if st.session_state.orbit_data is not None:

    orbit = (
        st.session_state.orbit_data
    )

    x = orbit["x"]
    y = orbit["y"]
    z = orbit["z"]

    # ========================================================
    # SUMMARY
    # ========================================================

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
            f"{orbit['period_min']:.2f} minutes"
        )

    with col3:

        st.metric(
            "Altitude Range",
            (
                f"{orbit['periapsis']:.0f} - "
                f"{orbit['apoapsis']:.0f} km"
            )
        )


    # ========================================================
    # 2D
    # ========================================================

    if show_2d:

        st.subheader(
            "🌍 2D Orbit Visualization"
        )

        # Initial satellite location
        fig_2d = create_2d_plot(
            x,
            y,
            0
        )

        st.pyplot(
            fig_2d,
            clear_figure=True
        )


    # ========================================================
    # 3D
    # ========================================================

    if show_3d:

        st.subheader(
            "🌍 3D Orbit Simulation"
        )

        st.info(
            "Use ▶ Start Simulation to move the satellite. "
            "Use ⏹ Stop Simulation to pause it. "
            "The camera and zoom remain fixed."
        )

        fig_3d = (
            create_animated_3d_plot(
                x,
                y,
                z
            )
        )

        st.plotly_chart(
            fig_3d,
            use_container_width=True,
            config={
                "scrollZoom": True,
                "displayModeBar": True,
                "displaylogo": False
            }
        )


    # ========================================================
    # CSV
    # ========================================================

    st.subheader(
        "📥 Download Orbit Data"
    )

    csv_df = pd.DataFrame(
        {
            "Angle (degrees)": np.arange(
                360
            ),

            "X (km)": x,

            "Y (km)": y,

            "Z (km)": z
        }
    )

    csv_data = (
        csv_df
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )

    st.download_button(
        "📥 Download Orbit CSV",
        data=csv_data,
        file_name=(
            "satellite_orbit_data.csv"
        ),
        mime="text/csv"
    )


else:

    st.info(
        "Enter the orbit parameters in the sidebar "
        "and click Generate Orbit."
    )


# ============================================================
# REFERENCE TABLE
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

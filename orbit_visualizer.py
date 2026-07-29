import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import warnings

warnings.filterwarnings(
    "ignore",
    message="Thread 'MainThread': missing ScriptRunContext!"
)

st.set_page_config(
    page_title="Satellite Orbit Visualizer",
    layout="wide"
)

R_EARTH = 6371
MU = 398600


def classify_orbit(altitude_km, inclination_deg):
    if abs(inclination_deg - 90) < 5:
        if 600 <= altitude_km <= 800:
            return "Sun-Synchronous Orbit (SSO)"
        return "Polar Orbit"

    if abs(inclination_deg) < 5 and abs(altitude_km - 35786) < 200:
        return "Geostationary Orbit (GEO)"

    if 160 <= altitude_km <= 2000:
        return "Low Earth Orbit (LEO)"

    if 2000 < altitude_km < 35786:
        return "Medium Earth Orbit (MEO)"

    if altitude_km > 35786:
        return "High Earth Orbit (HEO)"

    if 200 <= altitude_km <= 35786:
        return "Geostationary Transfer Orbit (GTO)"

    return "Unclassified"


def generate_orbit(apoapsis_km, periapsis_km, inclination_deg, steps=360):
    if periapsis_km > apoapsis_km:
        periapsis_km, apoapsis_km = apoapsis_km, periapsis_km

    a = (apoapsis_km + periapsis_km + 2 * R_EARTH) / 2

    e = (
        (apoapsis_km - periapsis_km)
        / (apoapsis_km + periapsis_km + 2 * R_EARTH)
    )

    inc = np.radians(inclination_deg)
    theta = np.linspace(0, 2 * np.pi, steps, endpoint=False)

    r = (a * (1 - e**2)) / (1 + e * np.cos(theta))

    x = r * np.cos(theta)
    y_orbit = r * np.sin(theta)

    y = y_orbit * np.cos(inc)
    z = y_orbit * np.sin(inc)

    period = 2 * np.pi * np.sqrt(a**3 / MU) / 60

    min_alt = min(periapsis_km, apoapsis_km)
    max_alt = max(periapsis_km, apoapsis_km)

    orbit_type = classify_orbit(
        (min_alt + max_alt) / 2,
        inclination_deg
    )

    return x, y, z, period, (min_alt, max_alt), orbit_type


def create_earth_mesh():
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]

    x = R_EARTH * np.cos(u) * np.sin(v)
    y = R_EARTH * np.sin(u) * np.sin(v)
    z = R_EARTH * np.cos(v)

    return x, y, z


def create_3d_orbit_figure(x, y, z):
    max_range = np.max(
        np.abs(np.concatenate([x, y, z]))
    ) * 1.15

    earth_x, earth_y, earth_z = create_earth_mesh()

    camera = dict(
        eye=dict(x=1.5, y=1.5, z=1.2),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=0, z=1)
    )

    fig = go.Figure(
        data=[
            go.Surface(
                x=earth_x,
                y=earth_y,
                z=earth_z,
                colorscale="Blues",
                opacity=0.65,
                showscale=False,
                name="Earth"
            ),

            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(
                    color="red",
                    width=3
                ),
                name="Orbit"
            ),

            go.Scatter3d(
                x=[x[0]],
                y=[y[0]],
                z=[z[0]],
                mode="markers",
                marker=dict(
                    size=8,
                    color="red",
                    symbol="square"
                ),
                name="Satellite"
            )
        ]
    )

    frames = [
        go.Frame(
            name=str(i),
            data=[
                go.Scatter3d(
                    x=[x[i]],
                    y=[y[i]],
                    z=[z[i]],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color="red",
                        symbol="square"
                    )
                )
            ],
            traces=[2]
        )
        for i in range(len(x))
    ]

    fig.frames = frames

    fig.update_layout(
        title=dict(
            text="3D Orbit Visualization",
            x=0.5,
            xanchor="center"
        ),

        margin=dict(
            l=0,
            r=0,
            t=90,
            b=0
        ),

        scene=dict(
            camera=camera,

            xaxis=dict(
                range=[-max_range, max_range],
                autorange=False,
                title="X (km)"
            ),

            yaxis=dict(
                range=[-max_range, max_range],
                autorange=False,
                title="Y (km)"
            ),

            zaxis=dict(
                range=[-max_range, max_range],
                autorange=False,
                title="Z (km)"
            ),

            aspectmode="cube"
        ),

        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                direction="left",
                x=0.5,
                xanchor="center",
                y=1.20,
                yanchor="top",

                buttons=[
                    dict(
                        label="▶ Start Simulation",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {
                                    "duration": 50,
                                    "redraw": True
                                },
                                "transition": {
                                    "duration": 0
                                },
                                "fromcurrent": True,
                                "mode": "immediate"
                            }
                        ]
                    ),

                    dict(
                        label="⏹ Stop Simulation",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {
                                    "duration": 0,
                                    "redraw": False
                                },
                                "transition": {
                                    "duration": 0
                                },
                                "mode": "immediate"
                            }
                        ]
                    )
                ]
            )
        ]
    )

    return fig


def plot_2d(x, y):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(6, 6))

    earth = Circle(
        (0, 0),
        R_EARTH,
        color="blue",
        alpha=0.3
    )

    ax.add_patch(earth)

    ax.plot(
        x,
        y,
        "r-",
        label="Orbit Path"
    )

    ax.set_aspect("equal")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_title("2D Orbit Visualization")
    ax.legend()
    ax.grid(True)

    st.pyplot(
        fig,
        clear_figure=True
    )


st.title(
    "🛰️ Satellite Orbit Visualizer (2D & 3D)"
)


with st.sidebar:
    st.header("Input Parameters")

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
        "Inclination (°)",
        0,
        180,
        0
    )

    show_2d = st.checkbox(
        "Show 2D Orbit",
        value=True
    )

    show_3d = st.checkbox(
        "Show 3D Orbit",
        value=True
    )


if "orbit_data" not in st.session_state:
    st.session_state.orbit_data = None


if st.button("Generate Orbit"):
    (
        x,
        y,
        z,
        period_min,
        alt_range,
        orbit_type
    ) = generate_orbit(
        apoapsis,
        periapsis,
        inclination
    )

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

    st.markdown(
        f"**Orbit Type:** {od['orbit_type']}"
    )

    st.markdown(
        f"**Orbital Period:** "
        f"{od['period_min']:.2f} minutes"
    )

    st.markdown(
        f"**Altitude Range:** "
        f"{od['alt_range'][0]:.0f} km "
        f"to "
        f"{od['alt_range'][1]:.0f} km"
    )

    if show_2d:
        plot_2d(
            od["x"],
            od["y"]
        )

    if show_3d:
        fig3d = create_3d_orbit_figure(
            od["x"],
            od["y"],
            od["z"]
        )

        st.plotly_chart(
            fig3d,
            use_container_width=True,
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "scrollZoom": True
            }
        )

else:
    st.info(
        "Please generate an orbit first "
        "using the inputs above."
    )


st.subheader(
    "📋 Orbit Type Reference Table"
)

orbit_table = pd.DataFrame({
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
        160,
        2000,
        35786,
        35786,
        600,
        "Varies (~90° inclination)",
        200,
        "-"
    ],

    "Apoapsis (km)": [
        2000,
        35786,
        "100000",
        35786,
        800,
        "Varies (~90° inclination)",
        35786,
        "-"
    ]
})

st.table(
    orbit_table
)

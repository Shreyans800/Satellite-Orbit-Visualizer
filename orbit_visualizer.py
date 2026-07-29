import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import warnings

warnings.filterwarnings(
"ignore",
message="Thread 'MainThread': missing ScriptRunContext!"
)

# ============================================================

# CONSTANTS

# ============================================================

R_EARTH = 6371.0
MU = 398600.0

# ============================================================

# ORBIT CLASSIFICATION

# ============================================================

def classify_orbit(altitude_km, inclination_deg):

```
if abs(inclination_deg - 90) < 5:

    if 600 <= altitude_km <= 800:
        return "Sun-Synchronous Orbit (SSO)"

    else:
        return "Polar Orbit"

elif (
    abs(inclination_deg) < 5
    and abs(altitude_km - 35786) < 200
):

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
```

# ============================================================

# GENERATE ORBIT

# ============================================================

def generate_orbit(
apoapsis_km,
periapsis_km,
inclination_deg,
steps=360
):

```
# Make sure values are numbers
apoapsis_km = float(apoapsis_km)
periapsis_km = float(periapsis_km)
inclination_deg = float(inclination_deg)

# If periapsis is greater than apoapsis,
# swap the values
if periapsis_km > apoapsis_km:

    periapsis_km, apoapsis_km = (
        apoapsis_km,
        periapsis_km
    )

# Semi-major axis
a = (
    apoapsis_km
    + periapsis_km
    \+ 2 * R_EARTH
) / 2

# Eccentricity
e = (
    apoapsis_km
    - periapsis_km
) / (
    apoapsis_km
    + periapsis_km
    \+ 2 * R_EARTH
)

# Inclination in radians
inc_rad = np.radians(
    inclination_deg
)

# Complete 360 degree orbit
theta = np.linspace(
    0,
    2 * np.pi,
    steps,
    endpoint=False
)

# Orbital radius
r = (
    a * (1 - e**2)
) / (
    1 + e * np.cos(theta)
)

# Coordinates before inclination rotation
x = (
    r * np.cos(theta)
)

y_orbital = (
    r * np.sin(theta)
)

# Apply inclination
y = (
    y_orbital
    * np.cos(inc_rad)
)

z = (
    y_orbital
    * np.sin(inc_rad)
)

# Orbital period
period_seconds = (
    2
    * np.pi
    * np.sqrt(
        a**3 / MU
    )
)

period_minutes = (
    period_seconds / 60
)

# Altitude range
min_alt = min(
    periapsis_km,
    apoapsis_km
)

max_alt = max(
    periapsis_km,
    apoapsis_km
)

# Orbit classification
orbit_type = classify_orbit(
    (
        min_alt
        + max_alt
    ) / 2,
    inclination_deg
)

return (
    x,
    y,
    z,
    period_minutes,
    (
        min_alt,
        max_alt
    ),
    orbit_type
)
```

# ============================================================

# CREATE EARTH

# ============================================================

def create_earth_mesh():

```
u, v = np.mgrid[
    0:2 * np.pi:60j,
    0:np.pi:30j
]

xs = (
    R_EARTH
    * np.cos(u)
    * np.sin(v)
)

ys = (
    R_EARTH
    * np.sin(u)
    * np.sin(v)
)

zs = (
    R_EARTH
    * np.cos(v)
)

return (
    xs,
    ys,
    zs
)
```

# ============================================================

# CREATE 2D PLOT

# ============================================================

def plot_2d(x, y):

```
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

fig, ax = plt.subplots(
    figsize=(6, 6)
)

# Earth
earth = Circle(
    (0, 0),
    R_EARTH,
    color="blue",
    alpha=0.3
)

ax.add_patch(
    earth
)

# Orbit path
ax.plot(
    x,
    y,
    "r-",
    label="Orbit Path"
)

# Equal scaling
ax.set_aspect(
    "equal"
)

ax.set_xlabel(
    "X (km)"
)

ax.set_ylabel(
    "Y (km)"
)

ax.set_title(
    "2D Orbit Visualization"
)

ax.legend()

ax.grid(
    True
)

st.pyplot(
    fig,
    clear_figure=True
)
```

# ============================================================

# CREATE 3D ORBIT WITH ANIMATION

# ============================================================

def create_3d_orbit_figure(
x,
y,
z,
sat_idx=0
):

```
# --------------------------------------------------------
# FIXED AXIS RANGE
# --------------------------------------------------------

max_range = (
    np.max(
        np.abs(
            np.concatenate(
                [x, y, z]
            )
        )
    )
    * 1.2
)

# --------------------------------------------------------
# EARTH
# --------------------------------------------------------

xs, ys, zs = (
    create_earth_mesh()
)

# --------------------------------------------------------
# FIXED CAMERA
# --------------------------------------------------------

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
    ),

    up=dict(
        x=0,
        y=0,
        z=1
    )
)

# --------------------------------------------------------
# CREATE FIGURE
# --------------------------------------------------------

fig = go.Figure(

    data=[

        # EARTH
        go.Surface(

            x=xs,

            y=ys,

            z=zs,

            colorscale="Blues",

            opacity=0.6,

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

                color="red",

                width=3
            ),

            name="Orbit"
        ),

        # SATELLITE
        go.Scatter3d(

            x=[x[sat_idx]],

            y=[y[sat_idx]],

            z=[z[sat_idx]],

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

# --------------------------------------------------------
# CREATE ANIMATION FRAMES
# --------------------------------------------------------

frames = []

for i in range(
    len(x)
):

    frame = go.Frame(

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

        # Update only satellite
        # Trace 0 = Earth
        # Trace 1 = Orbit
        # Trace 2 = Satellite
        traces=[2]
    )

    frames.append(
        frame
    )

fig.frames = frames

# --------------------------------------------------------
# FIGURE LAYOUT
# --------------------------------------------------------

fig.update_layout(

    title=(
        "3D Orbit Visualization"
    ),

    margin=dict(

        l=0,

        r=0,

        t=40,

        b=0
    ),

    scene=dict(

        # Fixed camera
        camera=camera,

        # Fixed X axis
        xaxis=dict(

            range=[

                -max_range,

                max_range
            ],

            autorange=False,

            title="X (km)"
        ),

        # Fixed Y axis
        yaxis=dict(

            range=[

                -max_range,

                max_range
            ],

            autorange=False,

            title="Y (km)"
        ),

        # Fixed Z axis
        zaxis=dict(

            range=[

                -max_range,

                max_range
            ],

            autorange=False,

            title="Z (km)"
        ),

        # Keep equal proportions
        aspectmode="cube"
    ),

    # ----------------------------------------------------
    # ANIMATION BUTTONS
    # ----------------------------------------------------

    updatemenus=[

        dict(

            type="buttons",

            showactive=False,

            direction="left",

            x=0.1,

            y=1.12,

            buttons=[

                # START BUTTON
                dict(

                    label=(
                        "▶ Start Simulation"
                    ),

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

                # STOP BUTTON
                dict(

                    label=(
                        "⏹ Stop Simulation"
                    ),

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
    ]
)

return fig
```

# ============================================================

# STREAMLIT PAGE CONFIG

# ============================================================

st.set_page_config(

```
page_title=(
    "Satellite Orbit Visualizer"
),

layout="wide"
```

)

# ============================================================

# TITLE

# ============================================================

st.title(
"🛰️ Satellite Orbit Visualizer (2D & 3D)"
)

# ============================================================

# SIDEBAR

# ============================================================

with st.sidebar:

```
st.header(
    "Input Parameters"
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
```

# ============================================================

# SESSION STATE

# ============================================================

if "orbit_data" not in st.session_state:

```
st.session_state.orbit_data = None
```

# ============================================================

# GENERATE ORBIT BUTTON

# ============================================================

if st.button(
"Generate Orbit"
):

```
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

    inclination,

    steps=360
)

st.session_state.orbit_data = {

    "x": x,

    "y": y,

    "z": z,

    "period_min": period_min,

    "alt_range": alt_range,

    "orbit_type": orbit_type
}
```

# ============================================================

# DISPLAY GENERATED ORBIT

# ============================================================

if st.session_state.orbit_data:

```
od = (
    st.session_state.orbit_data
)

# --------------------------------------------------------
# ORBIT SUMMARY
# --------------------------------------------------------

st.subheader(
    "🛰️ Orbit Summary"
)

st.markdown(

    f"**Orbit Type:** "
    f"{od['orbit_type']}"
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

# --------------------------------------------------------
# 2D VISUALIZATION
# --------------------------------------------------------

if show_2d:

    plot_2d(

        od["x"],

        od["y"]
    )

# --------------------------------------------------------
# SATELLITE POSITION SLIDER
# --------------------------------------------------------

pos_deg = st.slider(

    "Satellite Position (degrees)",

    0,

    360,

    0,

    step=1
)

# Convert 0-360 degrees
# to array index

if pos_deg == 360:

    pos_idx = 0

else:

    pos_idx = int(

        (
            pos_deg
            / 360
        )
        * len(
            od["x"]
        )
    )

    pos_idx = min(

        pos_idx,

        len(
            od["x"]
        ) - 1
    )

# --------------------------------------------------------
# 3D VISUALIZATION
# --------------------------------------------------------

if show_3d:

    fig3d = (

        create_3d_orbit_figure(

            od["x"],

            od["y"],

            od["z"],

            pos_idx
        )
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
```

else:

```
st.info(

    "Please generate an orbit first "
    "using the inputs above."
)
```

# ============================================================

# ORBIT REFERENCE TABLE

# ============================================================

st.subheader(
"📋 Orbit Type Reference Table"
)

orbit_table = pd.DataFrame(

```
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
}
```

)

st.table(
orbit_table
)

import streamlit as st
from Cubesat_And_Cold_Gas_Propulsion import your_simulation_function  # replace with your function name

st.title("CubeSat Cold Gas Propulsion Simulation")

# Example: add input for thrust value
thrust = st.number_input("Enter thrust (mN):", min_value=0.0, max_value=100.0, value=10.0)

# Run your simulation function with input
result = your_simulation_function(thrust)  # replace with your actual function call

# Show result
st.write(f"Simulation output: {result}")

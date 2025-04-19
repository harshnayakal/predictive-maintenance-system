import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/random_forest_model.pkl")

# UI
st.set_page_config(page_title="Predictive Maintenance", layout="centered")
st.title("Predictive Maintenance System")
st.markdown("### Input Sensor Readings")

# Input sliders
temp = st.slider("Temperature (°C)", min_value=30, max_value=100, value=65)
vib = st.slider("Vibration Level (mm/s)", min_value=0, max_value=10, value=5)
press = st.slider("Pressure (bar)", min_value=1, max_value=100, value=50)
rpm = st.slider("RPM", min_value=500, max_value=3000, value=1500)

# Prediction
input_df = pd.DataFrame([[temp, vib, press, rpm]],
                        columns=["Temperature", "Vibration", "Pressure", "RPM"])

if st.button("Predict Machine Status"):
    pred = model.predict(input_df)[0]
    result = "Machine is HEALTHY" if pred == 0 else "**FAILURE DETECTED – Maintenance Required**"
    st.subheader(f"Result: {result}")

import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("cyber_model.pkl")
scaler = joblib.load("scaler.pkl")

# List your selected features (must match model training order!)
selected_features = [
    'Source Port', 'Destination Port', 'Protocol', 'Packet Length', 'Packet Type',
    'Traffic Type', 'Malware Indicators', 'Anomaly Scores', 'Severity Level'
    # Add/remove features as per your model
]

st.title("Cyber Security Threat Prediction")

st.markdown("Enter network traffic features to predict the type of cyber attack.")

user_input = []
for feature in selected_features:
    value = st.number_input(f"Enter value for {feature}:", step=1.0)
    user_input.append(value)

if st.button("Predict Attack Type"):
    input_np = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_np)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Attack Type: {prediction[0]}")

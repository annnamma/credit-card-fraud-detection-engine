import streamlit as st
import pandas as pd
import joblib

# Load trained model
model, scaler, features = joblib.load("fraud_engine.pkl")

st.title("💳 Credit Card Fraud Detection Engine")

st.write(
    "Enter transaction details below to assess fraud risk."
)

# User inputs
amount = st.number_input(
    "Transaction Amount ($)",
    min_value=0.0,
    value=50.0
)

hours = st.number_input(
    "Hours Since Last Transaction",
    min_value=0.01,
    value=12.0
)

distance = st.number_input(
    "Distance From Home (Miles)",
    min_value=0.0,
    value=5.0
)

device = st.selectbox(
    "Device Type",
    ["Mobile", "Desktop", "Tablet", "Unknown"]
)

# Feature Engineering
velocity = amount / (hours + 0.1)
is_unknown = 1 if device == "Unknown" else 0

# Create input dataframe
input_df = pd.DataFrame(
    [[
        amount,
        hours,
        distance,
        velocity,
        is_unknown
    ]],
    columns=features
)

# Scale input
input_scaled = scaler.transform(input_df)

# Prediction button
if st.button("Predict Fraud Risk"):

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(
            f"🚨 Fraud Detected! Risk Score: {probability:.2%}"
        )
    else:
        st.success(
            f"✅ Legitimate Transaction. Risk Score: {probability:.2%}"
        )

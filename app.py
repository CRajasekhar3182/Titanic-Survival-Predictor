# streamlit_app.py

import streamlit as st
import numpy as np
import joblib

# Custom CSS for styling the app
st.markdown("""
    <style>
    /* Background for the app */
    .stApp {
        background-color: #f0f4f7;
    }

    /* Title styling */
    .title h1 {
        font-size: 48px;
        font-weight: bold;
        color: #009688;
        text-align: center;
        margin-bottom: 40px;
    }

    /* Subtle box shadow for input widgets */
    .stNumberInput, .stSelectbox {
        box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
        border-radius: 8px;
        margin-bottom: 20px;
    }

    /* Button styling */
    .stButton button {
        background-color: #009688;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 8px;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
        transition: background-color 0.3s ease;
    }

    /* Button hover effect */
    .stButton button:hover {
        background-color: #00796b;
    }

    /* Text area styling for results */
    .stTextArea {
        font-size: 18px;
        color: #424242;
        border: 1px solid #bdbdbd;
        padding: 20px;
        border-radius: 10px;
        background-color: #e0f7fa;
    }
    </style>
""", unsafe_allow_html=True)

# Load the Titanic model
model = joblib.load("titanic.pkl")

# Title with custom styling
st.markdown("<div class='title'><h1>Titanic Survival Predictor</h1></div>", unsafe_allow_html=True)
st.image("titanic.webp", caption="Titanic Survival Prediction", use_column_width=True)

# User inputs with section header
st.header("Please fill in the following details to check survival status:")

# User inputs with custom placeholders and labels
gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
age = st.number_input("Enter your age", min_value=1, max_value=100, step=1, key="age")
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3], key="pclass")
ticket_fare = st.number_input("Ticket Fare", min_value=0.0, max_value=512.0, step=0.1, key="ticket_fare")

# Convert gender to numeric for prediction
gender_values = 1 if gender == "Male" else 2

# Prepare input features
input_features = [[pclass, gender_values, age, ticket_fare]]

# Prediction with a button
if st.button("Predict Survival"):
    prediction = model.predict(input_features)

    # Display prediction result
    if prediction == 1:
        st.text_area("Result", "This passenger would have survived.")
    else:
        st.text_area("Result", "This passenger would not have survived.")

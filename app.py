import streamlit as st
import pickle
import numpy as np

# Load the trained stacked model
with open("stacked_model.pkl", "rb") as file:
    stacked_model = pickle.load(file)

# Streamlit UI
st.title("Heart Disease Prediction Using Stacked Model")
st.write("Enter patient details to predict heart disease:")

# Input fields
age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.2, value=1.0, step=0.1)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

# Convert categorical inputs to numerical values
sex = 1 if sex == "Male" else 0

# Prepare input data for prediction
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

# Predict on user input
target_map = {0: "No Heart Disease", 1: "Heart Disease Present"}
if st.button("Predict"):
    prediction = stacked_model.predict(input_data)
    st.write("Prediction:", target_map[int(prediction[0])])




import numpy as np
import joblib
import streamlit as st

# Load the saved models
models = {}
for model_name in ["LSTM", "CNN", "RNN", "ANN"]:
    filename = f"{model_name}_model.pkl"
    models[model_name] = joblib.load(filename)

meta_learner = joblib.load("stacked_model.pkl")

# Streamlit UI
st.title("Heart Disease Prediction using Stacked Ensemble Model")
st.write("Enter the input features below:")

# Input fields for user input
input_features = []
feature_labels = ["Age", "Sex", "CP", "Trestbps", "Chol", "FBS", "RestECG", "Thalach", "Exang", "Oldpeak", "Slope", "Ca", "Thal"]

default_values = [58, 0, 0, 100, 248, 0, 0, 122, 0, 1, 1, 0, 2]

for i, label in enumerate(feature_labels):
    value = st.number_input(label, value=default_values[i])
    input_features.append(value)

# Convert input to numpy array
new_input = np.array([input_features])

# Reshape input for LSTM, CNN, RNN
new_input_reshaped = new_input.reshape(new_input.shape[0], new_input.shape[1], 1)

# Generate predictions from individual models
if st.button("Predict"):
    individual_predictions = []
    for name, model in models.items():
        if name == "ANN":
            pred = model.predict(new_input).flatten()
        else:
            pred = model.predict(new_input_reshaped).flatten()
        individual_predictions.append(pred)
    
    # Stack the predictions
    stacked_input = np.column_stack(individual_predictions)
    
    # Make a prediction using the stacked model
    final_prediction = meta_learner.predict(stacked_input)
    
    # Display result
    st.write("### Final Prediction (Stacked Model):")
    if final_prediction[0] == 1:
        st.success("The model predicts the presence of heart disease.")
    else:
        st.success("The model predicts no heart disease.")

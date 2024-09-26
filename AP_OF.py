import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import xgboost

# Load the model
model = joblib.load(r"C:\Users\zhang\Desktop\app_AP_OF\XGBoost_clinical.pkl")

# Define feature names
feature_names = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j"
]

# Streamlit user interface
st.title("Prediction Model")

# Continuous variable inputs
a = st.number_input("a:", value=0.0)
b = st.number_input("b:", value=0.0)
c = st.number_input("c:", value=0.0)
d = st.number_input("d:", value=0.0)
e = st.number_input("e:", value=0.0)
f = st.number_input("f:", value=0.0)
g = st.number_input("g:", value=0.0)
h = st.number_input("h:", value=0.0)
i = st.number_input("i:", value=0.0)
j = st.number_input("j:", value=0.0)

# Process inputs and make predictions
feature_values = [a, b, c, d, e, f, g, h, i, j]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"Based on the model, there is a high probability ({probability:.1f}%) of the event occurring. "
            "Please consider taking appropriate actions."
        )
    else:
        advice = (
            f"Based on the model, there is a low probability ({probability:.1f}%) of the event occurring. "
            "However, it's always good to stay cautious and prepared."
        )

    st.write(advice)

    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
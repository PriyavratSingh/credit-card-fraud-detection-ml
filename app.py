import streamlit as st
import pickle
import numpy as np

# Load model & scaler
model = pickle.load(open("D:/AI Codes/Fraud-Detection/models/random_forest.pkl", "rb"))
scaler = pickle.load(open("D:/AI Codes/Fraud-Detection/models/scaler.pkl", "rb"))

st.title("AI-Based Credit Card Fraud Detection System")
st.write("Enter 30 transaction feature values separated by commas.")

input_data = st.text_area("Transaction Features")

if st.button("Predict"):
    try:
        features = [float(x) for x in input_data.split(",")]
        
        if len(features) != 30:
            st.error("Please enter exactly 30 feature values.")
        else:
            features = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features)

            prediction = model.predict(features_scaled)

            if prediction[0] == 1:
                st.error("⚠️ Fraudulent Transaction Detected!")
            else:
                st.success("✅ Legitimate Transaction")

    except:
        st.error("Invalid input format.")
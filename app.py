
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="European Bank Churn Dashboard", layout="wide")
st.title("European Bank Customer Churn Prediction")
st.write("Predict customer churn using a trained Random Forest model.")

bundle = joblib.load("churn_model.pkl")
model = bundle["model"]
features = bundle["features"]

age = st.number_input("Age", min_value=18, max_value=100, value=40)
credit = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
balance = st.number_input("Account Balance", min_value=0.0, value=50000.0)
salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
tenure = st.slider("Tenure (Years)", 0, 10, 5)
products = st.selectbox("Number of Products", [1, 2, 3, 4])
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Female", "Male"])
card = st.selectbox("Has Credit Card", [0, 1])
active = st.selectbox("Is Active Member", [0, 1])

if st.button("Predict Churn"):
    data = pd.DataFrame(0, index=[0], columns=features)
    values = {
        "CreditScore": credit,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": card,
        "IsActiveMember": active,
        "EstimatedSalary": salary,
        "Geography_" + geography: 1,
        "Gender_" + gender: 1
    }
    for key, value in values.items():
        if key in data.columns:
            data.loc[0, key] = value

    probability = model.predict_proba(data)[0][1]
    st.metric("Churn Probability", f"{probability:.1%}")
    st.write("Predicted Outcome:", "Likely to Churn" if model.predict(data)[0] == 1 else "Likely to Stay")

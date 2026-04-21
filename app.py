# app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# Load dataset (only needed if you want to retrain locally)
df = pd.read_csv(r"D:\archive\salary_data_cleaned.csv")

# Train once locally and save model
# Uncomment these lines if you want to retrain
# X = df[["age", "experience"]]   # adjust columns if needed
# y = df["avg_salary"]
# model = LinearRegression()
# model.fit(X, y)
# joblib.dump(model, "salary_model.pkl")

# Load pre-trained model
model = joblib.load("salary_model.pkl")

# Streamlit UI
st.title("💼 Salary Prediction App")

st.write("Enter your details to predict the average salary:")

# User inputs
age = st.number_input("Age", min_value=18, max_value=65, value=25)
experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=1)

# Prediction
if st.button("Predict Salary"):
    pred_salary = model.predict([[age, experience]])
    st.success(f"Predicted Average Salary: ₹{pred_salary[0]:,.2f}")

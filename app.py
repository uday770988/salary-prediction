# app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
df = pd.read_csv(r"D:\archive\salary_data_cleaned.csv")

# Features and target
X = df[["age"]]   # using age as the feature (dataset doesn't have experience column)
y = df["avg_salary"]

# Train model (or load if already saved)
model = LinearRegression()
model.fit(X, y)

# Optionally save model for reuse
joblib.dump(model, "salary_model.pkl")

# Streamlit UI
st.title("💼 Salary Prediction App")

st.write("Enter your details to predict the average salary:")

# User inputs
age = st.number_input("Age", min_value=18, max_value=65, value=25)

# Prediction
if st.button("Predict Salary"):
    pred_salary = model.predict([[age]])
    st.success(f"Predicted Average Salary: ₹{pred_salary[0]:,.2f}")

# train_salary_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv(r"D:\archive\salary_data_cleaned.csv")

# Inspect first few rows
print(df.head())

# Print columns
print("Columns:", df.columns.tolist())

# Features (X) and target (y)
# Adjust column names if needed
X = df[["age"]]   # independent variable (assuming age relates to experience)
y = df["avg_salary"]              # dependent variable

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
print("Model Coefficient:", model.coef_)
print("Model Intercept:", model.intercept_)
print("Test Score (R^2):", model.score(X_test, y_test))

# Example prediction
years = [[12]]
pred_salary = model.predict(years)
print(f"Predicted salary for 12 years experience: {pred_salary[0]:.2f}")

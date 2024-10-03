import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# CRISP-DM Step 1: Business Understanding
st.title("Linear Regression Analysis Using CRISP-DM")
st.write("""
### This app demonstrates Linear Regression using a synthetic dataset.
You can adjust the slope, noise scale, and number of data points to see how the model's performance changes.
""")

# Sidebar for user inputs (Step 6: Deployment)
st.sidebar.header("Adjust Parameters:")
a = st.sidebar.slider("Slope (a)", -100.0, 100.0, 1.0)  # Slope
c = st.sidebar.slider("Noise Scale (c)", 0.0, 100.0, 10.0)  # Noise scale
n = st.sidebar.slider("Number of Points (n)", 10, 500, 100)  # Number of data points

# Step 2: Data Understanding
st.write("#### Step 2: Data Understanding")
st.write(f"Slope (a): {a}, Noise Scale (c): {c}, Number of Points (n): {n}")

# Step 3: Data Preparation
def generate_data(a, c, n):
    X = np.linspace(-10, 10, n)  # X values from -10 to 10
    noise = np.random.normal(0, c, n)  # Normally distributed noise
    y = a * X + 50 + noise  # y = a * X + 50 + noise
    return pd.DataFrame({"X": X, "y": y})

data = generate_data(a, c, n)

st.write("Generated Dataset:")
st.write(data.head())  # Displaying first few rows of the dataset

# Step 4: Modeling
X = data[['X']]
y = data['y']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write("#### Step 5: Model Evaluation")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R-squared (R²): {r2:.2f}")

# Step 6: Visualization
# Scatter plot of actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label="Actual", color="blue")
plt.plot(X_test, y_pred, label="Predicted", color="red", linewidth=2)
plt.title("Actual vs. Predicted Values")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
st.pyplot(plt)

# Step 7: Rerun functionality
st.write("You can adjust the parameters in the sidebar to see the effects on the model and re-run the analysis.")
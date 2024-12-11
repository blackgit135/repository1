import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load or retrain model
def train_model():
    # Example dataset
    data = {
        "feature1": [10, 20, 15, 30, 25],
        "feature2": ["A", "B", "A", "B", "A"],
        "feature3": [100, 200, 150, 300, 250],
        "price": [500, 1000, 750, 1500, 1250]
    }
    df = pd.DataFrame(data)

    # Split features and target
    X = df.drop(columns=["price"])
    y = df["price"]

    # Preprocessing
    numeric_features = ["feature1", "feature3"]
    categorical_features = ["feature2"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Define the model pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42))
    ])

    # Train the model
    model.fit(X, y)
    return model

# Train the model
model = train_model()

# Streamlit app
st.title("Product Price Prediction App")

# Inputs
st.header("Enter Product Features:")
feature1 = st.number_input("Feature 1 (e.g., numeric feature):", min_value=0, step=1, value=10)
feature2 = st.selectbox("Feature 2 (e.g., category):", ["A", "B"])
feature3 = st.number_input("Feature 3 (e.g., numeric feature):", min_value=0, step=1, value=100)

# Predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        "feature1": [feature1],
        "feature2": [feature2],
        "feature3": [feature3]
    })

    predicted_price = model.predict(input_data)
    st.success(f"Predicted Price: ${predicted_price[0]:.2f}")

# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.datasets import load_iris

# Load model, scaler, and dataset
model = joblib.load("iris_model.pkl")
scaler = joblib.load("scaler.pkl")
iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')
target_names = iris.target_names

st.set_page_config(page_title="🌸 Iris Predictor", layout="centered")

# Title and instructions
st.title("🌼 Iris Flower Prediction App")
st.markdown("Provide flower measurements to predict the **species** and understand model behavior.")

# Input fields
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
user_input_scaled = scaler.transform(user_input)

# Prediction
if st.button("🔍 Predict"):
    prediction = model.predict(user_input_scaled)[0]
    st.success(f"🌸 **Predicted Species:** {target_names[prediction].capitalize()}")

    # Compare with dataset
    st.subheader("📊 Your Input vs Dataset Averages")
    df_input = pd.DataFrame(user_input, columns=X.columns)
    df_avg = pd.DataFrame(X.mean()).T
    df_combined = pd.concat([df_input, df_avg])
    df_combined.index = ['Your Input', 'Dataset Avg']

    st.dataframe(df_combined)

    # Bar chart comparison
    st.subheader("📈 Feature Comparison")
    fig, ax = plt.subplots(figsize=(10, 5))
    df_combined.T.plot(kind='bar', ax=ax)
    plt.title("Your Input vs Average")
    plt.ylabel("Feature Value (cm)")
    st.pyplot(fig)

    # Feature importance
    st.subheader("📌 Feature Importance (Model's Decision Basis)")
    importances = model.feature_importances_
    df_imp = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    fig2, ax2 = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=df_imp, palette='spring', ax=ax2)
    ax2.set_title("Feature Importance from Random Forest")
    st.pyplot(fig2)

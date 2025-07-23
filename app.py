import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and encoders
model = joblib.load("salary_predictor_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load label encoders
categorical_cols = ['workclass', 'occupation', 'gender']
encoders = {col: joblib.load(f"{col}_encoder.pkl") for col in categorical_cols}

st.title("Employee Salary Prediction")

# Minimal Input Fields
age = st.number_input("Age", 18, 80, 30)
workclass = st.selectbox("Workclass", encoders['workclass'].classes_)
occupation = st.selectbox("Occupation", encoders['occupation'].classes_)
gender = st.selectbox("Gender", encoders['gender'].classes_)
hours = st.slider("Hours per Week", 1, 100, 40)

if st.button("Predict"):
    input_dict = {
        'age': age,
        'workclass': workclass,
        'occupation': occupation,
        'gender': gender,
        'hours-per-week': hours
    }

    df = pd.DataFrame([input_dict])

    # Encode categorical
    for col in ['workclass', 'occupation', 'gender']:
        df[col] = encoders[col].transform([df[col].values[0]])

    # Add dummy/default values for missing features
    df['fnlwgt'] = 150000
    df['educational-num'] = 10
    df['marital-status'] = 1
    df['relationship'] = 1
    df['race'] = 1
    df['capital-gain'] = 0
    df['capital-loss'] = 0
    df['native-country'] = 1

    # Reorder columns to match model
    final_cols = ['age', 'workclass', 'fnlwgt', 'educational-num',
                  'marital-status', 'occupation', 'relationship',
                  'race', 'gender', 'capital-gain', 'capital-loss',
                  'hours-per-week', 'native-country']
    df = df[final_cols]

    df_scaled = scaler.transform(df)
    result = model.predict(df_scaled)[0]

    st.success(f"Predicted income class: {result}")

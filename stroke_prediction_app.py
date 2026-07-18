import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Cerebral Stroke Prediction App")
st.write("Enter the patient details below to check the stroke risk.")

with open('stroke_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

age = st.slider("Age", 0.0, 100.0, 50.0)
avg_glucose_level = st.number_input("Average Glucose Level", 50.0, 300.0, 100.0)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

hypertension = st.selectbox("Hypertension?", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease?", ["No", "Yes"])
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
ever_married = st.selectbox("Ever Married?", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence = st.selectbox("Residence Type", ["Urban", "Rural"])
smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "unknown"])

user_features = {
    'age': age,
    'hypertension': 1.0 if hypertension == "Yes" else 0.0,
    'heart_disease': 1.0 if heart_disease == "Yes" else 0.0,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'gender_Male': 1.0 if gender == "Male" else 0.0,
    'gender_Other': 1.0 if gender == "Other" else 0.0,
    'ever_married_Yes': 1.0 if ever_married == "Yes" else 0.0,
    'work_type_Never_worked': 1.0 if work_type == "Never_worked" else 0.0,
    'work_type_Private': 1.0 if work_type == "Private" else 0.0,
    'work_type_Self-employed': 1.0 if work_type == "Self-employed" else 0.0,
    'work_type_children': 1.0 if work_type == "children" else 0.0,
    'Residence_type_Urban': 1.0 if residence == "Urban" else 0.0,
    'smoking_status_never smoked': 1.0 if smoking == "never smoked" else 0.0,
    'smoking_status_smokes': 1.0 if smoking == "smokes" else 0.0,
    'smoking_status_unknown': 1.0 if smoking == "unknown" else 0.0
}

df_input = pd.DataFrame([user_features])

expected_order = [
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
    'gender_Male', 'gender_Other', 'ever_married_Yes', 'work_type_Never_worked',
    'work_type_Private', 'work_type_Self-employed', 'work_type_children',
    'Residence_type_Urban', 'smoking_status_never smoked', 'smoking_status_smokes',
    'smoking_status_unknown'
]
df_input = df_input[expected_order]

if st.button("Predict"):
    # Create a copy to manipulate
    df_to_scale = df_input.copy()
    
    num_cols = ['age', 'avg_glucose_level', 'bmi']
    df_to_scale[num_cols] = scaler.transform(df_to_scale[num_cols])
    
    prediction = model.predict(df_to_scale)[0]
    prob = model.predict_proba(df_to_scale)[0][1]
    
    st.write("---")
    if prediction == 1:
        st.error(f"High Risk of Stroke! Probability: {prob:.2%}")
    else:
        st.success(f"Low Risk of Stroke. Probability: {prob:.2%}")
import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load("heart_disease_model.pkl")
features = joblib.load("model_features.pkl")

# App UI
st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀")
st.title("🫀 Heart Disease Risk Prediction System")
st.markdown("Provide patient details below to assess heart disease risk using a trained machine learning model.")

st.sidebar.markdown("## 📋 Instructions")
st.sidebar.markdown("""
1. Fill out the patient's information.
2. Click **Predict** to receive a risk evaluation.
3. Use the result and confidence score to support early clinical decision-making.
""")

# Input Form
name = st.text_input("👤 Patient Name")

with st.form("patient_form"):
    st.subheader("📌 General Information")
    col1, col2 = st.columns(2)
    with col1:
        bmi = st.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0)
        age = st.selectbox("Age Category", ['18-24', '25-29', '30-34', '35-39', '40-44'])
        sex = st.selectbox("Sex", ['Male', 'Female'])
        race = st.selectbox("Race", ['White', 'Black', 'Asian', 'Hispanic'])
    with col2:
        physical_health = st.slider("Physical Health (days unwell)", 0, 30, 0)
        mental_health = st.slider("Mental Health (days unwell)", 0, 30, 0)
        sleep_time = st.slider("Average Sleep (hrs/night)", 1, 24, 8)

    st.subheader("🩺 Medical History & Lifestyle")
    col3, col4 = st.columns(2)
    with col3:
        smoking = st.selectbox("Smoking", ['Yes', 'No'])
        alcohol = st.selectbox("Heavy Alcohol Use", ['Yes', 'No'])
        stroke = st.selectbox("History of Stroke", ['Yes', 'No'])
        diabetic = st.selectbox("Diabetic", ['Yes', 'No'])
    with col4:
        diff_walking = st.selectbox("Difficulty Walking", ['Yes', 'No'])
        physical_activity = st.selectbox("Engages in Physical Activity", ['Yes', 'No'])
        gen_health = st.selectbox("General Health Rating", ['Poor', 'Fair', 'Good', 'Very good', 'Excellent'])

    submitted = st.form_submit_button("Predict")

# Prediction
if submitted:
    input_data = pd.DataFrame([{
        'BMI': bmi,
        'AgeCategory': age,
        'Sex': sex,
        'Race': race,
        'PhysicalHealth': physical_health,
        'MentalHealth': mental_health,
        'SleepTime': sleep_time,
        'Smoking': smoking,
        'AlcoholDrinking': alcohol,
        'Stroke': stroke,
        'Diabetic': diabetic,
        'DiffWalking': diff_walking,
        'PhysicalActivity': physical_activity,
        'GenHealth': gen_health
    }])

    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=features, fill_value=0)

    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    # Display result
    st.subheader("🔎 Prediction Result")
    st.markdown(f"**Patient Name**: `{name if name else 'Anonymous'}`")
    st.markdown(f"**Confidence Score**: `{probability:.2f}`")

    if prediction == 1:
        st.error("⚠️ This patient is **AT RISK** of heart disease.")
        st.markdown("""
        ### 🩺 Recommendations:
        - Recommend immediate medical consultation and diagnostic testing.
        - Encourage lifestyle changes (smoking cessation, better sleep, exercise).
        - Monitor blood pressure, cholesterol, and glucose levels regularly.
        - Refer to cardiology if risk factors are confirmed.
        """)
    else:
        st.success("✅ This patient is **NOT at risk** of heart disease at the moment.")
        st.markdown("""
        ### 🛡️ Recommendations:
        - Continue promoting healthy lifestyle habits.
        - Encourage annual check-ups and wellness screenings.
        - Educate on symptoms and prevention of cardiovascular issues.
        """)

    # Optional Debug Panel
    with st.expander("🔧 View Prediction Debug Info"):
        st.json({
            "Prediction": int(prediction),
            "Confidence": round(probability, 2),
            "Patient Inputs": input_data.to_dict(orient='records')[0]
        })

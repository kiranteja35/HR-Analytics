import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("Model.h2")

st.title("HR Employee Promotion Predictor")
st.markdown("Enter employee details to predict whether the employee will be promoted or not.")

# Inputs
age = st.slider("Age", 20, 60, 30)
no_of_trainings = st.slider("Number of Trainings", 1, 10, 2)
length_of_service = st.slider("Length of Service (Years)", 1, 35, 5)
avg_training_score = st.slider("Average Training Score", 40, 100, 60)
previous_year_rating = st.slider("Previous Year Rating", 1.0, 5.0, 3.0, step=0.5)

department = st.selectbox("Department", ['Sales', 'Operations', 'HR', 'IT', 'Procurement', 'other'])
region = st.selectbox("Region", ['region_1', 'region_2', 'region_7', 'other'])
education = st.selectbox("Education", ['Bachelor’s', 'Master’s', 'Other'])
gender = st.selectbox("Gender", ['Male', 'Female'])
recruitment_channel = st.selectbox("Recruitment Channel", ['sourcing', 'referred', 'other'])
kpi_met = st.radio("Has KPI > 80% been met?", ["Yes", "No"])
award_won = st.radio("Has the employee won an award?", ["Yes", "No"])

# Convert to binary
kpi_met = 1 if kpi_met == "Yes" else 0
award_won = 1 if award_won == "Yes" else 0

# Create input DataFrame
input_df = pd.DataFrame([{
    'department': department,
    'region': region,
    'education': education,
    'gender': gender,
    'recruitment_channel': recruitment_channel,
    'no_of_trainings': no_of_trainings,
    'age': age,
    'previous_year_rating': previous_year_rating,
    'length_of_service': length_of_service,
    'KPIs_met >80%': kpi_met,
    'awards_won?': award_won,
    'avg_training_score': avg_training_score
}])

# Predict
if st.button("Predict Promotion"):
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        if prediction == 1:
            st.success(f"✅ Likely to be promoted. Confidence: {round(probability*100, 2)}%")
        else:
            st.error(f"❌ Not likely to be promoted. Confidence: {round((1-probability)*100, 2)}%")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

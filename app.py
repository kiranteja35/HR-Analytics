import streamlit as st
import pandas as pd
import joblib


# Load the trained model
model = joblib.load("Model.h1")  
st.title("HR Employee Promotion Predictor")
st.markdown("Enter employee details to predict if they are likely to be promoted.")

# === User Inputs ===
age = st.slider("Age", 20, 60, value=30)
no_of_trainings = st.slider("Number of Trainings", 1, 10, value=2)
length_of_service = st.slider("Length of Service (Years)", 1, 35, value=5)
avg_training_score = st.slider("Average Training Score", 40, 100, value=60)
previous_year_rating = st.slider("Previous Year Rating", 1.0, 5.0, step=0.5, value=3.0)

department = st.selectbox("Department", ['Sales', 'Operations', 'HR', 'IT', 'Procurement', 'other'])
region = st.selectbox("Region", ['region_1', 'region_2', 'region_7', 'other'])
education = st.selectbox("Education", ['Bachelor’s', 'Master’s', 'Other'])
gender = st.selectbox("Gender", ['Male', 'Female'])
recruitment_channel = st.selectbox("Recruitment Channel", ['sourcing', 'referred', 'other'])

# Additional features
kpi_met = st.radio("Has KPI met > 80%?", options=["Yes", "No"])
award_won = st.radio("Has the employee won any awards?", options=["Yes", "No"])

# Convert to binary (match training encoding)
kpi_met = 1 if kpi_met == "Yes" else 0
award_won = 1 if award_won == "Yes" else 0

# === Manual Encoding ===
def encode_column(value, column_name):
    mappings = {
        'department': {'HR': 0, 'IT': 1, 'Operations': 2, 'Procurement': 3, 'Sales': 4, 'other': 5},
        'region': {'region_1': 0, 'region_2': 1, 'region_7': 2, 'other': 3},
        'education': {'Bachelor’s': 0, 'Master’s': 1, 'Other': 2},
        'gender': {'Male': 0, 'Female': 1},
        'recruitment_channel': {'sourcing': 0, 'referred': 1, 'other': 2}
    }
    return mappings[column_name].get(value, -1)

# Apply encodings
department_encoded = encode_column(department, 'department')
region_encoded = encode_column(region, 'region')
education_encoded = encode_column(education, 'education')
gender_encoded = encode_column(gender, 'gender')
recruitment_channel_encoded = encode_column(recruitment_channel, 'recruitment_channel')

# Create DataFrame in correct order
user_input = pd.DataFrame([{
    'department': department_encoded,
    'region': region_encoded,
    'education': education_encoded,
    'gender': gender_encoded,
    'recruitment_channel': recruitment_channel_encoded,
    'no_of_trainings': no_of_trainings,
    'age': age,
    'previous_year_rating': previous_year_rating,
    'length_of_service': length_of_service,
    'KPIs_met >80%': kpi_met,
    'awards_won?': award_won,
    'avg_training_score': avg_training_score
}])

# Final column order matching training
user_input = user_input[
    ['department', 'region', 'education', 'gender', 'recruitment_channel',
     'no_of_trainings', 'age', 'previous_year_rating', 'length_of_service',
     'KPIs_met >80%', 'awards_won?', 'avg_training_score']
]

# === Predict ===
if st.button("Predict Promotion"):
    try:
        prediction = model.predict(user_input)[0]
        probability = model.predict_proba(user_input)[0][1]  # confidence of promotion (class 1)

        if prediction == 1:
            st.success(f"✅ The employee is likely to be promoted. Confidence: {round(probability * 100, 2)}%")
        else:
            st.error(f"❌ The employee is NOT likely to be promoted. Confidence: {round((1 - probability) * 100, 2)}%")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

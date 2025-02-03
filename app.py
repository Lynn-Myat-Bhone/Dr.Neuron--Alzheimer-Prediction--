import streamlit as st
import pandas as pd
import xgboost as xgb

# Define the columns expected by the model (without the extra columns)
model_columns = ['Age', 'Education Level', 'BMI', 'Diabetes', 'Hypertension', 'Cholesterol Level', 
                 'Family History of Alzheimer’s', 'Cognitive Test Score', 'Depression Level', 'Sleep Quality', 
                 'Air Pollution Exposure', 'Genetic Risk Factor (APOE-ε4 allele)', 'Social Engagement Level', 
                 'Income Level', 'Stress Levels', 'Gender_Male', 'Physical Activity Level_1', 'Physical Activity Level_2', 
                 'Smoking Status_Former', 'Smoking Status_Never', 'Alcohol Consumption_Occasionally', 
                 'Alcohol Consumption_Regularly', 'Dietary Habits_Healthy', 'Dietary Habits_Unhealthy', 
                 'Employment Status_Retired', 'Employment Status_Unemployed']

# Function to handle user input
def user_input_features():
    # User input for each feature
    age = st.number_input("Age", min_value=0, max_value=100, value=25)
    education_level = st.number_input("Education Level", min_value=0, max_value=20)
    bmi = st.number_input("BMI", min_value=10, max_value=50, value=22)
    
    # Convert 'Yes'/'No' to binary for categorical variables
    diabetes = st.selectbox("Diabetes", ['No', 'Yes'])
    hypertension = st.selectbox("Hypertension", ['No', 'Yes'])
    cholesterol_level = st.selectbox("Cholesterol Level", ['Normal', 'High'])
    family_history = st.selectbox("Family History of Alzheimer’s", ['No', 'Yes'])
    cognitive_score = st.number_input("Cognitive Test Score", min_value=0, max_value=100, value=75)
    depression_level = st.selectbox("Depression Level", ['Low', 'Medium', 'High'])
    sleep_quality = st.selectbox("Sleep Quality", ['Good', 'Average', 'Poor'])
    air_pollution = st.selectbox("Air Pollution Exposure", ['Low', 'Medium', 'High'])
    genetic_risk = st.selectbox("Genetic Risk Factor (APOE-ε4 allele)", ['No', 'Yes'])
    social_engagement = st.selectbox("Social Engagement Level", ['Low', 'Medium', 'High'])
    income_level = st.selectbox("Income Level", ['Low', 'Medium', 'High'])
    stress_levels = st.selectbox("Stress Levels", ['Low', 'Medium', 'High'])
    marital_status = st.selectbox("Marital Status", ['Single', 'Married'])
    urban_rural = st.selectbox("Urban vs Rural Living", ['Urban', 'Rural'])

    # Gender selection (binary for Male/Female)
    gender = st.selectbox("Gender", ['Male', 'Female'])

    # Physical Activity Level (multiple levels)
    physical_activity_level = st.selectbox("Physical Activity Level", ['Inactive', 'Moderate', 'Active'])

    # Smoking Status (Former, Never)
    smoking_status = st.selectbox("Smoking Status", ['Former', 'Never'])

    # Alcohol Consumption (Occasionally, Regularly)
    alcohol_consumption = st.selectbox("Alcohol Consumption", ['Occasionally', 'Regularly'])

    # Dietary Habits (Healthy, Unhealthy)
    dietary_habits = st.selectbox("Dietary Habits", ['Healthy', 'Unhealthy'])

    # Employment Status (Retired, Unemployed)
    employment_status = st.selectbox("Employment Status", ['Retired', 'Unemployed'])

    # Map categorical inputs to numbers
    categorical_map = {
        'Yes': 1,
        'No': 0,
        'Normal':0,
        'High': 2,
        'Medium': 1,
        'Low': 0,
        'Good': 2,
        'Average': 1,
        'Poor': 0,
        'Single': 0,
        'Married': 1,
        'Urban': 1,
        'Rural': 0,
        'Male': 1,
        'Female': 0,
        'Inactive': 0,
        'Moderate': 1,
        'Active': 2,
        'Former': 0,
        'Never': 1,
        'Occasionally': 0,
        'Regularly': 1,
        'Healthy': 1,
        'Unhealthy': 0,
        'Retired': 1,
        'Unemployed': 0
    }

    # Convert user input into a dictionary
    data = {
        'Age': age,
        'Education Level': education_level,
        'BMI': bmi,
        'Diabetes': categorical_map[diabetes], 
        'Hypertension': categorical_map[hypertension],
        'Cholesterol Level': categorical_map[cholesterol_level],
        'Family History of Alzheimer’s': categorical_map[family_history],
        'Cognitive Test Score': cognitive_score,
        'Depression Level': categorical_map[depression_level],
        'Sleep Quality': categorical_map[sleep_quality],
        'Air Pollution Exposure': categorical_map[air_pollution],
        'Genetic Risk Factor (APOE-ε4 allele)': categorical_map[genetic_risk],
        'Social Engagement Level': categorical_map[social_engagement],
        'Income Level': categorical_map[income_level],
        'Stress Levels': categorical_map[stress_levels],
        'Marital Status': categorical_map[marital_status],
        'Urban vs Rural Living': categorical_map[urban_rural],
        'Gender_Male': categorical_map[gender],
        'Physical Activity Level_1': 1 if physical_activity_level == 'Moderate' else 0,
        'Physical Activity Level_2': 1 if physical_activity_level == 'Active' else 0,
        'Smoking Status_Former': 1 if smoking_status == 'Former' else 0,
        'Smoking Status_Never': 1 if smoking_status == 'Never' else 0,
        'Alcohol Consumption_Occasionally': categorical_map[alcohol_consumption],
        'Alcohol Consumption_Regularly': 1 if alcohol_consumption == 'Regularly' else 0,
        'Dietary Habits_Healthy': categorical_map[dietary_habits],
        'Dietary Habits_Unhealthy': 1 if dietary_habits == 'Unhealthy' else 0,
        'Employment Status_Retired': categorical_map[employment_status],
        'Employment Status_Unemployed': 1 if employment_status == 'Unemployed' else 0
    }

    # Create user data DataFrame
    user_data = pd.DataFrame(data, index=[0])

    # Reindex to match model columns, fill missing columns with 0
    user_data = user_data.reindex(columns=model_columns, fill_value=0)

    # # Debug: Check the user data
    # st.write(f"User Data: {user_data}")

    return user_data

model = xgb.Booster()
model.load_model('xgb_model3.json')


# Streamlit interface to display prediction
st.title("Alzheimer’s Prediction")

# Get user input features
user_data = user_input_features()

# Convert user data to DMatrix for XGBoost
input_dmatrix = xgb.DMatrix(user_data, enable_categorical=True)

st.write("User Input Data:")
st.write(user_data)


# Predict button
if st.button("Predict Alzheimer’s Risk"):
    # Make prediction
    prediction = model.predict(input_dmatrix)
    st.success(prediction)
    # Display result
    if prediction[0] > 0.6:
        st.error("🧠 This person will likely to have Alzheimer")
    else:
        st.success("✅ This person will unlikely to have Alzheimer")

# streamlit_app.py
# This is the Streamlit application for deploying your ML fitness project.
# Place this file in your project root or a new directory.
# Run with streamlit run streamlit_app.py
# Assumptions
# - Models are saved as .pkl files in the 'models' directory (or downloaded via loaders.py).
# - I've fixed the weight prediction logic by assuming you retrain the weight models with 'Weight (kg)' included in features.
# - For the weight model fix, see the updated train_weight_model_gradient_boosting.py code below this script.
# - Similarly for SVM if needed.
# - Weekly plan uses logic from plan_week.py (rules-based).
# - I've used 'recommendation_gradient_boosting.pkl', 'calories_model_gradient_boosting.pkl', 'weight_model_gradient_boosting.pkl'.
# - If models fail to load, check paths or HF downloads.
# - Install requirements streamlit, pandas, scikit-learn, joblib

import streamlit as st
import pandas as pd
import os
import random

# Import your model loader (adjust path if needed)
from models.loaders import load_all_models

# Load models once
@st.cache_resource
def get_models()
    return load_all_models()

models = get_models()

# Check specific models
calories_model = models.get('calories_model_gradient_boosting')
weight_model = models.get('weight_model_gradient_boosting')
rec_loaded = models.get('recommendation_gradient_boosting')
rec_model = rec_loaded['model'] if rec_loaded else None
rec_encoder = rec_loaded['workout_encoder'] if rec_loaded else None

# Sidebar for common user profile
st.sidebar.header(User Profile)
age = st.sidebar.number_input(Age, min_value=18, max_value=100, value=30)
gender = st.sidebar.selectbox(Gender, [Male, Female])
weight = st.sidebar.number_input(Weight (kg), min_value=40.0, max_value=150.0, value=70.0)
height = st.sidebar.number_input(Height (m), min_value=1.4, max_value=2.2, value=1.75)
fat_percentage = st.sidebar.number_input(Fat Percentage (%), min_value=5.0, max_value=50.0, value=20.0)
goal = st.sidebar.selectbox(Goal, [Maintain, Lose Weight, Gain Muscle])

bmi = weight  (height  2) if height  0 else 0
st.sidebar.text(fCalculated BMI {bmi.2f})

# Tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs([Calories Prediction, Weight Prediction, Workout Recommendation, Weekly Plan])

with tab1
    st.header(Calories Burned Prediction)
    st.write(Predict calories burned in a session based on your inputs.)

    if calories_model is None
        st.error(Calories model not loaded. Check 'calories_model_gradient_boosting.pkl'.)
    else
        avg_bpm = st.number_input(Average BPM, min_value=60, max_value=200, value=120)
        resting_bpm = st.number_input(Resting BPM, min_value=40, max_value=100, value=70)
        session_duration = st.number_input(Session Duration (hours), min_value=0.1, max_value=5.0, value=1.0)
        workout_frequency = st.number_input(Workout Frequency (daysweek), min_value=1, max_value=7, value=4)
        experience_level = st.selectbox(Experience Level, [1, 2, 3], help=1 Beginner, 2 Intermediate, 3 Advanced)
        workout_type = st.selectbox(Workout Type, [Cardio, Strength, HIIT, Yoga])

        if st.button(Predict Calories)
            data = {
                'Age' age,
                'Weight (kg)' weight,
                'Height (m)' height,
                'Avg_BPM' avg_bpm,
                'Resting_BPM' resting_bpm,
                'Session_Duration (hours)' session_duration,
                'Fat_Percentage' fat_percentage,
                'Workout_Frequency (daysweek)' workout_frequency,
                'Experience_Level' experience_level,
                'BMI' bmi,
                'Workout_Type' workout_type
            }
            df = pd.DataFrame([data])
            pred = calories_model.predict(df)[0]
            st.success(fPredicted Calories Burned {pred.2f})

with tab2
    st.header(Future Weight Prediction)
    st.write(Predict your weight in the future based on habits. Note Model assumes starting from your current weight (fixed in training).)

    if weight_model is None
        st.error(Weight model not loaded. Check 'weight_model_gradient_boosting.pkl'.)
    else
        days_ahead = st.number_input(Days Ahead, min_value=1, max_value=365, value=30)
        steps = st.number_input(Daily Steps, min_value=1000, max_value=30000, value=10000)
        calories_burned = st.number_input(Daily Calories Burned, min_value=100, max_value=2000, value=500)
        workout_type = st.selectbox(Planned Workout Type, [Cardio, Strength, HIIT, Yoga])

        if st.button(Predict Weight)
            data = {
                'days_future' days_ahead,
                'steps' steps,
                'Calories_Burned' calories_burned,
                'Workout_Type' workout_type,
                'Weight (kg)' weight  # Added for fixed model
            }
            df = pd.DataFrame([data])
            pred = weight_model.predict(df)[0]
            st.success(fPredicted Weight in {days_ahead} days {pred.2f} kg)

with tab3
    st.header(Workout Recommendation)
    st.write(Get a personalized workout type recommendation.)

    if rec_model is None or rec_encoder is None
        st.error(Recommendation model not loaded. Check 'recommendation_gradient_boosting.pkl'.)
    else
        session_duration = st.number_input(Average Session Duration (hours), min_value=0.1, max_value=5.0, value=1.0)
        calories_burned = st.number_input(Average Calories Burned per Session, min_value=100, max_value=2000, value=500)

        if st.button(Get Recommendation)
            goal_str = goal.lower().replace( , _)
            goal_map = {lose_weight 0, gain_muscle 1, maintain 2}
            goal_encoded = goal_map.get(goal_str, 2)
            gender_encoded = 0 if gender == Male else 1

            data = {
                'BMI' bmi,
                'Fat_Percentage' fat_percentage,
                'Age' age,
                'Session_Duration (hours)' session_duration,
                'Calories_Burned' calories_burned,
                'goal' goal_encoded,
                'Gender' gender_encoded
            }
            df = pd.DataFrame([data])
            pred_encoded = rec_model.predict(df)[0]
            recommended = rec_encoder.inverse_transform([pred_encoded])[0]
            st.success(fRecommended Workout Type {recommended})

with tab4
    st.header(Weekly Workout Plan)
    st.write(Generate a personalized weekly plan based on your goal.)

    target_weight = st.number_input(Target Weight (kg), min_value=40.0, max_value=150.0, value=weight)
    days_ahead = st.number_input(Plan Horizon (days for calorie calc), min_value=7, max_value=365, value=30)
    recommended_workout = st.text_input(PreferredAI-Recommended Workout, value=HIIT)  # Can link from tab3

    if st.button(Generate Plan)
        goal_str = goal.lower().replace( , _)
        weight_diff = target_weight - weight
        total_cal = abs(weight_diff)  7700
        daily_cal_shift = total_cal  days_ahead if days_ahead  0 else 0

        # Base plans from plan_week.py
        base_plans = {
            lose_weight [
                {day Mon, workout HIIT, base_duration 45},
                {day Tue, workout Cardio, base_duration 60},
                {day Wed, workout Strength, base_duration 50},
                {day Thu, workout HIIT, base_duration 40},
                {day Fri, workout Cardio, base_duration 50},
                {day Sat, workout Yoga, base_duration 30},
                {day Sun, workout Rest, base_duration 0},
            ],
            gain_muscle [
                {day Mon, workout Strength, base_duration 60},
                {day Tue, workout Strength, base_duration 60},
                {day Wed, workout Rest, base_duration 0},
                {day Thu, workout Strength, base_duration 60},
                {day Fri, workout Strength, base_duration 60},
                {day Sat, workout Cardio, base_duration 30},
                {day Sun, workout Rest, base_duration 0},
            ],
            maintain [
                {day Mon, workout Cardio, base_duration 45},
                {day Tue, workout Yoga, base_duration 45},
                {day Wed, workout Strength, base_duration 45},
                {day Thu, workout Cardio, base_duration 45},
                {day Fri, workout HIIT, base_duration 30},
                {day Sat, workout Yoga, base_duration 45},
                {day Sun, workout Rest, base_duration 0},
            ]
        }

        plan_template = base_plans.get(goal_str, base_plans[maintain])

        week_plan = []
        for day in plan_template
            if day[workout] == Rest
                week_plan.append(day.copy())
                continue

            workout = recommended_workout if random.random()  0.6 else day[workout]

            base_dur = day[base_duration]
            if goal_str == lose_weight and weight_diff  0
                duration = int(base_dur + (abs(daily_cal_shift)  25))
            elif goal_str == gain_muscle and weight_diff  0
                duration = int(base_dur + (daily_cal_shift  30))
            else
                duration = base_dur

            duration = max(20, min(90, duration))

            week_plan.append({
                day day[day],
                workout workout,
                duration duration if duration  0 else 0
            })

        st.write(### Your Weekly Plan)
        st.table(week_plan)
        st.info(fDaily Calorie Shift for Goal {daily_cal_shift.0f} (positive = surplus for gain))
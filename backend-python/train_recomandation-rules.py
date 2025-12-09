# backend/train_recommendation_rules.py
import pandas as pd
import joblib

# Load your cleaned dataset
df = pd.read_csv('../datasets/gym_members_exercise_tracking_synthetic_data.csv')

# Clean numeric cols (same logic as before)
numeric_cols = ['Avg_BPM', 'Max_BPM', 'Session_Duration (hours)', 
                'Weight (kg)', 'Height (m)', 'BMI', 'Fat_Percentage', 'Calories_Burned']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'\\t', '', regex=True).str.strip(), errors='coerce')
df = df.dropna(subset=numeric_cols + ['Workout_Type'])

# === RULE-BASED RECOMMENDATION ENGINE ===
# We derive rules from data patterns
rules = []

# 1. Lose Weight → High calorie burn, Cardio/HIIT, longer sessions
high_cal = df[df['Calories_Burned'] > df['Calories_Burned'].quantile(0.75)]
lose_weight = high_cal['Workout_Type'].value_counts().head(2).index.tolist()
rules.append({
    "goal": "lose_weight",
    "workout_types": lose_weight,
    "min_duration": 0.75,
    "intensity": "high",
    "frequency_per_week": 5
})

# 2. Gain Muscle → Strength, moderate duration, high weight
strength = df[df['Workout_Type'] == 'Strength']
gain_muscle = ['Strength']
rules.append({
    "goal": "gain_muscle",
    "workout_types": gain_muscle,
    "min_duration": 0.6,
    "intensity": "moderate-high",
    "frequency_per_week": 4
})

# 3. Maintain → Balanced mix
maintain = ['Cardio', 'Strength']
rules.append({
    "goal": "maintain",
    "workout_types": maintain,
    "min_duration": 0.5,
    "intensity": "moderate",
    "frequency_per_week": 3
})

# Save rules
joblib.dump(rules, 'recommendation_rules.pkl')
print("Recommendation rules saved → recommendation_rules.pkl")
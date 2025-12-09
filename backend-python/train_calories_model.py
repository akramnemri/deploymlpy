# backend/train_calories_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# === 1. Load and CLEAN the CSV ===
csv_path = '../datasets/gym_members_exercise_tracking_synthetic_data.csv'
df = pd.read_csv(csv_path)

print(f"Original shape: {df.shape}")
print("First few rows:")
print(df.head())

# === 2. Clean numeric columns: remove tabs, strip, convert to float ===
numeric_cols = ['Avg_BPM', 'Max_BPM', 'Session_Duration (hours)', 
                'Weight (kg)', 'Height (m)', 'BMI', 'Fat_Percentage', 
                'Calories_Burned']

for col in numeric_cols:
    if col in df.columns:
        # Replace tabs and strip whitespace
        df[col] = df[col].astype(str).str.replace(r'\\t', '', regex=True).str.strip()
        # Convert to numeric, coercing errors
        df[col] = pd.to_numeric(df[col], errors='coerce')

# === 3. Drop rows with NaN in critical columns ===
df = df.dropna(subset=numeric_cols + ['Workout_Type'])
print(f"After cleaning: {df.shape}")

# === 4. Prepare features and target ===
X = df[['Avg_BPM', 'Max_BPM', 'Session_Duration (hours)', 
        'Weight (kg)', 'Height (m)', 'BMI', 'Fat_Percentage', 'Workout_Type']]
y = df['Calories_Burned']

# === 5. Build pipeline with OneHotEncoder ===
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Workout_Type'])
    ],
    remainder='passthrough'
)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# === 6. Train and save ===
model.fit(X, y)
joblib.dump(model, 'model.pkl')
print("Calories model trained and saved as model.pkl")
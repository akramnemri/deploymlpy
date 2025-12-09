# backend/train_calories_model_svm.py
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import numpy as np

# === 1. Load and CLEAN the CSV ===
csv_path = '../datasets/gym_members_exercise_tracking_synthetic_data.csv'
df = pd.read_csv(csv_path)

print(f"Original shape: {df.shape}")

# === 2. Clean numeric columns: remove tabs, strip, convert to float ===
numeric_cols = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM',
                'Session_Duration (hours)', 'Calories_Burned', 'Fat_Percentage', 
                'Water_Intake (liters)', 'Workout_Frequency (days/week)', 'Experience_Level', 'BMI']

for col in numeric_cols:
    if col in df.columns:
        # Replace tabs and strip whitespace, handle mixed types
        df[col] = df[col].astype(str).str.replace(r'\t', '', regex=True).str.strip()
        # Convert to numeric, coercing errors
        df[col] = pd.to_numeric(df[col], errors='coerce')

# === 2.5 Clean categorical columns ===
if 'Workout_Type' in df.columns:
    df['Workout_Type'] = df['Workout_Type'].astype(str).str.replace(r'\\t|\\n|\\r', '', regex=True).str.strip()
    df['Workout_Type'].replace(['nan', ''], np.nan, inplace=True)  # Handle 'nan' strings and empties as NaN

# === 3. Drop rows with NaN in critical columns ===
critical_cols = ['Age', 'Weight (kg)', 'Height (m)', 'Avg_BPM', 'Resting_BPM',
                'Session_Duration (hours)', 'Calories_Burned', 'Fat_Percentage', 'Workout_Frequency (days/week)', 
                'Experience_Level', 'BMI', 'Workout_Type']
df = df.dropna(subset=critical_cols)
print(f"After cleaning: {df.shape}")

# === 4. Prepare features and target ===
feature_cols = ['Age', 'Weight (kg)', 'Height (m)', 'Avg_BPM', 'Resting_BPM',
                'Session_Duration (hours)', 'Fat_Percentage', 'Workout_Frequency (days/week)', 
                'Experience_Level', 'BMI', 'Workout_Type']
X = df[feature_cols]
y = df['Calories_Burned']

# === 5. Split data for evaluation ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 6. Build pipeline with SVM ===
numeric_features = ['Age', 'Weight (kg)', 'Height (m)', 'Avg_BPM', 'Resting_BPM',
                   'Session_Duration (hours)', 'Fat_Percentage', 'Workout_Frequency (days/week)', 
                   'Experience_Level', 'BMI']
categorical_features = ['Workout_Type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', SVR(
        kernel='rbf',
        C=1000,
        gamma=0.1,
        epsilon=0.1
    ))
])

# === 7. Train and evaluate ===
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5  # Manual calculation for compatibility
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"SVM Results:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")

# === 8. Train on full dataset and save ===
model.fit(X, y)
joblib.dump(model, 'calories_model_svm.pkl')
print("Calories SVM model saved!")